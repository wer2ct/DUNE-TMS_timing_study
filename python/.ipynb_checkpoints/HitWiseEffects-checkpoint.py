#This script takes as input an edep-sim file and outputs a numpy array file with hit-wise TMS detector effects applied

#Kieran Wall - University of Virginia - June 2025
#I apologize to any CS folks who may have to read this

#Run - python3 HitWiseEffects.py "edep_sim_file" "output directory"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import math
import ROOT as root
from array import array
from numba import jit #sorta optional (TODO - Implement toggling for numba optimization)
from collections import defaultdict
import sys

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Classes (there are a few)

#TMS_hit_simple class is the initial class that edep-sim hitsegments get tossed into. This handles the center of hit segments and stores other information
# that it would be tedious to have to load in every time. 
class TMS_hit_simple:
    def __init__(self, tms_hit_seg, neutrino_number, hit_number):
        self.tms_hit_seg = tms_hit_seg
        self.neutrino_number = neutrino_number #associated with what neutrino vertex
        self.hit_number = hit_number #which TMS hit in the event
        self.x_diff = 0.
        self.y_diff = 0.
        self.z_diff = 0.
        self.Dx = 0.
        
    #Defining attributes based on the tms_hit_seg, avg position in each dimension plus time and energy deposition
    def GetNeutrinoNumber(self):
        return(self.neutrino_number)
        
    def GetHitNumber(self):
        return(self.hit_number)
        
    def GetHitX(self):
        x_pos = (self.tms_hit_seg.GetStart()[0] + self.tms_hit_seg.GetStop()[0]) / 2 
        return(x_pos)
        
    def GetHitY(self):
        y_pos = (self.tms_hit_seg.GetStart()[1] + self.tms_hit_seg.GetStop()[1]) / 2
        return(y_pos)
        
    def GetHitZ(self):
        z_pos = (self.tms_hit_seg.GetStart()[2] + self.tms_hit_seg.GetStop()[2]) / 2
        return(z_pos)
        
    def GetHitT(self):
        time = (self.tms_hit_seg.GetStart()[3] + self.tms_hit_seg.GetStop()[3]) / 2
        return(time)
        
    def GetDx(self):
        self.x_diff = (self.tms_hit_seg.GetStart()[0] - self.tms_hit_seg.GetStop()[0])
        self.y_diff = (self.tms_hit_seg.GetStart()[1] - self.tms_hit_seg.GetStop()[1])
        self.z_diff = (self.tms_hit_seg.GetStart()[2] - self.tms_hit_seg.GetStop()[2])
        self.Dx = ((self.x_diff)**2 + (self.y_diff)**2 + (self.z_diff)**2)**(1/2)
        return(self.Dx)
        
    def GetPrimaryDeposit(self):
        PrimaryDeposit = self.tms_hit_seg.GetEnergyDeposit()
        return(PrimaryDeposit)



#TMS_hit_reco is where the effects of our detector sim is stored. It inherits directly from a TMS_hit_simple. TODO - write this using SUPERs to make inheritance easier lol
class TMS_hit_reco:
    def __init__(self, tms_hit_simple, geo):
        self.tms_hit_simple = tms_hit_simple
        self.geo = geo
        #Run the ModuleFinder() function immediately
        self.widths, self.bar_orientation, self.bar_no, self.layer_no, self.bar_positions = ModuleFinder(self.tms_hit_simple.GetHitX(), self.tms_hit_simple.GetHitY(), self.tms_hit_simple.GetHitZ() , self.geo)
        
    #Attributes which inherit directly from tms_hit_simple!!
    
    #We will want to include neutrino number here - directly inherited from the other class. Will probably often be just dealing with these hits. 
    def GetNeutrinoNumber(self):
        return(self.tms_hit_simple.GetNeutrinoNumber())

    def GetHitNumber(self):
        return(self.tms_hit_simple.GetHitNumber())
        
    def GetTrueHitPosition(self):
        return ( (self.tms_hit_simple.GetHitX(), self.tms_hit_simple.GetHitY(), self.tms_hit_simple.GetHitZ()) )
    
    def GetTrueHitT(self):
        return(self.tms_hit_simple.GetHitT())

    def GetDx(self):
        return(self.tms_hit_simple.GetDx())
        
    #One final inheritance - the true hit primary deposition. 
    
    def GetTrueHitPrimDeposition(self):
        return(self.tms_hit_simple.GetPrimaryDeposit())

    def GetRecoPE(self):
        #note, the 0th index here is what we generally want, the other two are for detector sim calls. 
        initial_PE = self.tms_hit_simple.GetPrimaryDeposit() * 50.0  #E to PE with conversion 
        suppressed_PE = BirkSuppress( self.tms_hit_simple.GetPrimaryDeposit(), self.GetDx(), initial_PE ) #apply BirkSuppression
        reco_PE, short_pe, long_pe = FiberLengthSim(suppressed_PE, self)
        return(reco_PE, short_pe, long_pe)
        
    def GetPedestalSubtractedStatus(self): #true if hit is pedestal subtracted
        if (self.GetRecoPE()[0] < 3.0):
            return(True)

        else:
            return(False)
        
    #Attributes which are dependent on the bar segmentation
    
    def GetBarHitPos(self):
        return(self.bar_positions)
        
    def GetBarNo(self):
        return(self.bar_no)
        
    def GetBarLayer(self):
        return(self.layer_no)  

    def GetBarOrientation(self):
        return(self.bar_orientation)
        
    #this will be updated with timing info!
    def GetRecoHitT(self):
        return(HitTimingSim(self))
        

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions (there are many)

#This one is self explanatory - creates vertex container objects
def CreateVtxContainers(tracker_tree):

    #Create dictionary with vertex information - for easy use in other parts of the script
    edep_true_neutrino_vtx = [] 
    vtxs = array('d', [0.0]*4) 
    tracker_tree.SetBranchAddress("EvtVtx", vtxs)
    for i in range(tracker_tree.GetEntries()):
        tracker_tree.GetEntry(i)
        #don't forget to scale positions (tracker inexplicably uses m)
        vtx_data = {
            "neutrino_num": i,
            "x": vtxs[0]*1000,
            "y": vtxs[1]*1000,
            "z": vtxs[2]*1000,
            "t": vtxs[3],
        }
        edep_true_neutrino_vtx.append(vtx_data)

    #Create an array with vertex information - for saving as output.
    neutrino_number = []
    neutrino_x = []
    neutrino_y = []
    neutrino_z = []
    neutrino_t = []
    for entry in edep_true_neutrino_vtx:
        neutrino_number.append(entry['neutrino_num'])
        neutrino_x.append(entry['x'])
        neutrino_y.append(entry['y'])
        neutrino_z.append(entry['z'])
        neutrino_t.append(entry['t'])

    neutrino_vertex_array = np.column_stack((neutrino_number,neutrino_x,neutrino_y,neutrino_z,neutrino_t))

    return(edep_true_neutrino_vtx , neutrino_vertex_array) #output - CreateVtxContainers()[0] = dictionary, CreateVtxContainers()[1] = array

#Converts our reco hits into an array, this is one of the two arrays that actually leaves this script, so add things here.
#this requires the TMS event processor to have been run!
def RecoToArray(reco_hits):
    reco_hit_xs = []
    reco_hit_zs = []
    reco_hit_ys = []
    reco_hit_ts = []
    reco_hit_nn = []
    reco_hit_pe = []
    reco_hit_hn = []
    reco_hit_bar_orient = []
    for hit in reco_hits:
        if not hit['pedestal_subtracted?']:
            reco_hit_xs.append(hit['hit_reco_position'][0])
            reco_hit_ys.append(hit['hit_reco_position'][1])
            reco_hit_zs.append(hit['hit_reco_position'][2])
            reco_hit_ts.append(hit['hit_reco_T'])
            reco_hit_nn.append(hit['neutrino_num'])
            reco_hit_hn.append(hit['hit_num'])
            reco_hit_pe.append(hit['hit_reco_PE'][0])
            reco_hit_bar_orient.append(hit['hit_reco_bar_orient'])


    array = np.column_stack((reco_hit_nn, reco_hit_hn, reco_hit_xs, reco_hit_ys, reco_hit_zs, reco_hit_ts, reco_hit_pe, reco_hit_bar_orient))
    return(array)

#super important function that handles the geometry finding. 
def ModuleFinder(x,y,z,geom):
    module_names = {'modulelayervol1' : 'U' ,'modulelayervol2' : 'V' ,'modulelayervol3': 'X' ,'modulelayervol4' : 'Y'}
    node = geom.FindNode(x, y, z)
    orientation = 'null'
    bar_number = 'null'
    layer_number = 'null'
    widths = [0,0,0]
    bar_positions = array('d', [x, y, z]) #hold bar positions as just hit x,y,z for now bar_positions = array('d', [0.0, 0.0, 0.0])
    while node:
        #print(node.GetName())
        #print(node.GetNumber())
        if 'scinBox' in node.GetName(): #this is like a rlly stupid line but let's just hope it works
            bar_number = node.GetNumber()
            box = geom.GetCurrentVolume().GetShape()
            local = array('d', [0.0, 0.0, 0.0])  # 'd' = C double
            geom.GetCurrentMatrix().LocalToMaster(local, bar_positions) #assigning the center of the bar to the hit -- basically just PyROOTED dune-tms code.
            widths[0] = 2 * box.GetDX()
            widths[1] = 2 * box.GetDY()
            widths[2] = 2 * box.GetDZ()
        if "modulelayervol" in node.GetName():
            layer_number = node.GetNumber()
            for module_name in module_names.keys():
                if module_name in node.GetName():
                    orientation = module_names[module_name]
                    #print(module_name)
        if 'volDetEnclosure' in node.GetName():
            break
        geom.CdUp()
        node = geom.GetCurrentNode()

    if orientation == 'X':
        bar_positions[0] = -99999000  # remove X information -- irrelevant given geometry
        xw, yw = yw, xw               # flip widths --apparently root handles incorrectly for x-bars
    elif orientation in ['U', 'V', 'Y']:
        bar_positions[1] = -99999000  # remove Y information -- irrelevant given geometry
    else:
        bar_positions = [-99999000, -99999000, -99999000]

    return (widths, orientation, bar_number, layer_number, bar_positions)

# ------ Functions pertaining to the Optical Simulation ------- #

#TODO - just make all these effect functions and class definitions their own python script (for neatness)
rand_seed = 42 #this is technically a global variable 
def GetTrueDistanceFromReadout(TMS_hit_reco):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    #XBars are weird
    if (TMS_hit_reco.GetBarOrientation == 'X'):
        if (TMS_hit_reco.GetTrueHitPosition()[0] < 0):
            return(TMS_hit_reco.GetTrueHitPosition()[0] - TMS_Start_Bars_Only[0])
        else:
            return(TMS_End_Bars_Only[0] - TMS_hit_reco.GetTrueHitPosition()[0])
    #All other bar orientations (U,V,Y)
    else:
        return(TMS_End_Bars_Only[1] - TMS_hit_reco.GetTrueHitPosition()[1])

def GetTrueLongDistanceFromReadout(TMS_hit_reco):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    additional_length = 0.
    #XBars
    if (TMS_hit_reco.GetBarOrientation == 'X'):
        additional_length = 2 * TMS_End_Bars_Only[0] #2 * XBar Length
    #All other bar orientations (U,V,Y)
    else:
        additional_length = 2 * (TMS_End_Bars_Only[1] - TMS_Start_Bars_Only[1]) #2* YBar Length
    return( additional_length - GetTrueDistanceFromReadout(TMS_hit_reco) )

def GetTrueDistanceFromMiddle(TMS_hit_reco):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    additional_length = 0.
    #XBars
    if (TMS_hit_reco.GetBarOrientation == 'X'):
        additional_length = 0.5 * TMS_End_Bars_Only[0]
    #All other bar orientations (U,V,Y)
    else:
        additional_length = 0.5 * (TMS_End_Bars_Only[1] - TMS_Start_Bars_Only[1])
    return( GetTrueDistanceFromReadout(TMS_hit_reco) - additional_length )
        
def GetTrueLongDistanceFromMiddle(TMS_hit_reco):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    additional_length = 0.
    #XBars
    if (TMS_hit_reco.GetBarOrientation == 'X'):
        additional_length = 0.5 * TMS_End_Bars_Only[0]
    #All other bar orientations (U,V,Y)
    else:
        additional_length = 0.5 * (TMS_End_Bars_Only[1] - TMS_Start_Bars_Only[1])
    return( GetTrueLongDistanceFromReadout(TMS_hit_reco) - additional_length )

#Models the effect of birk suppression on the total PE
def BirkSuppress(de, dx, pe): #de is energy
    birks_constant = 0.0905
    dedx = 0
    if (dx > 1e-8):
        dedx = de / dx
    else: 
        dedx = de / 1.0;
    return(pe * (1.0 / (1.0 + birks_constant * dedx)))

#Selects PEs as traveling in the long or short directions - then models the effect on the PE for traveling the long or short distance. Returns corrected PE
def FiberLengthSim(pe, TMS_hit_reco):
    #Constant used
    wsf_decay_constant = 1 / 4.160
    wsf_length_multiplier = 1.8 
    wsf_fiber_reflection_eff = 0.95
    readout_coupling_eff = 1.0
    
    #Select the PEs traveling in either direction. 
    rng = np.random.default_rng(seed = rand_seed)
    pe_short = pe #what is number of pe's that travel the short path
    pe_long = 0 #what is the number of pe's that travel the long path
    pe = rng.poisson(pe)
    pe_short = rng.binomial(n = pe, p = 0.5)
    pe_long = pe - pe_short
    
    #these next calculations are done in meters to make it a bit easier, hence the conversions, models attenuation. 
    distance_from_end = GetTrueDistanceFromReadout(TMS_hit_reco) * 1e-3
    long_way_distance_from_end = GetTrueLongDistanceFromReadout(TMS_hit_reco) * 1e-3
    distance_from_end *= wsf_length_multiplier
    long_way_distance_from_end *= wsf_length_multiplier

    #attentuation
    pe_short = pe_short * math.exp(-wsf_decay_constant * distance_from_end)
    pe_long = pe_long * math.exp(-wsf_decay_constant * long_way_distance_from_end) * wsf_fiber_reflection_eff
    
    #could couple to additional optical fiber but we are going to go ahead and neglect that - default setting is to neglect. 
    pe_short *= readout_coupling_eff
    pe_long *= readout_coupling_eff

    #return the PE post attenuation, as well as some other useful bits of information for other functions 
    return( ( (pe_short + pe_long), pe_short, pe_long ) )

    
# ------ Functions pertaining to the Hit-Wise Timing Simulation ------- #

def HitTimingSim(TMS_hit_reco):
    rng = np.random.default_rng(seed=rand_seed)
    #Constant
    SPEED_OF_LIGHT = 0.2998  # m/ns
    FIBER_N = 1.5
    SPEED_OF_LIGHT_IN_FIBER = SPEED_OF_LIGHT / FIBER_N
    scintillator_decay_time = 3.0  # ns
    wsf_decay_time = 20.0  # ns
    wsf_length_multiplier = 1.8 

    # Gaussian noise: mean = 0.0, std dev = 1.0 ns
    noise_distribution = lambda: rng.normal(loc=0.0, scale=1.0)

    # Coin flip: returns 0 or 1
    coin_flip = lambda: rng.integers(low=0, high=2)  # high=2 is exclusive

    # Exponential decay distributions (rate = 1 / decay time)
    exp_scint = lambda: rng.exponential(scale=scintillator_decay_time)  # mean = 3 ns
    exp_wsf = lambda: rng.exponential(scale=wsf_decay_time)             # mean = 20 ns

    #begin determining timing effect t here is the added time. 
    t = 0.
    t += noise_distribution()
    distance_from_middle = (GetTrueDistanceFromMiddle(TMS_hit_reco) * 1e-3) * wsf_length_multiplier
    long_way_distance = (GetTrueLongDistanceFromMiddle(TMS_hit_reco) * 1e-3) * wsf_length_multiplier
    time_correction = distance_from_middle / SPEED_OF_LIGHT_IN_FIBER;
    time_correction_long_way = long_way_distance / SPEED_OF_LIGHT_IN_FIBER;

    pe_short_path = TMS_hit_reco.GetRecoPE()[1]
    pe_long_path = TMS_hit_reco.GetRecoPE()[2]
    minimum_time_offset = 1e100

    MAX_PE_THROWS = 300

    #serves to cap the number of throws we do (little difference past 300)
    if (pe_short_path > MAX_PE_THROWS):
        pe_short_path = MAX_PE_THROWS
    while (pe_short_path > 0):
        time_offset = time_correction
        time_offset += exp_scint()
        time_offset += exp_wsf()
        minimum_time_offset = min(time_offset, minimum_time_offset) #take whatever is less
        pe_short_path -= 1

    #serves to cap the number of throws that we do (little difference past 300)
    if (pe_long_path > MAX_PE_THROWS):
        pe_long_path = MAX_PE_THROWS
    
    while (pe_long_path > 0):
        time_offset = time_correction_long_way
        time_offset += exp_scint()
        time_offset += exp_wsf()
        minimum_time_offset = min(time_offset, minimum_time_offset) #take whatever is less
        pe_long_path -= 1

    t += minimum_time_offset
    return (t + TMS_hit_reco.GetTrueHitT()) #return the true hit time + the determined travel effects.

#The core processing function of this script, takes an edep-sim events tree as an argument and uses all the stuff we've created to generate an output dictionary of truth 
#and rec (detector effects applied) hits. 
def TMS_Event_Processor(edep_evts, geometry, n_events = 10, verbose = False):
    
    edep_evt = root.TG4Event()
    edep_evts.SetBranchAddress("Event",root.AddressOf(edep_evt)) #I think it is ok to set branch address here, guess we will find out!
    true_hits_info = []
    reco_hits_info = []
    tally = 0.
    for n in range(n_events):
        edep_evts.GetEntry(n) #the nth event, associated with the nth neutrino vertex
        hit_segments = edep_evt.SegmentDetectors['volTMS'] #fetching the hit_segment vector for this event
        if hit_segments.size() > 0: #checking the size of the hit segment, make sure we have TMS events 
            if verbose:
                print(f"{hit_segments.size()} TMS hits found in neutrino interaction number {n}, creating truth and rec hit objects") 
            tally += 1 
            for i in range(hit_segments.size()):
                hit_true = TMS_hit_simple(hit_segments[i], n, i) #takes the hit segment, the neutrino number, and the hit number
                hit_reco = TMS_hit_reco(hit_true, geometry)
                bar_orientation = hit_reco.GetBarOrientation()
                bar_orientation_coded = -1
                #this is silly but to save to array need to have our orientation coded to a number 0 = U, 1 = Y, 2 = X, -1 = Other (??)
                if bar_orientation == 'U':
                    bar_orientation_coded = 0
                elif bar_orientation == 'V':
                    bar_orientation_coded = 1
                elif bar_orientation == 'X':
                    bar_orientation_coded = 2
                else:
                    bar_orientation_coded = -1
                
                #Fill the truth 
                hit_truth_data = {
                    "neutrino_num": hit_true.GetNeutrinoNumber(),
                    "hit_num": hit_true.GetHitNumber(),
                    "x": hit_true.GetHitX(),
                    "y": hit_true.GetHitY(),
                    "z": hit_true.GetHitZ(),
                    "t": hit_true.GetHitT(),
                    "primary_energy": hit_true.GetPrimaryDeposit(),
                    }
                hit_reco_data = {
                    "neutrino_num": hit_reco.GetNeutrinoNumber(),
                    "hit_num": hit_reco.GetHitNumber(),
                    "hit_reco_position": hit_reco.GetBarHitPos(),
                    "hit_reco_PE": hit_reco.GetRecoPE(),
                    "pedestal_subtracted?" : hit_reco.GetPedestalSubtractedStatus(),
                    "hit_true_T" : hit_reco.GetTrueHitT(),
                    "hit_reco_T" : hit_reco.GetRecoHitT(),
                    "hit_reco_bar_orient" : bar_orientation_coded,
                    }
                true_hits_info.append(hit_truth_data)
                reco_hits_info.append(hit_reco_data)
            #a little print output to tell us how processing is going
            if ((tally % 10) == 0):
                print(f"Processed {tally} events")
        else:
            if verbose:
                print(f"NO TMS hits found in neutrino interaction number {n}")
        
    return( (true_hits_info , reco_hits_info ) )


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing. 
def main():
    print("initializing")
    #Initializing
    edep_file = root.TFile(sys.argv[1]) #grabbing the input file (ex. "/sdf/home/t/tanaka/MicroProdN4p1_NDComplex_FHC.spill.full.0002459.EDEPSIM_SPILLS.root")
    output_dir = str(sys.argv[2])
    
    geom = edep_file.Get("EDepSimGeometry") #fetching the geometry from the edep_file
    edep_evts = edep_file.Get("EDepSimEvents") #fetching the events tree (contains the hit segments, etc)
    total_events = edep_evts.GetEntries()
    print(f"Found {total_events} events in the edep-sim file")

    edep_detsim = edep_file.Get("DetSimPassThru") #grab detsim, should contain the gRooTracker
    gRooTrackerTree = edep_detsim.gRooTracker #the gRooTracker contains neutrino vertex information

    #Let's generate the neutrino vertex containers
    neutrino_vtx_dict, neutrino_vtx_array = CreateVtxContainers(gRooTrackerTree)
    print("Fetched neutrino vertices")

    #Let's do the detector simulation
    print("About to run detector sim, this could take up to 10 minutes!")
    events_info = TMS_Event_Processor( edep_evts, geom, n_events = total_events) #TODO - allow custom number of events to be run. 
    print("Successfully finished the detector sim!!")
    truth_hits_info = events_info[0]
    detectorsim_hits_info = events_info[1]

    #Let's package things nicely for output
    detsim_events_array = RecoToArray(detectorsim_hits_info)
    print(f"Output hits array has shape: {np.shape(detsim_events_array)}, Output vtx array has shape: {np.shape(neutrino_vtx_array)}")

    #this line is going to need to be tweaked depending on where we are pulling out files! 
    output_tag = str(edep_file)[64:71]
    output_file_path = output_dir + 'hitwise_detector_sim_' + output_tag + '.npz'
    
    print(f"Saving with file and path: {output_file_path}")
    

    #Now finally, save our output to an npz file for the next step!
    np.savez(output_file_path, first = neutrino_vtx_array, second = detsim_events_array)

    print("File saved!")

main()

    

    

























    
    
    
    