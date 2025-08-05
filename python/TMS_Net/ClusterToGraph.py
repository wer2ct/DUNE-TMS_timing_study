#This script takes as input a npz file produced by CLusterDBSCAN, an edep-sim file (smh), and a merged hits file, and outputs a Graph constructed using the Delaunay Triangulation method. This graph is saved as a PyTorch Database for use in the TMS_net (better name pending)

#Kieran Wall - University of Virginia - July 2025
#I apologize to any CS folks who may have to read this

#Run - python3 ClusterToGraph "edep-file" "merged hits file" "dbscan file" "file number"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
import collections
import ROOT as root
import awkward as ak
import uproot
from array import array
import torch
import networkx as nx
from scipy.spatial import Delaunay
from torch_geometric.data import Data, Dataset
import sys
import os
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions

#pulls in the hits as well as dictionaries we generated to add information to our dbscan hits
def AddInfo(dbscan_hits, dbscan_file_dict, neutrino_vtxs_info):
    dbscan_cluster_list = []
    spills_in_file = np.unique(dbscan_hits[:,-4])
    for spill_number in spills_in_file:
        spill_level = dbscan_hits[dbscan_hits[:,-4] == spill_number]
        segments_in_spill = np.unique(spill_level[:,-2])
        for segment_number in segments_in_spill:
            time_segment_level = spill_level[spill_level[:,-2] == segment_number]
            clusters_in_segment = np.unique(time_segment_level[:,-1])
            for cluster_number in clusters_in_segment:
                dbscan_cluster_level = time_segment_level[time_segment_level[:,-1] == cluster_number]
                rows = []
                for hit in dbscan_cluster_level:
                    nn = hit[0]
                    hn = hit[1]
                    hit_info = (dbscan_file_dict[nn])[hn] #grabbing relevant hit information. 
                    hit_trackid, hit_pdgid = hit_info
                    in_tms = (neutrino_vtxs_info[int(nn)])[3]
                    addition = np.array((hit_trackid, hit_pdgid, in_tms))
                    new_row = np.hstack((hit, addition))
                    rows.append(new_row)
    
                dbscan_hits_with_info = np.vstack(rows)
                dbscan_cluster_list.append(dbscan_hits_with_info)
    
    return(dbscan_cluster_list)

#Old GraphPrep
"""
#A test dbscan cluster --> graph input ( x, z, PE , label)
def GraphPrep(dbscan_cluster_list_): #input, a list of the dbscan cluster hits, 
    number_of_clusters = len(dbscan_cluster_list_)
    formatted_clusters = [] #list of arrays containing clusters w properly formatted hits / nodes. 
    for i in range(number_of_clusters):
        hits_in_cluster = dbscan_cluster_list_[i]
        n_hits_in_cluster = len(hits_in_cluster) 
        #If we have less than 3 hits in our cluster cannot properly use Delaunay triangulation algorithm, so drop cluster. Lossy!
        if n_hits_in_cluster < 3:
            continue

        unique_xs = np.unique(hits_in_cluster[:,2])
        unique_zs = np.unique(hits_in_cluster[:,4])
        unique_nns = np.unique(hits_in_cluster[:,0])
        #this is another check I think.. will see if it clears. 
        #if n_hits_in_cluster == 3:
        if len(unique_xs) < 2 or len(unique_zs) < 2:
            continue

        if len(unique_nns) > 100: #filter out the weird outlier clusters. 
            continue
                
        #lets do a muon or non-muon semantic label. Muon/AntiMuon gets 1, else gets a 0
        nodes = []
        for hit in hits_in_cluster:
            nn = hit[0]
            x = hit[2]
            z = hit[4]
            PE = hit[6]
            pdgid = hit[-2] # hit pdgid
            in_tms = hit[-1] # did neutrino interaction which produced the hit occur in the TMS or outside?
            label = -1 
            #we can tweak the assigned labels a little bit. THis block does a full assignment, but 
            if (pdgid == 13 or pdgid == -13): #hit caused by a muon
                if (in_tms == 1): #caused by a neutrino interaction from inside the TMS
                    label = 1 #muon caused by an interaction inside the TMS
                else:
                    label = 2 #muon caused by an interaction outside the TMS
            else:
                if (in_tms == 1):
                    label = 3 #other caused by an interaction from inside the TMS
                else:
                    label = 4 #other caused by an interaction outside the TMS
            if (pdgid == 13 or pdgid == -13): #hit caused by a muon
                label = 1
            else: #hit not caused by a muon. 
                label = 0 
            
                
            nodes.append(np.array((z,x,PE,label)))

        formatted_clusters.append(np.vstack((nodes)))

    return(formatted_clusters)
"""
#The more advanced trackid-based semantic classification. 
def AssignLabels(dbscan_cluster_list, edep_events):
    #initialize 
    event = root.TG4Event()
    edep_events.SetBranchAddress("Event",root.AddressOf(event))
    #separate our cluster by trackid
    labeled_slice_list = []
    for chosen_slice in range(len(dbscan_cluster_list)):
        slice_hits = dbscan_cluster_list[chosen_slice]
        trackid_separated = []
        for nn in np.unique(slice_hits[:,0]):
            nn_sub_array = slice_hits[slice_hits[:,0] == nn]
            for trackid in np.unique(nn_sub_array[:,-3]):
                trackid_sub_array = nn_sub_array[nn_sub_array[:,-3] == trackid]
                trackid_separated.append(trackid_sub_array)
                
        #this statement need to be explicit or it can lead to errors for some reason, break into basic tracks and basic showers
        basic_tracks = []
        basic_showers = []
        for trackid_group in trackid_separated:
            if np.shape(trackid_group)[0] > 4:
                basic_tracks.append(trackid_group)
            if np.shape(trackid_group)[0] <= 4:
                basic_showers.append(trackid_group)

        #create primary track list!
        primary_track_list = []
        for track_group in basic_tracks:
            primary_track_list.append( (int(track_group[0][0]), int(track_group[0][-3])) )

        advanced_tracks = basic_tracks
        advanced_showers = []

        #local context, 
        for group in basic_showers:
            nn = group[0][0] #grab neutrino # for group
            trackid = group[0][-3] #grab the trackid for the group
            #now quickly grab the trackids of our primary track groups associated w/this event, for use
            edep_events.GetEntry(int(nn))
            event_trajectories = event.Trajectories #fetch trajectories vector
            group_traj = event_trajectories[int(trackid)] #our root trajectory 
            group_traj_parent = group_traj.GetParentId()
            parent_tuple_reference = (int(nn), int(group_traj_parent))
            added = False
    
            for i, ptref in enumerate(primary_track_list): #i will key which index of the track list to add to. 
                #check first if the parent matches
                if ptref == parent_tuple_reference:
                    #print('found a shower w/ a main track parent')
                    #print(f'{parent_tuple_reference} and {ptref}')
                    advanced_tracks[i] = (np.vstack((basic_tracks[i], group)) )
                    added = True
          
                #if not, check if the grandparent matches, to expand tracks even further. 
                if added ==  False:
                    group_traj_grandparent = event_trajectories[int(group_traj_parent)].GetParentId()
                    grandparent_tuple_reference = (int(nn), int(group_traj_grandparent))
                    if ptref == grandparent_tuple_reference:
                        advanced_tracks[i] = (np.vstack((basic_tracks[i], group)) )
                        added = True
       
                #if not check if the great grandparent matches, expands it even further!! This one may be a bit of a step far. 
        
                if added ==  False:
                    group_traj_great_grandparnet = event_trajectories[int(group_traj_grandparent)].GetParentId()
                    great_grandparent_tuple_reference = (int(nn), int(group_traj_great_grandparnet))
                    if ptref == great_grandparent_tuple_reference:
                        advanced_tracks[i] = (np.vstack((basic_tracks[i], group)) )
                        added = True
    
            if added == False:
                advanced_showers.append(group)
        #under construction
        """
        if len(advanced_tracks) > 0:
            advanced_tracks_array = np.vstack(advanced_tracks)
        if len(advanced_showers) > 0:
            advanced_showers_array = np.vstack(advanced_showers)
        """
        combined_list = advanced_tracks + advanced_showers #dodges one of them being absent
        combined_array = np.vstack(combined_list)
        
        sliced_labeled = np.column_stack(((advanced_tracks_array[:,4],advanced_tracks_array[:,2],advanced_tracks_array[:,6], np.zeros_like(advanced_tracks_array[:,0]))))
        
        #lets assign labels and concatenate, track = 0, else = 1, also collapse down to (z,x,PE,label)
        tracks_labeled = np.column_stack(((advanced_tracks_array[:,4],advanced_tracks_array[:,2],advanced_tracks_array[:,6], np.zeros_like(advanced_tracks_array[:,0]))))
        showers_labeled = np.column_stack(((advanced_showers_array[:,4],advanced_showers_array[:,2],advanced_showers_array[:,6], np.ones_like(advanced_showers_array[:,0]))))
        slice_labeled = np.vstack((tracks_labeled,showers_labeled))
        #now add to the list
        labeled_slice_list.append(slice_labeled)

        if (chosen_slice % 100 == 0):
            print(f'Assigned labels through {chosen_slice}')
        
    return(labeled_slice_list)

def RemoveDuplicates(labeled_slices_):
    filtered_slices = []
    for slice_labeled in labeled_slices_:
        #this little chat gpt tidbit efficiently sweeps for duplicates and keeps ones with label 1 
        xz = slice_labeled[:, [0, 1]]
        _, unique_indices, inverse = np.unique(xz, axis=0, return_index=True, return_inverse=True)

        # Step 2: prepare output list
        selected_indices = []

        # Step 3: for each group of same (x,z), pick one with label==0 if any
        for group_id in range(len(unique_indices)):
            group_indices = np.where(inverse == group_id)[0]
            group = slice_labeled[group_indices]
    
            # Find label==0 hit in group, if any
            label_0_indices = group_indices[slice_labeled[group_indices, 3] == 0]
            if len(label_0_indices) > 0:
                selected_indices.append(label_0_indices[0])  # pick first label==0
            else:
                selected_indices.append(group_indices[0])  # pick first if no label==0

        # Step 4: collect filtered hits
        filtered_labeled_slice = slice_labeled[selected_indices]
        filtered_slices.append(filtered_labeled_slice)
    
    return(filtered_slices)
        

#creates the graphs for the whole file and packages them into a list of Data objects
def CreateGraphList(graph_points_list_): #takes an argument of a list of arrays containing the nodes for each cluster properly formatted
    graph_list = [] #list of torch graph objects
    for i, graph_points in enumerate(graph_points_list_):
        dbscan_cluster_object = graph_points #initialize our dbscan cluster.
        #create our graph nodes and features
        node_positions = dbscan_cluster_object[:,0:2]
        node_features = dbscan_cluster_object[:,0:3]
        #node feature vector (z,x,PE)
        feature_vector = torch.tensor((node_features), dtype = torch.float)
        node_semantic_labels = torch.tensor((dbscan_cluster_object[:,3]), dtype = torch.long)
        #try-except block as a final filter for 
        try:
            tri = Delaunay(node_positions)
        except Exception as e:
            print(f"Skipping cluster {i} due to Delaunay error: {e}")
            continue
        #generate our graph simplicies
        tri = Delaunay(node_positions)
        #create the edge index object using a set to avoid repetition
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                #indexes like (u,v) where u and v are the indexes of the points making up the edge
                u = simplex[i]
                v = simplex[(i + 1) % 3] #smart little trick, thanks internet!
                edges.add((u,v)) 
                edges.add((v,u)) #we want to add both directions, since undirected.       

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous() #need to transpose since torch expects EdgeIndex like [2, # edges], we produced [# edges, 2] contiguous is a memory flag for gpu opt. 
        #now we can package into a Data object for PyG
        graph_data = Data( x = feature_vector, edge_index = edge_index, y = node_semantic_labels)

        graph_list.append(graph_data)

    return(graph_list)

#dataset class
class FileGraphDataset(Dataset):
    def __init__(self, root, file_number=None, data_list=None, transform=None, pre_transform=None):
        self.file_number = file_number
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)

        if self.data_list is None: #ie, trying to access 
            self.data_list = torch.load(self.processed_paths[0])
        else: #ie, trying to save
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(self.data_list, self.processed_paths[0])

    @property
    def processed_file_names(self):
        return( [f'file_{self.file_number}_graphs.pt'])

    def len(self):
        return( len(self.data_list) ) #we require datasets to have both a length attribute

    def get(self, idx):
        return(self.data_list[idx]) #as well as a get function to grab by index. 
    

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing
def main():
    #load everything
    print("initializing")
    DBSCAN_file = np.load(sys.argv[3]) #load dbscan hits
    dbscan_hits = DBSCAN_file['first']

    merged_hits_file = np.load(sys.argv[2]) #load merged hits
    merged_hits_array = merged_hits_file['first']
    n_neutrino_events = merged_hits_file['second']
    const_hit_array = merged_hits_file['third']

    f = root.TFile.Open(sys.argv[1]) #load edep-sim 
    edep_events = f.Get("EDepSimEvents")
    # Set the branch address.
    event = root.TG4Event()
    edep_events.SetBranchAddress("Event",root.AddressOf(event))

    file_number_ = int(sys.argv[4])
    print("loaded files")

    #Will make most sense to generate our truth label lookup table in main. This is a long block though
    """
    dbscan_file_dict = {} #holds the hit info dictionary for a given event. 
    for nn in nns_in_file:
        event_merged_hits = merged_hits_array[merged_hits_array[:,0] == nn ]
        hit_numbers = event_merged_hits[:,1]
        edep_events.GetEntry(int(nn))
        event_hit_segments = event.SegmentDetectors['volTMS']
        trackid_dict = {} #stores hit number trackid pair. (hit # : trackid)
    
        for hit_number in hit_numbers:
            hit_segment = event_hit_segments[int(hit_number)]
            contrib_vector = hit_segment.Contrib
            trackid_dict[int(hit_number)] = contrib_vector[0] #just pulling trackid of first contributor to segment 

        pdgid_dict = {} #(trackid : pdgid)
        event_trajectories = event.Trajectories
        for trackid in trackid_dict.values():
            pdgid = event_trajectories[trackid].GetPDGCode()
            pdgid_dict[trackid] = pdgid

        #now create a different dictionary - call it info dict. looks like (hit # : (trackid, pdgid) ). 
        hit_info_dict = {}
        for hit_number in hit_numbers:
            hit_trackid = trackid_dict[hit_number]
            hit_pdgid = pdgid_dict[hit_trackid]
            hit_info_dict[int(hit_number)] = (hit_trackid, hit_pdgid)

        dbscan_file_dict[int(nn)] = hit_info_dict

    """
    #large block which grabs dictionaries necessary for labels. 
    dbscan_file_dict = {}
    nns_in_file = np.unique(dbscan_hits[:,0])
    print("starting creation of necessary dictionaries, this step can be long!")
    for nn in nns_in_file:
        event_merged_hits = merged_hits_array[merged_hits_array[:,0] == nn ]
        const_hit_array_nn = const_hit_array[const_hit_array[:,0] == nn]

        np.unique(event_merged_hits[:,1]) #hit numbers
        edep_events.GetEntry(int(nn))
        event_hit_segments = event.SegmentDetectors['volTMS']
        event_trajectories = event.Trajectories

        trackid_dict = {}
        pdgid_dict = {}
        #collapsed the pdgid dict into this one too. 
        #print(f'{nn} \n \n')
        exceptions_list = []
        for hit_num in np.unique(event_merged_hits[:,1]):
            const_hit_array_nn[const_hit_array_nn[:,1] == hit_num] #grab const hn
            const_hit_info = const_hit_array_nn[const_hit_array_nn[:,1] == hit_num] #grab const hit numbers
            #grab the default
            merged_hit_segment = event_hit_segments[int(hit_num)]
            default_trackid = (merged_hit_segment.Contrib)[0]
            default_traj = event_trajectories[default_trackid] 
            default_pdgid = default_traj.GetPDGCode()
            #now loop over the other possible constituent hits, in the case that not already a muon hit
            #print(hit_num)
            #print(hit_num)
            if abs(default_pdgid) != 13:
                for i, const_hit_num in enumerate(const_hit_info[:,-1]):
                    #exception handling in the case of multi nn merge is going to be annoying, send to end!
                    if const_hit_info[i][-2] != nn:
                        exceptions_list.append(np.array((hit_num, const_hit_num, const_hit_info[i][-2], default_trackid)))
                        #print("mulit neutrino weirdness, sending to exceptions stack")
                        #print(f"to check, nn {nn}: {(hit_num, const_hit_num, const_hit_info[i][-2], default_trackid)}")
                        #we add here the merged hit #, constituent hit #, the constituent nn, and the trackid at the instance. 
                        continue
                        
                    #print(const_hit_num)
                    hit_segment = event_hit_segments[int(const_hit_num)]
                    contrib_vector = hit_segment.Contrib #grab associated trackid
                    associated_traj = event_trajectories[contrib_vector[0]] 
                    replaced = False
                    if abs(associated_traj.GetPDGCode()) == 13 and replaced == False:
                        default_trackid = contrib_vector[0] #if a muon, prioritize
                        default_pdgid = associated_traj.GetPDGCode()
                        replaced = True
                        #print(f'Reassigned, now has trackid {default_trackid}')

            trackid_dict[int(hit_num)] = default_trackid
            pdgid_dict[default_trackid] = default_pdgid
            #print(f'confirming {default_trackid}')

        #handle the issues (mulit neutrino events)
        for exception in exceptions_list:
            #check if we need to handle in the first place
            if abs(pdgid_dict[int(exception[3])]) != 13:
                #set the events to 
                edep_events.GetEntry(int(exception[2]))
                event_hit_segments = event.SegmentDetectors['volTMS']
                event_trajectories = event.Trajectories
                trackid = ((event_hit_segments[int(exception[1])]).Contrib)[0] #grab the segment
                #update if needed. Feel this will be fairly rare. 
                if abs(event_trajectories[trackid].GetPDGCode()) == 13:
                    trackid_dict[int(exception[0])] = trackid
                    pdgid_dict[trackid] = event_trajectories[trackid].GetPDGCode()
                    #print(f'Wow actually had to update something in exceptions! - {exception}')
        
        
        hit_info_dict = {}
        for hit_num in np.unique(event_merged_hits[:,1]):
            hit_trackid = trackid_dict[hit_num]
            hit_pdgid = pdgid_dict[hit_trackid]
            hit_info_dict[int(hit_num)] = (hit_trackid, hit_pdgid)

        dbscan_file_dict[int(nn)] = hit_info_dict 

    
    #This grabs in vs out of TMS based on position (this may need to be updated!!! - check at meeting) 
    edep_detsim = f.Get("DetSimPassThru") #grab detsim, should contain the gRooTracker
    tracker_tree = edep_detsim.gRooTracker 

    edep_true_neutrino_vtx = [] 
    vtxs = array('d', [0.0]*5) 
    tracker_tree.SetBranchAddress("EvtVtx", vtxs)
    neutrino_vtxs_info = {}
    for i in range(tracker_tree.GetEntries()):
        tracker_tree.GetEntry(i)
        #don't forget to scale positions (tracker inexplicably uses m)
        x = vtxs[0]*1000
        y = vtxs[1]*1000
        z = vtxs[2]*1000
        in_tms = 0 #default is not in the tms
        if (11185 <= z <= 18535): #check if in z range of tms
            if (-3730 <= x <= 3730): #check if in x range of tms (?)
                if (-2350 <= y <= 2350): #check if in y range of tms (?)
                    in_tms = 1
        neutrino_vtxs_info[i] = (vtxs[0]*1000,vtxs[1]*1000,vtxs[2]*1000, in_tms) #x, y, z, in_tms (0 if false, 1 if true)
            
    print("finished dictionary creation!")
    
    #pull dictionary information to assign labels. Then make graphs
    dbscan_cluster_list_ = AddInfo(dbscan_hits, dbscan_file_dict, neutrino_vtxs_info) #adding info
    print("starting to format into nodes and assigning labels")
    graph_points_list = AssignLabels(dbscan_cluster_list_, edep_events) #formatting into graphs w/ the new segmentation. 
    print("completed nodes")
    print("starting to create graphs")
    full_file_graph_list = CreateGraphList(graph_points_list)
    print("completed graphs")

    #save to our dataset
    file_graph_dataset = FileGraphDataset(root='/sdf/data/neutrino/summer25/ktwall/', data_list = full_file_graph_list, file_number = file_number_)
    print(f"Graph Dataset for file {file_number_} has been saved")


main()

    
    
    
    












