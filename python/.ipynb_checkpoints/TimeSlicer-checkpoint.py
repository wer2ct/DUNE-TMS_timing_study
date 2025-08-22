#This script takes as input a npz file produced by MultiHitEffects, a file number, time slicer thresholds, and outputs a .root file with the time slices and some basic metrics on the slicer performance

#For deeper analysis and multifile, need to develop an analysis specific jupyter notebook. 

#Kieran Wall - University of Virginia - July 2025
#I apologize to any CS folks who may have to read this

#Run - python3 TimeSlicer.py "input file" "output dir" "file number" "init_thresh" "drop_thresh"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
import collections
import ROOT as root
import sys
import awkward as ak
import uproot
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions

#Some useful sorting functions
def SortByTime(array):
    sorted_array = array[np.argsort(array[:,5])]
    return(sorted_array)

def SortByNN(array):
    sorted_array = array[np.argsort(array[:,0])]
    return(sorted_array)

#returns an array which contains total merged hits, PEs, for each neutrino event, used for creating metrics.
def ReturnNeutrinoInfoArray(time_sorted_hits, number_of_neutrinos):
    
    neutrino_PE_array_merged = np.zeros((number_of_neutrinos, 3))
    for i in range(number_of_neutrinos):
        PEs = 0.
        sub_array = time_sorted_hits[time_sorted_hits[:, 0] == i]
        n_hits = 0
        for hit in sub_array:
            n_hits += 1
            PEs += hit[6]
        neutrino_PE_array_merged[i][0] = i
        neutrino_PE_array_merged[i][1] = PEs
        neutrino_PE_array_merged[i][2] = n_hits

    return(neutrino_PE_array_merged)


#A time slicer based on the principles of the dune-tms slicer. takes thresholds and a hits array. I think this is working as intended. 
#This function will return a list of lists of arrays. lol
#heavily commented since this took a while for me to ensure it was doing what I wanted it to do! 
#An open issue/question is what to do with hits that are not grouped by the slicer!
def SimpleTimeSlicer(hits, initial_threshold_energy=50, drop_threshold_energy=40):
    sorted_hits = SortByTime(hits)
    events = []  # This object stores the final "events" or slices that are grouped by the algorithm
    window_duration_ns = 19 #maybe should make this adjustable?

    current_window_hits = collections.deque()
    current_window_energy_sum = 0.0

    in_accumulation_phase = False
    current_event_hits = []  # Stores hit info for our event that is accumulating

    for hit_row in sorted_hits:  # Iterate through each full hit row
        hit_time = hit_row[5]    
        hit_energy = hit_row[6]  

        # Add the current full hit row to the window
        current_window_hits.append(hit_row)  # Store the entire row
        current_window_energy_sum += hit_energy  # Add the hit energy to our current window

        # Remove hits that are outside the current window, basically sliding the deque forward
        # The window is defined by (current_hit_time - window_duration_ns, current_hit_time]
        while current_window_hits and current_window_hits[0][5] <= hit_time - window_duration_ns:
            # This loop removes all hits from the front of the window that are outside 
            # the time range (older than hit_time - window_duration_ns)
            oldest_hit_row = current_window_hits.popleft()  # Get the full oldest row
            oldest_hit_energy = oldest_hit_row[6]  # Access energy from the row
            current_window_energy_sum -= oldest_hit_energy

        # Handle accumulation logic
        if not in_accumulation_phase:
            # Not in accumulation, if we pass the threshold we should start
            if current_window_energy_sum >= initial_threshold_energy:
                in_accumulation_phase = True  # Here we activate accumulation mode - begin a slice.
                current_event_hits = list(current_window_hits)
        else:
            # We ARE currently in an accumulation phase
            # Check if the energy in the *current window* has dropped below the threshold,
            # BEFORE adding the most recent hit to the accumulating event
            if current_window_energy_sum < drop_threshold_energy:
                # If indeed we have dropped below threshold, go ahead and 
                events.append(current_event_hits)
                in_accumulation_phase = False  # Turning off accumulation mode
                current_event_hits = []  # Reset for the next event - dumping the current event's hit list
            else:
                # Still above threshold, so include the current hit
                current_event_hits.append(hit_row)

    # After iterating through all hits, check if we were still in an accumulation phase.
    # If so, the last event needs to be saved, it won't do this automatically!
    if in_accumulation_phase:
        events.append(current_event_hits)

    return events

#this function checks how many total events were placed into groups. 
def ValidateEvents(events, hit_array_merged):
    hits_sliced = 0
    for group in events:
        for hit in group:
            hits_sliced += 1
    missing = (np.shape(hit_array_merged)[0] - hits_sliced)
    missing_ratio = missing / np.shape(hit_array_merged)[0]
    print(f"Grouped {hits_sliced} hits into {len(events)} slices, {missing} hits were not grouped, {(missing_ratio * 100):2f}% of total")
    return(hits_sliced, missing, missing_ratio)

#this function will return a list of hits grouped into arrays based on their event. May incorporate into main function...
#then individual slices can be accessed by their index. 
def StackEvents(events):
    event_list = []
    for i in range(len(events)):
        event_list.append(np.vstack(events[i]))
    return(event_list)
            
#This function can print out the time slices, caution it prints a lot lol
def SlicerPrinting(events, loud = False):
    print(f"Detected {len(events)} slices:")
    for i, event in enumerate(events):
        print(f"\n--- Event {i+1} ---")
        total_energy_event = sum(hit_row[6] for hit_row in event) # Sum energy from full rows
        print(f"Total hits: {len(event)}")
        print(f"Total energy: {total_energy_event:.2f}")
        # Get min/max time from the full hit rows in the event
        if event: # Ensure event is not empty before accessing elements
            start_time = min(hit_row[5] for hit_row in event)
            end_time = max(hit_row[5] for hit_row in event)
            print(f"Time range: {start_time:.1f} ns - {end_time:.1f} ns")
            print(f"Total time: {(end_time - start_time):.2f} ns")
            
        if loud:
            print("Hits (full data) in event:")
            for hit_row in event:
                # Print specific fields for clarity, or the whole row
                print(f"  Neutrino: {hit_row[0]}, Hit: {hit_row[1]}, Time: {hit_row[5]:.1f}, Energy: {hit_row[6]:.1f}, X: {hit_row[2]}")

def SliceEvaluate(events, number_of_neutrinos, neutrino_PE_array_merged):
    slice_stack = StackEvents(events)
    #make sure you are careful about how assigning here, 
    containment_array = np.zeros((number_of_neutrinos,5))
    #containment_array = np.full((5978,5), -1.) #here, I am assigning -1, will tell us if the neutrino event is just never picked up by the slicer. 
    #if we filled with zeroes could mistake for something that was picked up but at very low efficiency %, a different thing!!

    containment_array = np.zeros((number_of_neutrinos,5)) #|nn | hit containment | PE containment | best hit containment slice | best PE containment slice |
    purity_array = np.zeros((len(slice_stack),2))

    for i in range(number_of_neutrinos):
        containment_array[i][0] = i

    for j in range(len(slice_stack)):
        slice_number = j
        selected_slice = slice_stack[j]
        nns = selected_slice[:,0] #grab the neutrino numbers
        nns_represented, counts = np.unique(nns, return_counts = True) #counts can be used to determine the hit containment, need to loop for PEs

        #fill purity array
        purity_array[slice_number][0] = slice_number
        purity_array[slice_number][1] = len(nns_represented) #fill our purity array with number of unique neutrino numbers in the slice. 
    
        for i, nn in enumerate(nns_represented):
            total_hits = neutrino_PE_array_merged[int(nn)][2] #total hits
            total_PEs = neutrino_PE_array_merged[int(nn)][1] #total PE
            hits_in_slice = counts[i] #hits in this slice
            filtered_rows = (slice_stack[slice_number][slice_stack[slice_number][:, 0] == nn]) #grab subarray of rows with our hits
            PEs_in_slice = np.sum(filtered_rows[:,6]) #sum all the PEs for this nn in the slice. 
        
            containment_array[int(nn)][0] = nn #neutrino_number
        
            if (containment_array[int(nn)][1]) <= (hits_in_slice / total_hits) :
                containment_array[int(nn)][1] = hits_in_slice / total_hits #hits contained
                containment_array[int(nn)][3] = slice_number

            if (containment_array[int(nn)][2]) <= (PEs_in_slice / total_PEs) :
                containment_array[int(nn)][2] = PEs_in_slice / total_PEs #PEs contained
                containment_array[int(nn)][4] = slice_number #this slice. 

    #Create a masked version of the array
    neutrinos_with_TMS_hits = np.where(neutrino_PE_array_merged[:, 2] != 0)[0] #check that number of hits is non-zero
    masked_containment_array = containment_array[neutrinos_with_TMS_hits]

    #return containment, masked containment, and purity. 
    return(containment_array, masked_containment_array, purity_array)
    

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing. Apparently this one saves our root file too.
def main():
    print("initializing")
    #setting up.
    numpy_file = sys.argv[1] #load in our .npz file
    output_dir = str(sys.argv[2]) #grab output directory
    file_number = int(sys.argv[3]) #grab the file number 
    init_threshold = float(sys.argv[4]) #grab initialization threshold
    drop_threshold = float(sys.argv[5]) #grab drop threshold
    data = np.load(numpy_file) #open it - keyed like merged hits = ['first']
    hit_array_merged = data['first']
    n_neutrino_events = data['second']

    print("Running Time Slicing")

    #Run the time slicing
    sorted_hits_merged = SortByTime(hit_array_merged) #do a time sorting of the merged hit array
    
    neutrino_PE_array_merged = ReturnNeutrinoInfoArray(sorted_hits_merged, n_neutrino_events)
    
    sliced = SimpleTimeSlicer(hit_array_merged, initial_threshold_energy =  init_threshold, drop_threshold_energy = drop_threshold) #the slicing!!
    
    put_in_slices, not_in_slices, frac_not_in_slices = ValidateEvents(sliced, hit_array_merged) #validation
    
    containment_array, masked_containment_array, purity_array = SliceEvaluate(sliced, n_neutrino_events, neutrino_PE_array_merged) #Evaluate the slices.

    print("Completed Time Slicing")

    #Doing metric packaging in main because I'm a little lazy!
    avg_PE_eff = np.mean(masked_containment_array[:,2])
    avg_hit_eff = np.mean(masked_containment_array[:,1])
    avg_purity = np.mean(purity_array[:,1])
    missing = not_in_slices
    n_slices = len(sliced)
    #Creating metric array
    metric_array = np.array((init_threshold, drop_threshold, avg_PE_eff, avg_hit_eff, avg_purity, missing, n_slices))

    #Doing the rest of the packaging 
    time_slices_stack = StackEvents(sliced)

    #The saving here is a little weird, since our data is "awkward" in the sense that time_slices_stack does not have homogenous entries. Easiest way I know to handle this is with uproot
    save_file =  output_dir + "file_" + str(file_number) + "_sliced_init-" + str(init_threshold) + "_drop-" + str(drop_threshold) + ".root"
    #Convert list of arrays to an awkward array:
    column_names = ["neutrino number", "hit number", "hit x", "hit y", "hit z", "hit time", "hit PE", "bar orientation"]
    records_per_event = [
        [dict(zip(column_names, row)) for row in arr]
        for arr in time_slices_stack
    ]
    awk_array = ak.Array(records_per_event)
    
    #saves two root trees, one holds all the time slices and another the metrics and file number, this could be hdf5 at somepoint but serves our purposes for now!
    with uproot.recreate(save_file) as f:
        f['Time Slices'] = awk_array 
        f["Metrics"] = {
            "init_thresh": np.array([metric_array[0]]),
            "drop_thresh": np.array([metric_array[1]]),
            "avg_pe_eff": np.array([metric_array[2]]),
            "avg_hit_eff": np.array([metric_array[3]]),
            "events_per_slice": np.array([metric_array[4]]),
            "missing_hits": np.array([metric_array[5]]),
            "n_slices": np.array([metric_array[6]]),
            "file number": np.array([file_number])
        }
        

    print("output saved")
    
main()









