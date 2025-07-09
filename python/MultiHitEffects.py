#This script takes as input a npz file produced by HitWiseEffects.py and outputs --tbd--

#Kieran Wall - University of Virginia - June 2025
#I apologize to any CS folks who may have to read this

#Run - python3

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, types
from numba.typed import Dict, List
import sys
import awkward as ak

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions

#This function groups our hits by bar, returns a dictionary keyed by the bar coordinates. 
def SortByBar(hit_array):
    N = hit_array.shape[0]

    # Group hits by (x, z)
    group_dict = Dict.empty(
        key_type=types.UniTuple(types.float64, 2),
        value_type=types.ListType(types.int64)
    )

    for i in range(N):
        x = hit_array[i, 2]
        z = hit_array[i, 4]
        key = (x, z)
        if key not in group_dict:
            group_dict[key] = List.empty_list(types.int64)
        group_dict[key].append(i)
        
    return(group_dict)

#This function will take our group dictionary, detsim_hit_array, and a readout window length, and sorts hits in bars into timing groups
def CreateTimeGroups(hit_array, bar_dictionary, readout_window = 120):
    all_groups = []
    for key in bar_dictionary:
        index_list = bar_dictionary[key]
        times = [hit_array[i][5] for i in index_list]
        sorted_idx = np.argsort(times) #returns indixes that would sort a list. - not matched to hits
        sorted_times = [times[i] for i in sorted_idx] #times sorted by times
        sorted_indices = [index_list[i] for i in sorted_idx] #return sorted list of hit indices
        start = 0
        time_index_array = np.column_stack((sorted_times, sorted_indices)) #perhaps a useful object
        groups = []
        current_group = [int(time_index_array[0][1])]
        t_start = time_index_array[0][0]

        for i in range(1, time_index_array.shape[0]):
            t = time_index_array[i][0]
            idx = int(time_index_array[i][1])

            if t - t_start <= readout_window:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
                t_start = t

        # Add the final group
        if current_group:
            groups.append(current_group)

        # Print and append result
        for i, group in enumerate(groups):
            #print(f"Key {key} - Group {i}: {group}")

            all_groups.append(group)
            
    return(all_groups)

#This function will do the hit merging, outputs
def MergeCoincidentHits(all_groups, hit_array):
    merged_hit_collection = []
    multi_neutrino_groups = 0
    for i in range(len(all_groups)):
        hit_indexes = all_groups[i]
        total_PE = 0.
        collective_neutrino_number = -1
        collective_hit_number = -1
        neutrino_numbers = []
        hit_numbers = []
        hit_pes = []
        hit_times = []

        bar_x = hit_array[hit_indexes[0]][2] 
        bar_y = hit_array[hit_indexes[0]][3] 
        bar_z = hit_array[hit_indexes[0]][4] 
        bar_orientation_code = hit_array[hit_indexes[0]][7]

        #for index in hit_indexes:
        #    print(hit_array[index])

        for index in hit_indexes:
            hit = hit_array[index]
            hit_times.append(hit[5])
            neutrino_numbers.append(hit[0])
            hit_numbers.append(hit[1])
            hit_pes.append(hit[6])
    

        if (len(np.unique(neutrino_numbers)) > 1):
            #print("Notice - multiple neutrino numbers in this group, will merge to earliest time")
            multi_neutrino_groups += 1

        min_time_index = np.argmin(hit_times)
        total_pe = sum(hit_pes)
        collective_neutrino_number = neutrino_numbers[min_time_index]
        collective_hit_number = hit_numbers[min_time_index]
        collective_index = hit_indexes[min_time_index]
        collective_time = hit_times[min_time_index]
        merged_hit_info = [collective_neutrino_number, collective_hit_number, bar_x, bar_y, bar_z, collective_time, total_pe, bar_orientation_code, hit_indexes]
        #merged_hit_info stores our new merged hit object |Collective Neutrino #|Collective Hit #|Bar x|Bar y|Bar z|Total PE|Bar Orientation Code|constitutent hit_indexes|
    
        merged_hit_collection.append(merged_hit_info)
     
    print(f"From Hit Merger: Found {multi_neutrino_groups} multi-neutrino groups!")
    print(f"From Hit Merger: Created {len(merged_hit_collection)} Merged Hits out of {len(hit_array)} Raw Hits")
    return(merged_hit_collection)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing. 
def main():
    print("initializing")
    numpy_file = sys.argv[1] #load in our .npz file
    data = np.load(numpy_file) #open it - keyed like neutrino vertices = data['first'], detector sim hit array = data['second']
    neutrino_vertices = data['first']
    detsim_hit_array = data['second'] #detector sim hit array: |neutrino #|hit #|hit x|hit y|hit z|hit T|hit PE|hit orientation code|
    print("Successfully loaded npz file")
    
    #Currently only supported multi hit effect is coincident hit merging - so merges hits incident in same bar within a 120ns period 
    print("About to apply coincidence merging")
    bar_dict = SortByBar(detsim_hit_array) #sort hits by bar
    groups = CreateTimeGroups(detsim_hit_array, bar_dict) #create timing groups within bars
    merged_hits_list = MergeCoincidentHits(groups, detsim_hit_array) #generate list of merged hits 
    
    #merged_hit_array stores our new merged hit object |Collective Neutrino #|Collective Hit #|Bar x|Bar y|Bar z|Total PE|Bar Orientation Code|constitutent hit_indexes|
    merged_hit_array = ak.Array(merged_hits_list) #converts the list of lists to an awkward array for storage. 
    print("Applied coincidence merging!")

    #Planning on including time slicing in this script, such that the final output here can be an hdf5 ready for input to SPINE.

main()
    
















    



