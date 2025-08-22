#This script takes as input a npz file produced by TimeCluster.py , a merged hits file (for evaluation), and yields a npz file with a biased DBSCAN Applied
#Ie, input = time segments, applies a biased DBSCAN, outputs = clusters within time segments. 

#Kieran Wall - University of Virginia - July 2025
#I apologize to any CS folks who may have to read this

#Run - python3 ClusterDBSCAN.py "input time segmented" "input merged hits" "file_number" "epsilon" "min_points" "output directory"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
import ROOT as root
import uproot
from sklearn.cluster import DBSCAN
import sys
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions

#Let's perform a DBSCAN run on just this example event to play around with some parameters. 
def BiasedDBSCAN(seg_hits_, Epsilon, MinCluster, scale_vec): #scale_vec is a tuple like (z_scale, x_scale, t_scale)
    z_scale, x_scale, t_scale = scale_vec
    nns = seg_hits_[:,0]
    hit_xs = seg_hits_[:,2]
    hit_zs = seg_hits_[:,4]
    hit_ts = seg_hits_[:,5]
    if len(np.unique(hit_xs)) > 1 and len(np.unique(hit_zs)) > 1 and len(np.unique(hit_ts)) > 1 : #this serves to check if there is more than 3 unique dimensions in our vector. If none, just assign same label 0 for whole group. 
        min_x = min(hit_xs)
        max_x = max(hit_xs)
        x_range = max_x - min_x
        normalized_xs = (hit_xs + abs(min_x)) / x_range
        scaled_xs = x_scale * normalized_xs

        #Normalize z
        min_z = min(hit_zs)
        max_z = max(hit_zs)
        z_range = max_z - min_z
        normalized_zs = (hit_zs - min_z) / (z_range)
        scaled_zs = z_scale * normalized_zs

        #Normalize t (this won't work for negative t I don't think..)
        min_t = min(hit_ts)
        max_t = max(hit_ts)
        t_range = max_t - min_t
        normalized_ts = (hit_ts - min_t) / (t_range)
        scaled_ts = t_scale * normalized_ts

        scaled_hit_vecs = []
        for i in range(len(scaled_xs)):
            scaled_hit_vec = [scaled_zs[i], scaled_xs[i], scaled_ts[i]]
            scaled_hit_vecs.append(scaled_hit_vec)

        scaled_hit_vec_array = np.array(scaled_hit_vecs)

        labels = (DBSCAN(eps = Epsilon, min_samples = MinCluster).fit(scaled_hit_vec_array)).labels_
    else:
        labels = np.zeros_like(hit_xs)

    
    return(np.column_stack((seg_hits_, labels)))

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

def SortByTime(array):
    sorted_array = array[np.argsort(array[:,5])]
    return(sorted_array)

#This function runs DBSCAN on a full file worth of hit segments. 
def FileRunDBSCAN(hits_segmented_, EPSILON, MINPOINTS, scale_vector):
    spills = np.unique(hits_segmented_[:,-3])
    labeled_segments = []
    for spill_no in spills:
        #the hits in a given spill
        spill_hits = hits_segmented_[(hits_segmented_[:,-3] == spill_no)]
        segments = np.unique(spill_hits[:,-1])
        for segment in segments:
            #the hits in a given segment of a spill
            segment_hits = spill_hits[spill_hits[:,-1] == segment]
            labeled_hits = BiasedDBSCAN(segment_hits, EPSILON, MINPOINTS, scale_vector)
            labeled_segments.append(labeled_hits)

    return(np.vstack(labeled_segments))


def FullFileDBSCANEvaluate(labeled_segment_array_, neutrino_truth_array, number_of_neutrinos):
    #We can define a containment array, will be looping over both spills and clusters and checking for the best cluster!
    containment_array = np.zeros((number_of_neutrinos,3)) #|nn | hit containment | PE containment | best hit containment spill | best hit containment cluster | best PE containment spill | best PE containment cluster |
    #Initialize with nns
    for i in range(number_of_neutrinos):
        containment_array[i][0] = i

    cluster_occupancy = []

    #now lets loop
    spills = np.unique(labeled_segment_array_[:,-4])
    for spill_no in spills:
        #the hits in a given spill
        spill_hits = labeled_segment_array_[(labeled_segment_array_[:,-4] == spill_no)]
        segments = np.unique(spill_hits[:,-2])
        for segment in segments:
            segment_hits = spill_hits[spill_hits[:,-2] == segment]
            clusters = np.unique(segment_hits[:,-1])
            for cluster in clusters:
                #all the hits within a spill, within a given segment, within a given cluster. 
                cluster_hits = segment_hits[segment_hits[:,-1] == cluster]
                #safety line?
                #if cluster_hits.ndim == 1:
                #    cluster_hits = cluster_hits.reshape(1, -1)
                nns_in_cluster = np.unique(cluster_hits[:,0])
                cluster_occupancy.append(len(nns_in_cluster))
                for nn in nns_in_cluster:
                    #print(cluster_hits)
                    #print(cluster_hits[:,0])
                    
                    nn_sub_array = cluster_hits[ cluster_hits[:,0] == nn ]  #array of hits in a cluster with a given neutrino number
                    #print("pass")
                    #Grab truth information for this neutrino number
                    total_PEs = neutrino_truth_array[int(nn)][1] #total PE
                    total_hits = neutrino_truth_array[int(nn)][2] #total hits
                    cluster_PE_sum = np.sum(nn_sub_array[:,6])
                    cluster_hits_sum = np.shape(nn_sub_array)[0]

                    if (containment_array[int(nn)][1]) <= (cluster_hits_sum / total_hits) :
                        containment_array[int(nn)][1] = cluster_hits_sum / total_hits #hits contained

                    if (containment_array[int(nn)][2]) <= (cluster_PE_sum / total_PEs) :
                        containment_array[int(nn)][2] = cluster_PE_sum / total_PEs #PEs contained
                        
                        
    #apply a masking such that we are only reviewing neutrino events with tms hits
    neutrinos_with_TMS_hits = np.where(neutrino_truth_array[:, 2] != 0)[0] #check that number of hits is non-zero
    masked_containment_array = containment_array[neutrinos_with_TMS_hits]

    return(containment_array, masked_containment_array, cluster_occupancy) #now we return both arrays

def SaveToNPZ(dbscan_clustered_hits, epsilon, file_number_, masked_cluster_eval_, cluster_occupancy_, out_dir):
    outpath = out_dir + 'hits_DBSCAN_clustered_' + 'epsilon_' + str(epsilon) + '_' + str(file_number_) + '.npz' 
    np.savez(outpath, first= dbscan_clustered_hits, second = masked_cluster_eval_, third = cluster_occupancy_)
    


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing. Apparently this one saves our root file too.
def main():
    
    print("initializing")
    time_segmented_file = np.load(sys.argv[1]) #load in the time segments.
    hits_segmented = time_segmented_file['first'] #grab the hits from the file.

    merged_hits_file = np.load(sys.argv[2]) #load in the merged hits file (for evaluation)
    merged_hits_array = merged_hits_file['first']
    n_neutrino_events = merged_hits_file['second']

    file_number = sys.argv[3]

    epsilon_ = float(sys.argv[4]) #grab epsilon
    min_points_ = int(sys.argv[5]) #grab the min points
    scale_vector_ = (1,1,0)

    output_directory = sys.argv[6]
    
    #Now run DBSCAN on the full hit_segments.
    print(f"running DBSCAN on file with e = {epsilon_}, min = {min_points_}, scale vector = {scale_vector_}")
    labeled_segment_array = FileRunDBSCAN(hits_segmented, epsilon_, min_points_, scale_vector_) 

    #Create a neutrino truth array (for evaluation)
    neutrino_truth_array = ReturnNeutrinoInfoArray(SortByTime(merged_hits_array), n_neutrino_events)

    #Run evaluation
    print("evaluating")
    unmasked_cluster_eval, masked_cluster_eval, cluster_occupancy = FullFileDBSCANEvaluate(labeled_segment_array, neutrino_truth_array, n_neutrino_events)

    SaveToNPZ(labeled_segment_array, epsilon_, file_number, masked_cluster_eval, cluster_occupancy, output_directory)
    print("saved!")

main()
    

    

    














