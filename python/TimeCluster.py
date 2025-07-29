#This script takes as input a npz file produced by MultiHitEffects, an edep-sim file (smh), a file number, KDE Bandwidth parameters, and outputs time segments. 
#Order of specificity goes file -> spill -> time segment -> (optional) DBSCAN cluster

#Kieran Wall - University of Virginia - July 2025
#I apologize to any CS folks who may have to read this

#Run - python3 TimeCluster.py "input edep-sim" "inputer merged_hits" "file_number" "fine bandwidth" "output file" "Use Truth Spill Information?" "Apply DBSCAN?"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
import collections
import ROOT as root
import awkward as ak
import uproot
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import sys
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions

def SortByTime(array):
    sorted_array = array[np.argsort(array[:,5])]
    return(sorted_array)

#This function adds truth level spill information to our hits.
def AddSpillInfo(time_sorted_merged_hits_ , full_spill_lookup_):
    nns_in_hits = np.unique(time_sorted_merged_hits_[:,0])
    relevant_spill_lookup = {}
    for nn_in_hits in nns_in_hits:
        spill = int(full_spill_lookup_[int(nn_in_hits)][1])
        relevant_spill_lookup[nn_in_hits] = spill  

    #Just go ahead and add the information to the time_sorted_merged_hits for convenience. 
    spill_nos = []
    for hit in time_sorted_merged_hits_:
        nn = hit[0]
        spill_nos.append(relevant_spill_lookup[nn])

    time_sorted_merged_hits_spillnos = np.column_stack((time_sorted_merged_hits_ ,spill_nos)) #use this object moving forward!

    return(time_sorted_merged_hits_spillnos)
    # Hit State - > (collective_neutrino_number, collective_hit_number, bar_x, bar_y, bar_z, collective_time, total_pe, bar_orientation_code, file_no, truth_spill)


#This function creates time segments using a Kernel Density Estimate, outputs the hits with the cluster labels appended. 
#This is generic, meaning that we can feed in basically any data sequence containing time series information and have it segmented. 
def MakeKDSegments(hits, bandwidth = 15, mesh_points = 1000, plot = False):
    
    hit_ts = hits[:,5] #extract just the time series data. 
    
    hit_ts_reshaped = hit_ts.reshape(-1, 1)

    # KDE fitting using a gaussian kernel. 
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(hit_ts_reshaped)

    # Evaluate KDE on a fine grid - project our function onto a fine grid to look for local minima
    #This is a necessary step to be able to find our local minima numerically. Basically the kde gives us our functional form, we need to plot it on points to get out the pdf. 
    #The plotting with the points is accomplished by pulling the score_samples method, which returns log likelihoods. We then convert these into actually probabilitiy densities. 
    t_plot = np.linspace(hit_ts.min() - 2, hit_ts.max() + 2, mesh_points).reshape(-1, 1) 
    log_dens = kde.score_samples(t_plot)
    dens = np.exp(log_dens)

    # Find valleys in KDE curve (ie, local minima)
    #Can use scipy peak finding, but invert data so we are really finding the local minima! Returns indices
    inverted = -dens
    valleys, _ = find_peaks(inverted)
    valley_positions = t_plot[valleys].flatten() #this step maps our indices onto actual times. 

    # Assign cluster index based on which region the point falls into
    def assign_cluster(t, boundaries):
        for i, b in enumerate(boundaries): #will loop through boundaries and the last one assigned will be the one it fits into.
            if t < b:
                return(i)
        return(len(boundaries))

    labels = np.array([assign_cluster(t, valley_positions) for t in hit_ts]) #assign labels to our hits. 

    
    if (plot == True): #output a KDE plot
        plt.plot(t_plot[:, 0], dens, label="KDE")
        plt.scatter(hit_ts, np.zeros_like(hit_ts), c=labels, cmap='tab20', s=50, label="Time Segments")
        for v in valley_positions:
            plt.axvline(v, color='gray', linestyle='--', linewidth=1)
        plt.title("KDE-based hit Clustering")
        plt.xlabel("Hit Time")
        plt.yticks([])
        plt.legend()
        plt.show()

    print(f"We found {len(np.unique(labels))} clusters")
    clusters_added = np.column_stack((hits, labels)) #save our data with cluster labels added to the end of each hit row. 
    
    return(clusters_added)

#The neutrino truth array generation function. 
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

#Records basic metrics on the performance of the fine clustering
def FullFileClusterEvaluate(list_of_spills_, neutrino_truth_array, number_of_neutrinos): #takes an argument like a list of spills with their fine clustering applied
    #We can define a containment array, will be looping over both spills and clusters and checking for the best cluster!
    containment_array = np.zeros((number_of_neutrinos,7)) #|nn | hit containment | PE containment | best hit containment spill | best hit containment cluster | best PE containment spill | best PE containment cluster |
    #Initialize with nns
    for i in range(number_of_neutrinos):
        containment_array[i][0] = i

    cluster_occupancy = []

    #Big nested loop coming up
    #Loop over each slice
    for i, spill_ in enumerate(list_of_spills_) :
        #Now we need to identify the clusters:
        cluster_numbers = np.unique(spill_[:,-1]) #grab the cluster numbers from within the spill
        for cluster_number in cluster_numbers:
            cluster_sub_array = spill_[ spill_[:,-1] == cluster_number ] #this is an array of all the hits in a given cluster. 
            #now do a grouping by neutrino number
            nns_in_cluster = np.unique(cluster_sub_array[:,0])
            cluster_occupancy.append(len(nns_in_cluster))
            #now loop over the neutrino numbers
            for nn in nns_in_cluster:
                nn_sub_array = cluster_sub_array[ cluster_sub_array[:,0] == nn ]  #array of hits in a cluster with a given neutrino number
                #Grab truth information for this neutrino number
                total_PEs = neutrino_truth_array[int(nn)][1] #total PE
                total_hits = neutrino_truth_array[int(nn)][2] #total hits
                cluster_PE = np.sum(nn_sub_array[:,6])
                cluster_hits = np.shape(nn_sub_array)[0]

                if (containment_array[int(nn)][1]) <= (cluster_hits / total_hits) :
                    containment_array[int(nn)][1] = cluster_hits / total_hits #hits contained
                    containment_array[int(nn)][3] = i #slice
                    containment_array[int(nn)][4] = cluster_number #cluster number within slice

                if (containment_array[int(nn)][2]) <= (cluster_PE / total_PEs) :
                    containment_array[int(nn)][2] = cluster_PE / total_PEs #PEs contained
                    containment_array[int(nn)][5] = i #this slice. 
                    containment_array[int(nn)][6] = cluster_number #cluster number within slice

    #apply a masking such that we are only reviewing neutrino events with tms hits
    neutrinos_with_TMS_hits = np.where(neutrino_truth_array[:, 2] != 0)[0] #check that number of hits is non-zero
    masked_containment_array = containment_array[neutrinos_with_TMS_hits]

    return(containment_array, masked_containment_array, cluster_occupancy) #now we return both arrays  

#This function can be called to save our output to an NPZ file, is generic so can save with or without the additioanl DBSCAN label. Tales as input an array of hits, so make sure to convert to that. 
def SaveToNPZ(segmented_hits, bandwidth_fine_, file_number_, masked_cluster_eval_, cluster_occupancy_, out_dir):
    outpath = out_dir + 'hits_time_segmented_' + 'band_fine_' + str(bandwidth_fine_) + '_' + str(file_number_) + '.npz' 
    np.savez(outpath, first= segmented_hits, second = masked_cluster_eval_, third = cluster_occupancy_)
    

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing. Apparently this one saves our root file too.
def main():
    print("initializing")
    
    #take the command line arguments of the input files and open them - there are dreadfully two of these since we need spill level info!
    edep_file = uproot.open(sys.argv[1]) #edep
    edep_detsim = edep_file["DetSimPassThru"] 
    gRooTrackerTree = edep_detsim['gRooTracker']
    
    merged_hits_file = np.load(sys.argv[2]) #merged hits
    merged_hits_array = merged_hits_file['first']
    n_neutrino_events = merged_hits_file['second']

    file_number = int(sys.argv[3]) 

    bandwidth_fine = int(sys.argv[4]) #grab the fine KDE bandwidth 

    output_directory = sys.argv[5]

    use_true_spills = sys.argv[6]

    #Now grab spill level truth labels - full lookup table, input for a function later.
    interaction_times = gRooTrackerTree['EvtVtx'].arrays().EvtVtx[:,3] / (10**9)
    spills = (interaction_times // 1.2)
    neutrino_numbers = np.arange(int(n_neutrino_events))
    full_spill_lookup = np.column_stack((neutrino_numbers,spills)) #full lookup

    merged_hits_spillnos = AddSpillInfo(SortByTime(merged_hits_array) , full_spill_lookup) #add truth level spill information. 

    #Now run KDE on our file to extract spills. Working on optimizing bandwidth and mesh parameters 
    file_clustered = MakeKDSegments(merged_hits_spillnos, bandwidth = 1000, mesh_points = 2500000, plot = False)
    print("Spill segmentation is complete!")
    
    #Now do a brief check to see whether the spill segmentation placed some events into the wrong spill. This is not the end of the world will just be an efficiency hit if stuff gets spill across spills. 
    spill_clustered_right = len(file_clustered[file_clustered[:,-1] == file_clustered[:,-2]])
    spills_correct = spill_clustered_right / len(file_clustered)
    if spills_correct != 1.0:
        print(f"Be advised, spill segmentation accuracy != 1.0, instead classified with {spills_correct}")

    #Now lets perform the fine time segmentation - ie breaking each spill into constituent time segments. 
    print("Beginning fine segmentation")

    spill_no_index = -1 #default is -1, will flip to -2 if using the truth level spill information. 
    if (use_true_spills == 'True'):
        print("Using truth level information for spill segmentation, disregarding first KDE clustering results")
        spill_no_index = -2
    
    spills_fine_segmented = [] #this list will store all of the hits with fine segmentation grouped by spill. 
    spills = np.unique(file_clustered[:,int(spill_no_index)]) #grab spill numbers (should always be 0,12 for current sim)
    for spill_no in spills:
        selected_spill = spill_no
        hits_in_spill = file_clustered[file_clustered[:,int(spill_no_index)] == selected_spill ] #grabbing our desired spill.  
        time_segmented = MakeKDSegments(hits_in_spill, bandwidth = bandwidth_fine, mesh_points = 5000, plot = False)
        spills_fine_segmented.append(time_segmented)
    print("Fine segmentation of spills is complete!")
    #Hit State -> (collective_neutrino_number, collective_hit_number, bar_x, bar_y, bar_z, collective_time, total_pe, bar_orientation_code, file_no, truth_spill, clustered_spill, clustered_segment)
    
    #Now let's record some basic metrics on the fine clustering performance
    neutrino_truth_array = ReturnNeutrinoInfoArray(SortByTime(merged_hits_array), n_neutrino_events)
    unmasked_cluster_eval, masked_cluster_eval, cluster_occupancy = FullFileClusterEvaluate(spills_fine_segmented, neutrino_truth_array, n_neutrino_events)
    print(f"Fine segmentation hit efficiency {np.mean(masked_cluster_eval[:,2])}, occupancy {np.mean(cluster_occupancy)}") #just a test printout. 

    #Now we can save our fully labeled hits to a .npz file -> doing it spill wise would result in jagged array, so can just pull that information in an analysis file with a sort. 
    hits_fully_labeled = np.vstack(spills_fine_segmented)
    
    SaveToNPZ(hits_fully_labeled, bandwidth_fine, file_number, masked_cluster_eval, cluster_occupancy, output_directory)
    
main()
    
    
        









    


    