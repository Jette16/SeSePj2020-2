import pickle
import numpy as np
import pandas as pd
import src.dtw_mean as DTW

def generate_save_paths(X_data, y_data, index, path_to_file):
    '''
    function to generate and save the warping paths
     - X_data: The data to generate the paths
     - y_data: The labels according to each time series
     - index: The index according to each time series in X_data
     - path_to_file: String with indicates where the file for the paths and their indices will be saved
    '''

    paths = []
    index_combination = []

    classes = np.unique(y_data)

    length = X_data.shape[1]
    for c in classes: 
        ind = index[y_data == c]
        for i in ind:
            for j in ind:
                if i == j:
                    continue
                d, path = DTW.dtw(np.reshape(X_data[i,:], (length,1)),np.reshape(X_data[j,:], (length,1)), path = True)
                paths.append(path)
                index_combination.append([c,i,j])
        print(f"Generated paths for class {c}")


    with open(path_to_file + "_paths.txt", "wb") as f:
        pickle.dump(paths, f)
    
    with open(path_to_file + "path_indices.txt", "wb") as f:
        pickle.dump(index_combination, f)
