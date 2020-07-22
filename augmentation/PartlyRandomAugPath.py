from augmentation.src import dtw_mean as DTW
import numpy as np
import pandas as pd


class PartlyRandomAug:
    
    #Class for the partly random approach with the variable config (dict) where:
    # -n: number of newly generated paths for each calculated path
    
    #ATTENTION: 
    # This augmentation method assumes that the path between the time series is already calculated. 
    # 
    
    def __init__(self, n):
        self.config = dict()
        self.config["n"] = n
        
    def get_config(self):
        return self.config
    
    
    def randomize_specific_section(self, path):
        path_copy = path.copy()
        end = path_copy.shape[0]
        
        # Choose random start and end point
        random_start = np.random.randint(0,end-2)
        random_end =  np.random.randint(random_start+1,end-1)
        end_idx = path_copy[random_end]
        actual = path_copy[random_start]
    
        # Copy every point from the original path until the start point to the new path
        new_path = path_copy[:random_start]
        
        # Add randomly new points to the new path
        while True:
            # Check if newly added point matches with the end point or the end of the path
            if actual[0] == end_idx[0] and actual[1] == end_idx[1] or (actual[0] == end and actual[1] == end):
                break
                
            # Add the new point to the new path
            new_path = np.concatenate((new_path,np.array([actual])))
        
            # Check if just one direction is possible
            if actual[0] == end_idx[0] or actual[0] == end:
                actual[1] = actual[1]+1
                continue
            
            elif actual[1] == end_idx[1] or actual[1] == end:
                actual[0] = actual[0]+1
                continue
            # if not, choose randomly the next point
            random_choice = np.random.randint(0,3)
        
            if random_choice == 0:
                actual[0] = actual[0]+1
            elif random_choice == 1:
                actual[1] = actual[1]+1
            else:
                actual[0] = actual[0]+1
                actual[1] = actual[1]+1
                
        # Add the rest of the original path from end point on until the end 
        new_path = np.concatenate((new_path,np.array(path_copy[random_end:])))
    
        return new_path
    

    def augment(self, X_data, y_data, indices, paths, path_indexes):
        #X_data: A numpy array containing the time series
        #y_data: A numpy array containing the class labels
        #indices: The indices of the different time series in X_data (regarding to all the available time series)
        #paths: All the paths belonging to this dataset
        #path_indexes: Containing triples [class, time series index 1, time series index 2]
        #to match the paths to the correct time series       
        
        if not(type(X_data) is np.ndarray and  type(y_data) is np.ndarray):
            print("Input type has to be np.ndarray")
            return
        
        X_data_copy = X_data.copy()
        y_data_copy = y_data.copy()
        
        # iterate through all the paths available
        for idx,p in enumerate(paths): 
            # check if both time series of the path are existing in X_data
            if not (path_indexes[idx][1] in indices and path_indexes[idx][2] in indices):
                continue
            # create for this path n times new paths
            for i in range(self.config["n"]):
                # create new path
                new_path = self.randomize_specific_section(np.array(p))
                # calculate the new time series and add it to X_data and y_data
                temp_W,temp_V = DTW.get_warp_val_mat(new_path)
                temp_V = np.diag(temp_V[:,0])
                new_series = (np.linalg.inv(temp_V)@temp_W)
                data = X_data[indices == path_indexes[idx][1]]
                c = path_indexes[idx][0]
                new_series = new_series@data[0]
                
                X_data_copy = np.vstack((X_data_copy, new_series))
                y_data_copy = np.hstack((y_data_copy, c))
            
        return X_data_copy, y_data_copy