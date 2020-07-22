from augmentation.src import dtw_mean as DTW
import numpy as np
import pandas as pd


class PermutationAug:
    
        
    #Class for the partly random approach with the variable config (dict) where:
    # -n: number of generated paths for each calculated path
    # -probability: the probability that a point of the path is permutated

    # ATTENTION:
    # This method assumes that the warping paths were already calculated
    
    def __init__(self, n, probability):
        self.config = dict()
        self.config["n"] = n
        self.config["probability"] = probability
        
    def get_config(self):
        return self.config
    
    
    def permutate(self,path):
        
        # initialize the variables
        
        path_copy = path.copy()
        prob = self.config["probability"]
        
        new_path = np.array([path_copy[0]])
        permutate = False
        prev = path_copy[0]
        curr = path_copy[1]
        permutation = [0.,0.]
        end = path_copy[-1][0]
        
        while curr[0] != end or curr[1] != end:  
            # Check if the algorithm is currently in the state "permutate"
            if not permutate:
                # Choose if the algorithm should permutate with probability prob
                choice = np.random.choice(2, 1, p=[1.0-prob, prob])
        
                # if choice is false or the path has just one direction to go
                if choice[0] == 0 or end in curr:
                    new_path = np.vstack((new_path,np.array(curr)))
                    # get next element of path
                    prev = curr.copy()
                
                    # get the index of the current point
                    temp1 = np.where(path_copy[:,0] == curr[0])[0]
                    temp2 = np.where(path_copy[:,1] == curr[1])[0]
                
                    # get the next point of the path
                    curr = path_copy[np.intersect1d(temp1,temp2)+1][0]
                
                # if choice is true, change the state permutate to True
                else: 
                    permutate = True
                    
                    # permutate the next point and check if the new point does not match with the real point
                    while True:
                        random_choice = np.random.randint(0,3)
                        permutation = prev.copy()
                        if random_choice == 0:
                            permutation[0] = permutation[0]+1
                    
                        elif random_choice == 1:
                            permutation[1] = permutation[1]+1
                        
                        else:
                            permutation[0] = permutation[0]+1
                            permutation[1] = permutation[1]+1
                    
                        if permutation[0] != curr[0] or permutation[1] != curr[1]:
                            break
            else:
                # Add the new permutation point
                new_path = np.vstack((new_path,np.array(permutation)))
            
                # Check if just one direction is possible
                if permutation[0] == end:
                    permutation[1] = permutation[1]+1
            
                elif permutation[1] == end:
                    permutation[0] = permutation[0]+1
            
                else:
                    # Otherwise: Add a random point
                    random_choice = np.random.randint(0,3)
                    if random_choice == 0:
                        permutation[0] = permutation[0]+1       
                    elif random_choice == 1:
                        permutation[1] = permutation[1]+1
                        
                    else:
                        permutation[0] = permutation[0]+1
                        permutation[1] = permutation[1]+1
                # Check if the new point is also in the original point, change the state permutate to False if so    
                test = (path_copy[:,0] == permutation[0]) & (path_copy[:,1] == permutation[1])
                if path_copy[test].shape[0] != 0:
                    permutate = False
                    curr = permutation.copy()
                
        new_path = np.vstack((new_path,np.array(curr)))     
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
                new_path = self.permutate(np.array(p))
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
