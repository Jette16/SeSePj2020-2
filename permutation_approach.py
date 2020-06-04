import src.dtw_mean as DTW
import numpy as np
import pandas as pd


class Permutator:
    
        
    #Class for the partly random approach with the variable config (dict) where:
    # -n: number of generated paths for each calculated path
    # -probability: the probability that a point of the path is permutated
    
    def __init__(self, n, probability):
        self.config = dict()
        self.config["n"] = n
        self.config["probability"] = probability
        
    def get_config(self):
        return self.config
    
    
    def permutate(self, path):
        path_copy = path.copy()
        prob = self.config["probability"]
    
        new_path = np.array([path_copy[0]])
        permutate = False
        prev = path_copy[0]
        i = path_copy[1]
        permutation = [0.,0.]
        end = path_copy[-1][0]
    
        while i[0] != end and i[1] != end:  
            if not permutate:
           
                choice = np.random.choice(2, 1, p=[1.0-prob, prob])
        
                if choice[0] == 0 or end in i:
                    new_path = np.concatenate((new_path,np.array([i])))
                    # get next element of path
                    prev = i.copy()
                
                    temp1 = np.where(path_copy[:,0] == i[0])[0]
                    temp2 = np.where(path_copy[:,1] == i[1])[0]
                
                    i = path_copy[np.intersect1d(temp1,temp2)+1][0]
                
                else: 
                    permutate = True
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
                    
                        if permutation[0] != i[0] or permutation[1] != i[1]:
                            break
            else:
                new_path = np.concatenate((new_path,np.array([permutation])))
            
                if permutation[0] == end:
                    permutation[1] = permutation[1]+1
            
                elif permutation[1] == end:
                    permutation[0] = permutation[0]+1
            
                else:
                    random_choice = np.random.randint(0,3)
                    if random_choice == 0:
                        permutation[0] = permutation[0]+1       
                    elif random_choice == 1:
                        permutation[1] = permutation[1]+1
                        
                    else:
                        permutation[0] = permutation[0]+1
                        permutation[1] = permutation[1]+1
                    
                test = (path_copy[:,0] == permutation[0]) & (path_copy[:,1] == permutation[1])
                if path_copy[test].shape[0] != 0:
                    permutate = False
                    i = permutation.copy()
                
        new_path = np.concatenate((new_path,np.array([i])))     
        return new_path
    
    def augment_paths(self, data):
        #number of timeseries
        count = data.shape[0]
        length = data.shape[1]
        new_series_list = []
        
        #calculate every path between every two timeseries
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                
                d, path = DTW.dtw(np.reshape(data[i], (length,1)),np.reshape(data[j], (length,1)), path = True)
                
                for n in range(self.config["n"]):
                    try:
                        new_path = self.permutate(path)
                        temp_W,temp_V = DTW.get_warp_val_mat(new_path)
                        temp_V = np.diag(temp_V[:,0])
                        new_series = (np.linalg.inv(temp_V)@temp_W)
                        new_series = new_series@data[i]
                        new_series_list.append(new_series)
                    except ValueError:
                        continue
        return np.array(new_series_list)
    
    def augment(self, X_data, y_data):
        
        if not(type(X_data) is np.ndarray and  type(y_data) is np.ndarray):
            print("Input type has to be np.ndarray")
            return
        
        classes = np.unique(y_data)
        X_data_copy = X_data.copy()
        y_data_copy = y_data.copy()
        
        for c in classes: 
            class_data = X_data[y_data == c]
            new_data = self.augment_paths(class_data)
            print(new_data.shape)
            labels = np.array([c] * new_data.shape[0])
            X_data_copy = np.concatenate((X_data_copy, new_data))
            y_data_copy = np.concatenate((y_data_copy, labels))
            print(X_data_copy.shape)
            
        return X_data_copy, y_data_copy