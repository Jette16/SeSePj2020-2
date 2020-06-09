import src.dtw_mean as DTW
import numpy as np
import pandas as pd


class Partly_Random:
    
    #Class for the partly random approach with the variable config (dict) where:
    # -n: number of generated paths for each calculated path
    
    def __init__(self, n):
        self.config = dict()
        self.config["n"] = n
        
    def get_config(self):
        return self.config
    
    
    def randomize_specific_section(self, path):
        path_copy = path.copy()
        end = path_copy.shape[0]
        random_start = np.random.randint(0,end-2)
        random_end =  np.random.randint(random_start+1,end-1)
    
        end_idx = path_copy[random_end]
        actual = path_copy[random_start]
    
        new_path = path_copy[:random_start]
        while True:
            if actual[0] == end_idx[0] and actual[1] == end_idx[1] or (actual[0] == end and actual[1] == end):
                break
            new_path = np.concatenate((new_path,np.array([actual])))
        
            if actual[0] == end_idx[0] or actual[0] == end:
                actual[1] = actual[1]+1
                continue
            
            elif actual[1] == end_idx[1] or actual[1] == end:
                actual[0] = actual[0]+1
                continue
            
            random_choice = np.random.randint(0,3)
        
            if random_choice == 0:
                actual[0] = actual[0]+1
            elif random_choice == 1:
                actual[1] = actual[1]+1
            else:
                actual[0] = actual[0]+1
                actual[1] = actual[1]+1
    
        new_path = np.concatenate((new_path,np.array(path_copy[random_end:])))
    
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
                    new_path = self.randomize_specific_section(path)
                    temp_W,temp_V = DTW.get_warp_val_mat(new_path)
                    temp_V = np.diag(temp_V[:,0])
                    new_series = (np.linalg.inv(temp_V)@temp_W)
                    new_series = new_series@data[i]
                    new_series_list.append(new_series)
                    
        return np.array(new_series_list)
    
    def augment(self, X_data, y_data):
        
        if not(type(X_data) is np.ndarray and  type(y_data) is np.ndarray):
            print("Input type has to be np.ndarray")
            return
        X_data_copy = X_data.copy()
        y_data_copy = y_data.copy()
        classes = np.unique(y_data)
        
        for c in classes:
            class_data = X_data[y_data == c]
            new_data = self.augment_paths(class_data)
            labels = [c] * new_data.shape[0]
            X_data_copy = np.vstack((X_data_copy, new_data))
            y_data_copy = np.hstack((y_data_copy, labels))
            
        return X_data_copy, y_data_copy