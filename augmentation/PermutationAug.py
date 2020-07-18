from augmentation.src import dtw_mean as DTW
import numpy as np



class PermutationAug:
    
        
    #Class for the partly random approach with the variable config (dict) where:
    # -n: number of generated paths for each calculated path
    # -probability: the probability that a point of the path is permutated
    
    def __init__(self, nb, probability):
        # self.config = dict()
        # self.config["n"] = n
        # self.config["probability"] = probability
        self.nb=nb
        self.probability=probability
        
    def get_config(self):
        return self.config
    
    
    def permutate(self,path):
        path_copy = path.copy()
        prob = self.probability
        
        new_path = np.array([path_copy[0]])
        permutate = False
        prev = path_copy[0]
        curr = path_copy[1]
        permutation = [0.,0.]
        end = path_copy[-1][0]
        
        while curr[0] != end or curr[1] != end:  
            if not permutate:
           
                choice = np.random.choice(2, 1, p=[1.0-prob, prob])
        
                if choice[0] == 0 or end in curr:
                    new_path = np.vstack((new_path,np.array(curr)))
                    # get next element of path
                    prev = curr.copy()
                
                    temp1 = np.where(path_copy[:,0] == curr[0])[0]
                    temp2 = np.where(path_copy[:,1] == curr[1])[0]
                
                    curr = path_copy[np.intersect1d(temp1,temp2)+1][0]
                
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
                    
                        if permutation[0] != curr[0] or permutation[1] != curr[1]:
                            break
            else:
                new_path = np.vstack((new_path,np.array(permutation)))
            
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
                    curr = permutation.copy()
                
        new_path = np.vstack((new_path,np.array(curr)))     
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
                
                for n in range(self.nb):
                    new_path = self.permutate(path)
                    temp_W,temp_V = DTW.get_warp_val_mat(new_path)
                    temp_V = np.diag(temp_V[:,0])
                    new_series = (np.linalg.inv(temp_V)@temp_W)
                    new_series = new_series@data[i]
                    new_series_list.append(new_series.tolist())

        return np.asarray(new_series_list)
    
    def augment(self, X_data, y_data):
        
        if not(type(X_data) is np.ndarray and  type(y_data) is np.ndarray):
            print("Input type has to be np.ndarray")
            return
        
        classes = np.unique(y_data)
        X_data_copy = X_data.copy()
        y_data_copy = y_data.copy()
        
        for c in classes: 
            class_data = X_data[y_data == c]
            print("data shape ",class_data.shape)
            new_data = self.augment_paths(class_data)
            print("new data shape ",new_data.shape)
            labels = np.array([c] * new_data.shape[0])
            X_data_copy = np.vstack((X_data_copy, new_data))
            y_data_copy = np.hstack((y_data_copy, labels))
            
        return X_data_copy, y_data_copy