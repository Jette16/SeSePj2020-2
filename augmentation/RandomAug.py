import numpy as np
# this class has 4 parameters
# diagonal - probability to go diagonal in random walk
# down - probability to go down in random walk
# right - probability to go right in random walk
# augment_each - specifies if each sequence should be augmented seperatly

class RandomAug:
    def __init__(self, diagonal=0.50, down=0.25, right=0.25, augment_each=True):
        self.diagonal = diagonal
        self.down = down
        self.right = right
        if self.diagonal+self.down+self.right != 1:
            raise Exception("Probabilities don't add up to 1")
        self.augment_each = augment_each
        self.config={"name":"RandomAugmentation", "diagonal":diagonal, "right":right, "augment_each":augment_each}

    def random_w_and_v(self,n,m,diagonal=0.50,down=0.25,right=0.25):
        W=np.zeros((n,m))
        W[0,0]=1
        i=0
        j=0
        while i<n-1 or j<m-1:
            number = np.random.rand(1)
            if i==n-1:
                j+=1
                W[i,j]=1
            elif j==m-1:
                i+=1
                W[i,j]=1
            elif number < self.right:
                i+=1
                W[i,j]=1
            elif number < self.right+self.down:
                j+=1
                W[i,j]=1
            else:
                i+=1
                j+=1
                W[i,j]=1

        V = np.sum(W, axis=1, keepdims=True)
        V= np.diag(V[:,0])
        return W,V
    
    
    #geenreate random data augmentation
    def augment(self, data_X, data_Y):
        if self.augment_each:
            augmented_data_X=np.zeros((data_X.shape[0]*2,data_X.shape[1]))
            augmented_data_Y=np.zeros((data_Y.shape[0]*2))
            augmented_data_X[data_X.shape[0]:]= data_X
            augmented_data_Y[data_Y.shape[0]:]= data_Y
            for i in range(data_X.shape[0]):
                W,V =self.random_w_and_v(data_X.shape[-1], data_X.shape[-1])
                augmented_data_X[i]= np.linalg.inv(V).dot(W).dot(data_X[i])
                augmented_data_Y[i]= data_Y[i]
            return augmented_data_X, augmented_data_Y 
        else:
            W,V = self.random_w_and_v(data_X.shape[-1], data_X.shape[-1])
            augmented_data_X= np.linalg.inv(V).dot(W).dot(data_X.T).T
            augmented_data_X= np.vstack((augmented_data_X, data_X))
            data_Y= np.hstack((data_Y, data_Y))
            return augmented_data_X, data_Y
        
    def get_config(self):
        return self.config
    

def test():
    import pandas as pd
    from glob import glob
    #this function load the n-th dataset by combining train and test data
    #it returns input data and labels
    def load_data(ID):
        data_explanation = pd.read_csv("data/DataSummary.csv")

        #reduce ID by one because of indexing
        ID = ID-1

        if len(data_explanation["ID"]) < ID:
            raise Exception("There is no dataset with this ID.")

        data_X=None
        data_Y=None

        if len(glob(f'data/UCRArchive_2018/'+ data_explanation["Name"][ID] +f'*' + "/*.tsv"))<1:
            raise Exception("The dataset couldn't be found.")

        #find
        for path in glob(f'data/UCRArchive_2018/'+ data_explanation["Name"][ID] +f'*' + "/*.tsv"):
            tmp = np.array(pd.read_csv(path, header=None,sep="\t"))

            if data_X is None:
                data_X = tmp[:,1:]
                data_Y = tmp[:,0]
            else:
                data_X=np.vstack((data_X,tmp[:,1:]))
                data_Y=np.hstack((data_Y,tmp[:,0]))


        if np.isnan(data_X).any() or np.isnan(data_Y).any() :
            raise Exception("Dataset has variable sequence length")

        #extract dimension
        seq_len= data_X.shape[1]
        n=data_X.shape[0]  

        return data_X, data_Y, data_explanation["Name"][ID]
    
    
    augment= RandomAugmentation(0.50,0.25,0.25,augment_each=True)
    X, y, name = load_data(1)
    X1, y1 = augment.augment(X,y)
    print(X1.shape)
    print(y1.shape)

    augment= RandomAugmentation(0.50,0.25,0.25,augment_each=False)
    X, y, name = load_data(1)
    X2, y2 = augment.augment(X,y)
    print(X2.shape)
    print(y2.shape)
    
    #print("amount of equal sequences")
    #print((X1[781:]==X2[781:]).sum()/176)
    #print((X1[:781]==X2[:781]).sum()/176)