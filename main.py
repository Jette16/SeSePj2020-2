import argparse
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
import time 
import pickle
from utils.utils import create_directory
from utils.utils import calculate_metrics#
from utils.utils import calculate_metrics1
from utils.utils import read_all_datasets
from utils.utils import check_dir
from utils.utils import generate_save_paths
from utils.utils import generate_results_overview
from utils.utils import plot_epochs_overview
from utils.utils import copy_eval_results_and_check_best_models
from utils.constants import ROOT_DIR
from utils.constants import ARCHIVE_NAMES
from utils.constants import dataset_names_for_archive
from utils.constants import ITERATIONS
from utils.constants import CV
from utils.constants import AUG
from utils.constants import RANDOM_AUG_PARAMETERS
from utils.constants import PRA_N
from utils.constants import PER_N
from utils.constants import PER_PROB
from utils.constants import CLASSIFIERS
from utils.constants import CLS_EPOCHS
from augmentation import RandomAug
from augmentation import PartlyRandomAugPath as PartlyRandomAug
from augmentation import PermutationAugPath as PermutationAug




def cv_fit_classifier_aug(args,augmentator,datasets_dict,dataset_name,classifier_name,epochs,output_directory,k):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    
    
    #concatenate all data
    X=np.concatenate((x_train, x_test))
    y=np.concatenate((y_train, y_test))
    
    #generate and save paths for all data
    gen_time=0
    if (type(augmentator)==type(PartlyRandomAug.PartlyRandomAug(1))) or (type(augmentator)==type(PermutationAug.PermutationAug(1, 0.1))):
       time0 = time.time() 
       index=np.arange(start=0, stop=len(y), step=1)
       generate_save_paths(X, y, index, output_directory)   
       gen_time=time.time() -  time0
       print('generation duration:',int(gen_time) )                                                                     
                                                                                
    #index=np.arange(start=0, stop=len(y), step=1)
    #generate_save_paths(X, y, index, output_directory)

    #save all the predictions and the corresponding true class
    predicted_y = []
    expected_y = []
    
    #start of CV
    start_time0 = time.time() 
    
    #split the data into k folds keeping the class imbalance
    skf = StratifiedKFold(n_splits=k)
    i=0
    aug=0
    #training and validation on each fold
    for train, test in skf.split(X, y):
        i+=1
        #print(train,test)
        x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        isplit='split'+str(i)+'/'
        print('\t\t\t\t'+isplit[:-1])
        output=output_directory+'/'+isplit
        #print(output)
        create_directory(output)
        original=len(y_train)
        augmentated=0
        if augmentator is not None:
            #do augmentation
            #start of CV
            print('augmentation begin...')
            print('original data:',original)
            start_time1 = time.time() 
         
                
            if (type(augmentator)==type(PartlyRandomAug.PartlyRandomAug(1))) or (type(augmentator)==type(PermutationAug.PermutationAug(1, 0.1))):
                                                                                    indices=train
            
                                                                                    with open(output_directory + "_paths.txt", 'rb') as pickle_file:
                                                                                        paths= pickle.load(pickle_file)
                                                                                    with open(output_directory +"path_indices.txt", 'rb') as pickle_file:
                                                                                        path_indexes= pickle.load(pickle_file)
                                                                                    x_train, y_train = augmentator.augment(x_train, y_train,indices, paths, path_indexes)
            else:
                x_train, y_train = augmentator.augment(x_train, y_train)
                
            augmentated=len(y_train)
            print('augmentated data:',augmentated)
            aug_time = time.time() - start_time1
            print('augmentation duration:',round(aug_time,2))
            aug+=aug_time
            
            if gen_time!=0:
                aug+=gen_time
        
        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    
        # transform the labels from integers to one hot vectors
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        
        
    
        # save orignal y because later we will use binary
        y_true = np.argmax(y_test, axis=1)
        
    
        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
        input_shape = x_train.shape[1:]
        
       
        
        classifier = create_classifier(args,classifier_name, epochs,input_shape, nb_classes, output)
    
        classifier.fit(x_train, y_train, x_test, y_test, y_true)
        
        y_pred=classifier.predict(x_test, y_true,x_train,y_train,y_test,return_df_metrics = False)
        
        # convert the predicted from binary to integer        
        y_pred = np.argmax(y_pred , axis=1)
        
        if (y_pred.shape==y_true.shape):
            predicted_y.extend(y_pred)
            expected_y.extend(y_true)
        else:
            raise Exception("FALSE: y_pred.shape==y_true.shape.")
              
        
    #totalduration=sum(durations)   
    totalduration = time.time() - start_time0
    df_metrics = calculate_metrics1(expected_y,predicted_y,totalduration,aug,original,augmentated)
    df_metrics.to_csv(output_directory + 'CV_metrics.csv', index=False)

    #print('Model saved:',output_directory)
    
    print('CV DONE!')
    print(df_metrics)
    
    return aug


    
def fit_classifier_aug(args,augmentator,datasets_dict,dataset_name,classifier_name,epochs,output_directory):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    
    if augmentator is not None:
            #do augmentation
            print('augmentation begin...')
            start_time1 = time.time() 
            x_train, y_train = augmentator.augment(x_train, y_train)
            aug_time = time.time() - start_time1
            print('augmentation done.')

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    
    classifier = create_classifier(args,classifier_name, epochs,input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    
    y_pred=classifier.predict(x_test, y_true,x_train,y_train,y_test,return_df_metrics = False)
    
    # convert the predicted from binary to integer        
    y_pred = np.argmax(y_pred , axis=1)
    
    return y_pred, y_true
   



def create_classifier(args,classifier_name,epochs, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':#
        from classifiers import resnet
        return resnet.Classifier_RESNET(epochs,output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':#
        from classifiers import encoder
        return encoder.Classifier_ENCODER(epochs,output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN #modified
        from classifiers import cnn
        from classifiers import cnnES
        if args.es_patience is None:
            #print('without early stopping')
            return cnn.Classifier_CNN(epochs,output_directory, input_shape, nb_classes, verbose)
        else: 
            print('with early stopping')
            return cnnES.Classifier_CNN(args.es_patience,epochs,output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
    
    
def run_iterations(args,augmentator,augmentator_name,tmp_output_directory,iterations,datasets_dict,classifier_name,epochs,start):
    print('\t\twithout augmentation: ', augmentator_name)
                        
    for dataset_name in dataset_names_for_archive[ARCHIVE_NAMES[0]]:
   
           print('\t\t\tdataset_name: ', dataset_name)
           
           upper_dir=tmp_output_directory + augmentator_name+'/' + dataset_name
           
           done=check_dir(upper_dir)
           
           if not done: 
               #save all the predictions and the corresponding true class
               predicted_y = []
               expected_y = []
                                   
               for iter in range(iterations):
                   print('\t\t\t\titer', iter)               
                   trr = '_itr_' + str(iter)
                       
                   output_directory = upper_dir + '/'+trr+'/'
                   #print(output_directory)
                   
                   create_directory(output_directory)
                   
                   y_pred,y_true=fit_classifier_aug(args,augmentator,datasets_dict,dataset_name,classifier_name,epochs,output_directory)
                   
   
                   print('\t\t\t\tDONE')
                   
                   # the creation of this directory means
                   create_directory(output_directory + '/DONE')
                   
                   if (y_pred.shape==y_true.shape):
                       predicted_y.extend(y_pred)
                       expected_y.extend(y_true)
                   else:
                       raise Exception("FALSE: y_pred.shape==y_true.shape.")
                         
   
               #totalduration=sum(durations)   
               totalduration = time.time() - start
               df_metrics = calculate_metrics(expected_y,predicted_y,totalduration)
               df_metrics.to_csv(upper_dir + '/avg_metrics.csv', index=False)
               create_directory(upper_dir + '/DONE')
           
               #print('Model saved:',upper_dir[len(ROOT_DIR):])
               
               print('iterations DONE!')
               print(df_metrics)
    
def run_cv(args,augmentator,augmentator_name,tmp_output_directory,datasets_dict,classifier_name,epochs,start,cv):
    print('\t\taugmentator_name: ', augmentator_name)                  
                       
    for dataset_name in dataset_names_for_archive[ARCHIVE_NAMES[0]]:

        print('\t\t\tdataset_name: ', dataset_name)
                
        output_directory = tmp_output_directory + augmentator_name + '/'+dataset_name
        
        #check_dir(output_directory)
        done=check_dir(output_directory)
        
        if not done:
        
            create_directory(output_directory)
            
            cv_fit_classifier_aug(args,augmentator,datasets_dict,dataset_name,classifier_name,epochs,output_directory,cv)
            
            create_directory(output_directory + '/DONE')

############################################### main



parser = argparse.ArgumentParser(description='Evaluate data augmentation')
#experiment approach
parser.add_argument('--approach', dest='approach',type=int,
                    help='normal(1) or CV(2) approach to conduct evaluation')
parser.add_argument('--iter', dest='iter',type=int,
                    help='nb of random init if do approach 1, default value in constants.py is used if not given')
parser.add_argument('--cv', dest='cv',type=int,
                    help='nb of cv if do approach 2, default value in constants.py is used if not given')
#augmentation method [RandomAug, PartlyRandomAug,PermutationAug,allAug]
parser.add_argument('--aug', dest='aug', 
                    help='augmentation method')
#classiefiers [allCls]
parser.add_argument('--cls', dest='cls', 
                    help='classiefer')

parser.add_argument('--cls_epochs', dest='cls_epochs', type=int,
                    help='epochs of training')
parser.add_argument('--es_patience', dest='es_patience', type=int,
                    help='early stopping patience')

parser.add_argument('--generate_results_overview', dest='generate_results_overview', action='store_true',
                    help='generate_results_overview')
parser.add_argument('--plot_epochs_overview', dest='plot_epochs_overview', action='store_true',
                    help='plot_epochs_overview')


def main():
    
    start=time.time()
    
    args = parser.parse_args()
    
    
    
    if args.approach == 1:
        print('Conduct evaluation using approach 1.')
        
        epochs=''
        
        if args.iter is not None:
            
            iterations = args.iter 
        else:
            iterations = ITERATIONS
            
        if args.cls_epochs is not None:
            epochs = args.cls_epochs
       
            
        
        if args.cls is not None:
            
            if args.cls=='allCls':
  
                for classifier_name in CLASSIFIERS:
                    
                    ARCHIVE_NAME=ARCHIVE_NAMES[0]
                    #print('\tarchive_name', ARCHIVE_NAME)
                    if epochs =='':
                        epochs = CLS_EPOCHS[classifier_name]
                        
                    if args.es_patience is None:
                        cl=classifier_name + '_ep'+str(epochs)
                    else:
                        cl=classifier_name + '_ep'+str(epochs)+'_patience'+str(args.es_patience )
                        
                        
                    print('\tclassifier_name', cl)
        
                        
                        
                    datasets_dict = read_all_datasets(ROOT_DIR, ARCHIVE_NAME)
        
                    tmp_output_directory = ROOT_DIR + '/results/' + cl + '/approach1_iter'+str(iterations) +'/'  
                    
                    if args.aug=='noAug':
                        augmentator=None
                        augmentator_name='NoAug'
                        
                        run_iterations(args,augmentator,augmentator_name,tmp_output_directory,iterations,datasets_dict,classifier_name,epochs,start)
                        
                        
                    if args.aug=='allAug':
                        
                        for aug in AUG:                              
                            
                            if aug=='Random':
                                
                                for parameters in RANDOM_AUG_PARAMETERS:
                                      
                                      diagonal = parameters[0]
                                      down = parameters[1]
                                      right = parameters[2]
                                      each = parameters[3]
                                      n = parameters[4]
                                      augmentator = RandomAug.RandomAug(n,diagonal,down,right,each)
                                      augmentator_name = aug+'_diag'+str(diagonal)+'_down'+str(down)+'_right'+str(right)+'_each'+str(int(each))+'_n'+str(n)
                            if aug=='PRA':
                                for n in PRA_N:
                                    augmentator=PartlyRandomAug.PartlyRandomAug(n)
                                    augmentator_name = aug+'_n'+str(n)
                                    
                            if aug=='Per':
                                for n in PER_N:
                                    for prob in PER_PROB:
                                        augmentator=PermutationAug.PermutationAug(n,prob)
                                        augmentator_name = aug+'_n'+str(n)+'_prob'+str(prob)
                            
                            run_iterations(args,augmentator,augmentator_name,tmp_output_directory,iterations,datasets_dict,classifier_name,epochs,start)
                                    
    if args.approach == 2:
        print('Conduct evaluation using approach 2.')
        epochs=''
        if args.cv is not None:
            
            cv = args.cv
        else:
            cv = CV
            
        if args.cls_epochs is not None:
            epochs = args.cls_epochs
        
        
        if args.cls is not None:
            
            if args.cls=='allCls':
  
                for classifier_name in CLASSIFIERS:
                    
                    
                   
                    ARCHIVE_NAME=ARCHIVE_NAMES[0] 
                    
                    if epochs =='':
                        epochs = CLS_EPOCHS[classifier_name]
                        
                    if args.es_patience is  None:
                        cl=classifier_name + '_ep'+str(epochs)
                    else:
                        cl=classifier_name + '_ep'+str(epochs)+'_patience'+str(args.es_patience )
                        
                        
                    print('\tclassifier_name', cl)
        
                    datasets_dict = read_all_datasets(ROOT_DIR, ARCHIVE_NAME)
                        
                    tmp_output_directory = ROOT_DIR + '/results/' + cl + '/approach2_cv'+str(cv)+'/'
                    
                     
                    if args.aug=='noAug':
                        augmentator=None
                        augmentator_name='NoAug'
                        
                        run_cv(args,augmentator,augmentator_name,tmp_output_directory,datasets_dict,classifier_name,epochs,start,cv)
                    
                    if args.aug=='allAug':
                        
                        for aug in AUG:                              
                            
                            if aug=='Random':
                                for parameters in RANDOM_AUG_PARAMETERS:
                                      
                                      diagonal = parameters[0]
                                      down = parameters[1]
                                      right = parameters[2]
                                      each = parameters[3]
                                      n = parameters[4]
                                      augmentator = RandomAug.RandomAug(n,diagonal,down,right,each)
                                      augmentator_name = aug+'_diag'+str(diagonal)+'_down'+str(down)+'_right'+str(right)+'_each'+str(int(each))+'_n'+str(n)
                                      
                            if aug=='PRA':
                                for n in PRA_N:
                                    augmentator=PartlyRandomAug.PartlyRandomAug(n)
                                    augmentator_name = aug+'_n'+str(n)
                                    
                            if aug=='Per':
                                for n in PER_N:
                                    for prob in PER_PROB:
                                        augmentator=PermutationAug.PermutationAug(n,prob)
                                        augmentator_name = aug+'_n'+str(n)+'_prob'+str(prob)
                                    
                            
                            run_cv(args,augmentator,augmentator_name,tmp_output_directory,datasets_dict,classifier_name,epochs,start,cv)
                                    
    if args.generate_results_overview:
        generate_results_overview()
        copy_eval_results_and_check_best_models()
        print('Results overview generated.')
    if args. plot_epochs_overview:
        plot_epochs_overview()     
        print('Epochs overview plotted.')                                                 
                                                        
                             
                                    
        

if __name__ == '__main__':
    main()


