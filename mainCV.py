from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
from utils.utils import calculate_metrics#
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import pandas as pd#
from sklearn.model_selection import cross_val_score#
from sklearn.model_selection import StratifiedKFold, KFold#
import time 




def cv_fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    
    #concatenate all data
    X=np.concatenate((x_train, x_test))
    y=np.concatenate((y_train, y_test))

    #save all the predictions and the corresponding true class
    predicted_y = []
    expected_y = []
    
    #start of CV
    start_time0 = time.time() 
    
    #split the data into k folds keeping the class imbalance
    skf = StratifiedKFold(n_splits=k)
    i=0
    #training and validation on each fold
    for train, test in skf.split(X, y):
        i+=1
        #print(train,test)
        x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
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
        
        isplit='split'+str(i)+'/'
        output=output_directory+isplit
        create_directory(output)
        
        classifier = create_classifier(classifier_name, input_shape, nb_classes, output)
    
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
    df_metrics = calculate_metrics(expected_y,predicted_y,totalduration)
    df_metrics.to_csv(output_directory + 'CV_df_metrics.csv', index=False)

   
    
    print('CV DONE!')
    print(df_metrics)
    
def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

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
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = 'C:/Users/admin/Desktop/SS/dl-4-tsc-master'

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + '/'

                    create_directory(output_directory)

                    fit_classifier()

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]
    cvN = sys.argv[5]
    k = int(cvN[-1:])
    
    
    if(k==0):
        print('We will not do cross-validation.')
    if(k!=0):
        print('We will do k-fold cross-validation with k=',str(k))

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr +cvN+ '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr,cvN)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
        
        if(k==0):
            fit_classifier()
            print('DONE')
        if(k!=0):
            cv_fit_classifier()
            
            #scores=cv_fit_classifier2()
            #print('scores',scores)

        

        

        # the creation of this directory means
        create_directory(output_directory + '/DONE')
