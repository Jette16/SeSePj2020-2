# change this directory for your machine
ROOT_DIR = 'D:/Shi/RQ/SeSe'

#datasets to use
UNIVARIATE_DATASET_NAMES_2018 = ['BeetleFly', 'BirdChicken','Beef','Wine','Car','ECG200','ArrowHead','DiatomSizeReduction','Fish','Adiac']
ARCHIVE_NAMES = ['UCRArchive_2018']

dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018}                              


#approach 1
ITERATIONS = 4  # nb of random runs for random initializations
#approach 2
CV = 4   # k-fold cross validation

#augmentation methods
AUG = ['PRA','Per','Random']#'Random','PRA','Per'
#parameters for RandomAug
RANDOM_AUG_PARAMETERS=[[0.5,0.25,0.25,True,1],[0.5,0.25,0.25,True,10]]


#parameters for PartlyRandomAug
PRA_N=[1]

#parameters for PermutationAug
PER_N=[1]
PER_PROB=[0.1]


#parameters for time series classification
CLASSIFIERS = ['cnn'] #'cnn','resnet', 'encoder', 'fcn', 'mlp',  'tlenet', 'mcnn', 'twiesn',  'mcdcnn', 'inception']
CLS_EPOCHS = {'cnn':2000,'fcn':2000,'encoder':100,'resnet':1500}

#not related to our project
MTS_DATASET_NAMES = ['ArabicDigits', 'AUSLAN', 'CharacterTrajectories', 'CMUsubject16', 'ECG',
                     'JapaneseVowels', 'KickvsPunch', 'Libras', 'NetFlow', 'UWave', 'Wafer', 'WalkvsRun']
UNIVARIATE_DATASET_NAMES = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                            'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
                            'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
                            'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
                            'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
                            'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
                            'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                            'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                            'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                            'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                            'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                            'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']
dataset_types = {'ElectricDevices': 'DEVICE', 'FordB': 'SENSOR',
                 'FordA': 'SENSOR', 'NonInvasiveFatalECG_Thorax2': 'ECG',
                 'NonInvasiveFatalECG_Thorax1': 'ECG', 'PhalangesOutlinesCorrect': 'IMAGE',
                 'HandOutlines': 'IMAGE', 'StarLightCurves': 'SENSOR',
                 'wafer': 'SENSOR', 'Two_Patterns': 'SIMULATED',
                 'UWaveGestureLibraryAll': 'MOTION', 'uWaveGestureLibrary_Z': 'MOTION',
                 'uWaveGestureLibrary_Y': 'MOTION', 'uWaveGestureLibrary_X': 'MOTION',
                 'Strawberry': 'SPECTRO', 'ShapesAll': 'IMAGE',
                 'ProximalPhalanxOutlineCorrect': 'IMAGE', 'MiddlePhalanxOutlineCorrect': 'IMAGE',
                 'DistalPhalanxOutlineCorrect': 'IMAGE', 'FaceAll': 'IMAGE',
                 'ECG5000': 'ECG', 'SwedishLeaf': 'IMAGE', 'ChlorineConcentration': 'SIMULATED',
                 '50words': 'IMAGE', 'ProximalPhalanxTW': 'IMAGE', 'ProximalPhalanxOutlineAgeGroup': 'IMAGE',
                 'MiddlePhalanxOutlineAgeGroup': 'IMAGE', 'DistalPhalanxTW': 'IMAGE',
                 'DistalPhalanxOutlineAgeGroup': 'IMAGE', 'MiddlePhalanxTW': 'IMAGE',
                 'Cricket_Z': 'MOTION', 'Cricket_Y': 'MOTION',
                 'Cricket_X': 'MOTION', 'Adiac': 'IMAGE',
                 'MedicalImages': 'IMAGE', 'SmallKitchenAppliances': 'DEVICE',
                 'ScreenType': 'DEVICE', 'RefrigerationDevices': 'DEVICE',
                 'LargeKitchenAppliances': 'DEVICE', 'Earthquakes': 'SENSOR',
                 'yoga': 'IMAGE', 'synthetic_control': 'SIMULATED',
                 'WordsSynonyms': 'IMAGE', 'Computers': 'DEVICE',
                 'InsectWingbeatSound': 'SENSOR', 'Phoneme': 'SENSOR',
                 'OSULeaf': 'IMAGE', 'FacesUCR': 'IMAGE',
                 'WormsTwoClass': 'MOTION', 'Worms': 'MOTION',
                 'FISH': 'IMAGE', 'Haptics': 'MOTION',
                 'Epilepsy': 'HAR', 'Ham': 'SPECTRO',
                 'Plane': 'SENSOR', 'InlineSkate': 'MOTION',
                 'Trace': 'SENSOR', 'ECG200': 'ECG',
                 'Lighting7': 'SENSOR', 'ItalyPowerDemand': 'SENSOR',
                 'Herring': 'IMAGE', 'Lighting2': 'SENSOR',
                 'Car': 'SENSOR', 'Meat': 'SPECTRO',
                 'Wine': 'SPECTRO', 'MALLAT': 'SIMULATED',
                 'Gun_Point': 'MOTION', 'CinC_ECG_torso': 'ECG',
                 'ToeSegmentation1': 'MOTION', 'ToeSegmentation2': 'MOTION',
                 'ArrowHead': 'IMAGE', 'OliveOil': 'SPECTRO',
                 'Beef': 'SPECTRO', 'CBF': 'SIMULATED',
                 'Coffee': 'SPECTRO', 'SonyAIBORobotSurfaceII': 'SENSOR',
                 'Symbols': 'IMAGE', 'FaceFour': 'IMAGE',
                 'ECGFiveDays': 'ECG', 'TwoLeadECG': 'ECG',
                 'BirdChicken': 'IMAGE', 'BeetleFly': 'IMAGE',
                 'ShapeletSim': 'SIMULATED', 'MoteStrain': 'SENSOR',
                 'SonyAIBORobotSurface': 'SENSOR', 'DiatomSizeReduction': 'IMAGE'}

themes_colors = {'IMAGE': 'red', 'SENSOR': 'blue', 'ECG': 'green',
                 'SIMULATED': 'yellow', 'SPECTRO': 'orange',
                 'MOTION': 'purple', 'DEVICE': 'gray'}
