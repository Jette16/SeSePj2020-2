We used the following github project as base code structure https://github.com/hfawaz/dl-4-tsc. We added the augmentation methods and changed the main.py.
>@article{IsmailFawaz2018deep,
>  Title                    = {Deep learning for time series classification: a review},
>  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
>  journal                  = {Data Mining and Knowledge Discovery},
>  Year                     = {2019},
>  volume                   = {33},
>  number                   = {4},
>  pages                    = {917--963},
>}

# SeSePj2020-2
1). before start, please setup the enviroment according to utils/pip-requirements.txt
<br>
2). parameters for augmentation and classification could be modified in utils/constants.py <br>
3). you can choose your own UCR datasets by changing the list UNIVARIATE_DATASET_NAMES_2018 in utils/constants.py <br>
## Two experimental approaches:
### approach 1 (not used in our experiments): 
normal approach to run serval iterations on the original splits of the dataset
<br>
### approach 2 (we use this approach in our experiments): 
stratified cross validation, the average accuracy of cross validation is taken as evaluation metric
<br>
## Time Series Classification without augmentation:
### approach 1
python main.py --approach 1 --aug noAug --cls allCls --generate_results_overview<br>
### approach 2
python main.py --approach 2 --aug noAug --cls allCls --generate_results_overview<br>
## Time Series Classification with augmentation:
### approach 1
python main.py --approach 1 --aug allAug --cls allCls --generate_results_overview<br>
### approach 2
python main.py --approach 2 --aug allAug --cls allCls --generate_results_overview<br>
## Generate results overview anytime:
python main.py --generate_results_overview<br>
## Plot epochs_loss(& val_loss & accuracy & val_accuracy) overview anytime:
python main.py --plot_epochs_overview<br>
