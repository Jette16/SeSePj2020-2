# SeSePj2020-2
1). before start, please create the env according to utils/pip-requirements.txt
<br>
2). parameters for augmentation and classification could be modified in utils/constants.py <br>
## two experimental approaches:
approach 1: normal approach to run serval iterations on the original data
<br>
approach 2(we use this approach in the report): stratified cross validation, the average accuracy of cross validation is taken as the evaluation metric
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
## generate results overview anytime:
python main.py --generate_results_overview<br>
## plot epochs_loss(&val_loss&accuracy&val_accuracy) overview anytime:
python main.py --plot_epochs_overview<br>
