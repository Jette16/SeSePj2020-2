# SeSePj2020-2
parameters for augmentation and classification could be modified in utils/constants.py <br>
## TSC without augmentation:
python main.py --approach 1 --aug noAug --cls allCls --generate_results_overview<br>
or<br>
python main.py --approach 2 --aug noAug --cls allCls --generate_results_overview<br>
## TSC with augmentation:
### approach 1
python main.py --approach 1 --aug allAug --cls allCls --generate_results_overview<br>
### approach 2
python main.py --approach 2 --aug allAug --cls allCls --generate_results_overview<br>
## generate results overview anytime:
python main.py --generate_results_overview<br>
## plot epochs_loss(&val_loss&accuracy&val_accuracy) overview anytime:
python main.py --plot_epochs_overview<br>
