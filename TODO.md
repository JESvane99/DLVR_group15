# TODO:

## Report usage

save figures and relevant data from following todo list

- [x] Verify chosen data quality (Lab 3)
  - [x] Visual inspection (including saved inspection figures)
  - [x] Mask/label imbalance (log true/false percentage of true mask)
  - [x] Image size distribution (used to argument for resizing)
  - [x] Cherrypick remove doublegangers and leeks in test and train data
  - [x] check for corrupt data
- [] Tensor statisticks after normalization
- [] Use torchmetrics BinaryJaccardIndex for IOU (Lab4b)
- [] Use Torchmetrics Dicescore (Lab4b)
- [] Use softmax to get probabilitymap before thresholding to binary
- [] Use optimizer (hyperparameter)
- [] Use learningrate scheduler (hyperparameter)
- 


## General oversights and notes

- [] adjust the train, validation, and test dataset size to reflect actual values instead of percentages.