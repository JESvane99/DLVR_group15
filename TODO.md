# TODO:

## Report usage

save figures and relevant data from following todo list

- [] Verify chosen data quality (Lab 3)
  - [] Visual inspection (including saved inspection figures)
  - [] Mask/label imbalance (log true/false percentage of true mask)
  - [] Image size distribution (used to argument for resizing)
  - [] Cherrypick remove doublegangers and leeks in test and train data
  - [] check for corrupt data
- [] Tensor statisticks after normalization
- [] Use torchmetrics BinaryJaccardIndex for IOU (Lab4b)
- [] Use Torchmetrics Dicescore (Lab4b)
- [] Use softmax to get probabilitymap before thresholding to binary
- [] Use optimizer (hyperparameter)
- [] Use learningrate scheduler (hyperparameter)
- 


## General oversights and notes

- [] adjust the train, validation, and test dataset size to reflect actual values instead of percentages.