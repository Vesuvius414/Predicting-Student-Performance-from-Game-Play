# Predicting Student Performance from Game Play
This repository holds an attempt to predict whether a student gets a question correct or not based on the dataset Kaggle has provided for this competition: Predict Student Performance from Game Play | Kaggle.

## Overview
The challenge for this Kaggle competition is to predict how a student performs on questions from features involving their game session. This problem resembles a binary classification task. I performed feature engineering by aggregating event-level interactions into session-level counts, applied one-hot encoding to categorical actions, and utilized a Gradient Boosting model (LightGBM) to capture non-linear patterns in student behavior. The model achieved a validation accuracy of about 70.44% and an F1 Score of 0.8198.

## Summary of Work Done
### Data
- Type - train.csv file with 20 features for game interactions that result in a binary setting indicating whether a student got a question correct or not
- Size - the train.csv file contains over 26 million rows of data, but the sample training set consisted of the first 5 million rows for easier handling
- Instances - The data was split 60% Training, 20% Validation, and 20% Test based on unique session IDs
#### Preprocessing / Clean up
- raw event data was grouped by session ID to sum occurrences of specific event types.
- session_id was standardized to string types to ensure seamless merging between the feature matrix and labels
- StandardScaler was applied to all numerical features to normalize the "click counts" to a mean of 0 and standard deviation of 1
- reindex was utilized during inference to ensure the test dataset columns matched the training features exactly, filling missing columns with 0
#### Data Visualization
- Matplotlib was used to create histograms to visualize some features between classes
- successful students showed a higher density of navigation and interaction events, suggesting that higher game engagement is a strong predictor of success.
- outliers were identified with a threshold of the mean plus 3 standard deviations for a better understanding of distributions with tail ends
### Problem Formulation
- Input - vector of aggregated and scaled event counts per session
- Output - binary classification (1 = correct, 0 = incorrect)
- Model - LightGBM (Light Gradient Boosting Machine) - deals with tabular data, missing values, and class imbalance efficiently and quickly
- Hyperparameters - used 100 estimators with a learning rate of 0.05
### Training
- Software/Hardware - trained and developed in VSCode on local PC (laptop)
- 5 million sample size gathered using chunking for ease of loading the dataset
- Difficulties - main difficulty was syncing session IDs between train.csv and train_labels.csv because they were formatted differently on both files, but this was solved by explicit type casting during the merge phase
### Performance Comparison
- Accuracy - 0.704493
- Precision - 0.709739
- Recall - 0.970242
- F1 Score - 0.819794
- ROC AUC - 0.644371

### Conclusions
- features that show students’ activity are shown to be really important, and they indicate that engagement as a whole are significant in determining their performance on questions
- LightGBM works well with datasets like this one, one with high dimensionality and class imbalance
### Future Work
- more models could be used to explore this dataset and perhaps get a better accuracy
- features could be separated by level groups
## How to Reproduce Results
- Download the Kaggle dataset for this competition here: Predict Student Performance from Game Play | Kaggle.
- Make sure you have train.csv, train_labels.csv, and test.csv in the directory you’re working in
- Run the Python script in this repository
- The script will result in a submission file similar to sample_submission.csv
### Overview of Files in Repository
- predicting_student_performance.ipynb - script with all steps performed on data
- train.csv - training data where most of the data lies
- train_labels.csv - data applying binary conditions to respective sessions
- test.csv - data to test your model on
- sample_submission.csv - file to model your submission file on
- submission.csv - resulting file containing model predictions
### Software Setup
- packages required: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm


