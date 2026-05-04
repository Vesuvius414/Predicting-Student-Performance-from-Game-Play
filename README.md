# Predicting Student Performance from Game Play
This repository holds an attempt to predict whether a student gets a question correct or not based on the dataset Kaggle has provided for this competition: Predict Student Performance from Game Play | Kaggle.

## Overview
The challenge for this Kaggle competition is to predict how a student performs on questions from features involving their game session. This problem resembles a binary classification task, identifying whether a user in a session will get certain questions correct or not. I did feature engineering based on sessions which would be easier to rely on rather than features based on individual events like the original dataset has. I used the logistic regression, random forest classifier, and LightGBM models. All models performed with an ROC AUC of above 0.75, indicating strong ranking ability in distinguishing correct vs incorrect responses  (the LightGBM model performed the best with an ROC AUC of ~0.774.

## Summary of Work Done
### Data
- Type - The train.csv file is a tabular dataset with 20 features for game interactions that result in a binary setting indicating whether a student got a question correct or not
  - Each row resembled an individual event for a session
- Size - The train.csv file contains over 26 million rows of data, but the dataset the models trained on with the engineered features consisted of 424,116 sessions and questions 
- Instances - The data was split 70% training, 15% validation, and 15% testing based on unique session IDs
#### Preprocessing / Clean up
- Because the rows resembled individual events, feature engineering was performed to form variables that were based on sessions as a whole to better suit the problem at hand
- Features for new dataset:
  - session_id
  - correct
  - question
  - event_count
  - elapsed_time_sum
  - level_mean
  - event_entropy
  - time_per_event
  - events_per_second
  - question_difficulty
- StandardScaler was applied to the engineered features, mainly for better visualization and for the logistic regression model
#### Data Visualization
- Matplotlib was used to create bar graphs and histograms to visualize the class imbalance (~71% correct, ~29% incorrect), both generally and within the engineered features
<img width="516" height="374" alt="question_difficulty" src="https://github.com/user-attachments/assets/6221a2d6-c40a-4cd7-b525-8a9661f81abc" />
<img width="516" height="374" alt="event_count" src="https://github.com/user-attachments/assets/beed4161-9d0b-46b8-bb55-19528cde820b" />

  - Also used to look at accuracy per question and session
<img width="580" height="455" alt="Distribution of Session Accuracy" src="https://github.com/user-attachments/assets/526fadce-52bb-43a3-97e8-2ff0249a74fb" />
<img width="567" height="458" alt="Accuracy per Question" src="https://github.com/user-attachments/assets/6d36a481-d43d-49d4-94c0-b50c294a6a15" />

- The LightGBM model was also used to determine feature importance
<img width="716" height="455" alt="Feature Importance (Gain)" src="https://github.com/user-attachments/assets/fc5a21e6-0f9d-451d-b000-36d1dd5b1ed1" />

- Based on the histograms and the feature importance graph, question_difficulty and event_count are the most influential features when determining if a session user gets a question correct or not (presumably because they’re based on interaction and engagement)
### Problem Formulation
- Input - dataset of engineered features (refer to the above list)
- Output - binary classification (1 = correct, 0 = incorrect)
- Models:
  - Logistic Regression
    - Baseline linear classifier
    - Required feature scaling
    - Used to establish baseline performance
  - Random Forest Classifier
    - Nonlinear tree-based ensemble model
    - Captures feature interactions without scaling
  - LightGBM Classifier
    - Gradient boosting framework
    - Best performing model
    - Handles nonlinear relationships and feature interactions effectively
### Training
- Software/Hardware - trained and developed in VSCode on Jupyter notebook on local PC (laptop)
- Whole train.csv dataset loaded in chunks (memory efficient) then aggregated into engineered features to create a new dataset based on sessions for the models to train over
- Training splits - 70% training, 15% validation, 15% testing
- Model performance based on ROC AUC
- Performance Comparison
  - LogReg AUC: ~0.764
  - RF AUC: ~0.752
  - LGBM AUC: ~0.774
<img width="691" height="547" alt="ROC Curves" src="https://github.com/user-attachments/assets/610f4b9f-9651-4c5b-af78-8b03de354644" />

### Conclusions
- Behavioral features extracted from session logs are predictive of student correctness
- Time-based and interaction-based features were more important than static identifiers
- LightGBM was the most effective model due to its ability to capture nonlinear feature interactions
- Feature engineering had a larger impact than model selection alone
### Future Work
- More models could be explored and trained on the data
- There could be a way to incorporate event-based data in training to capture certain nuances
- More session-based features could be engineered for better accuracy
## How to Reproduce Results
- Download the files for this competition here: Predict Student Performance from Game Play | Kaggle.
- Make sure you have train.csv, train_labels.csv, test.csv, and sample_submission.csv in the directory you’re working in
- Run the Python script “predicting_student performance.ipynb” in this repository
- The script will result in a submission file similar to sample_submission.csv
### Overview of Files in Repository
- predicting_student performance.ipynb - script with all steps performed on data
- submission.csv - resulting file containing model predictions
### Software Setup
- packages required: pandas, numpy, matplotlib, scikit-learn, lightgbm
