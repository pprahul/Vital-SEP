# Tele-SEP
A Sepsis Prediction Engine for Telehealth applications.

An early warning system for sepsis called the Tele-Sepsis Prediction Engine (Tele-SEP), uses plug-and-play machine learning algorithms to compute intrinsic correlations between the changes in vital signs, designed to predict sepsis up to six hours prior to its onset. Tele-SEP was trained and validated on independent datasets drawn from the widely accepted MIMIC-II database.


### Input data
Input to Tele-SEP can be hourly measurements of any one or a combination of the following vital parameters:
- heart rate
- SpO2 (blood oxygen)
- respiratory rate
- temperature

The input should be formatted so that the measurements span a minimum of 3 hours and a maximum of 6 hours.

### Output
Tele-SEP model predicts the probability of occurance of sepsis based on the input vital parameter measurements.

### Trained Models
Pre-trained XGBoost models for different lead times of prediction ranging from 3 hour to 6 hours is provided for use at [our github repository](https://github.com/pprahul/Tele-SEP/tree/main/trained-models/XGBoost)

### Usage
Program to load pre-trained models and predict sepsis is provided [here](https://github.com/pprahul/Tele-SEP/blob/main/Tele-SEP-ModelLoadRunOnly.py). An example is given below.

```markdown
import pickle

model_filename = 'trained-models/XGBoost/XGB-Model-PPG-RR-Temp-L6-M4-verified.sav'

# load the model from disk
loaded_model = pickle.load(open(model_filename, 'rb'))

# make predictions for test data
y_pred = loaded_model.predict(X_test)

# print classification report 
print(classification_report(y_test, y_pred)) 

#confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

```


### Datasets
*Hospital A*
[Non-sepsis Patients](https://github.com/pprahul/Tele-SEP/blob/main/trainingA_nonsepsis.zip)
[Sepsis Patients](https://github.com/pprahul/Tele-SEP/blob/main/trainingA_sepsis_15h.zip)

*Hospital B*
[Non-sepsis Patients](https://github.com/pprahul/Tele-SEP/blob/main/trainingB_nonsepsis.zip)
[Sepsis Patients](https://github.com/pprahul/Tele-SEP/blob/main/trainingB_sepsis_15h.zip)

This project is supported by [Amrita Vishwa Vidyapeetham](https://www.amrita.edu/).



