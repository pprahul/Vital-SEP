This work is part of the paper titled - "Minimal Vital Sensor Architectures for Early Warning of Sepsis in Telehealth Patients" (under review)

# Tele-SEP
A Sepsis Prediction Engine for Telehealth applications that employs Gradient Boosted Decision Tree (XGBoost) on features extracted from vitals obtained from wearable sensors

### Programming Platforms Used
- Python 3
- MatplotLib
- NumPy
- Pandas

### Input Dataset
Sequence of hourly measurements of the following vital signs:
- heart rate
- respiratory rate
- SpO2 (blood oxygen)
- temperature
These measurements obtained from patients of two different hospitals are contained in the following zip files. Each zip file when extracted generates the individual patient data files.

The raw files refer to Physionet CinC 2019 database, which are then preprocessed (as per inclusion exclusion criteria etc.) to generate the curated datasets used for this study.

The input should be formatted so that the measurements span a minimum of 3 hours and a maximum of 6 hours.

Input data files are zipped and can be accessed from the repository:
Raw Dataset
- hospitalA(BIDMC)_sepsis_raw.zip
- hospitalA(BIDMC)_controls_raw.zip
- hospitalB(Emory)_sepsis_raw.zip
- hospital B_(Emory)_controls_raw.zip 

Curated dataset for this study
- hospitalA(BIDMC)_sepsis_curated.zip
- hospitalA(BIDMC)_controls_curated.zip
- hospitalB(Emory)_sepsis_curated.zip
- hospital B_(Emory)_controls_curated.zip


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




