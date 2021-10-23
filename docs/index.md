This work is part of the paper titled - "Minimal Vital Sensor Architectures for Early Warning of Sepsis in Telehealth Patients" (under review)

# Vital-SEP
A Sepsis Prediction Engine that employs Gradient Boosted Decision Tree (XGBoost) on features extracted from vitals obtained from wearable sensors

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
- [hospitalA(BIDMC)_sepsis_raw.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalA(BIDMC)_sepsis_raw.zip)
- [hospitalA(BIDMC)_controls_raw.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalA(BIDMC)_controls_raw.zip)
- [hospitalB(Emory)_sepsis_raw.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalB(Emory)_sepsis_raw.zip)
- [hospital B_(Emory)_controls_raw.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalB(Emory)_controls_raw.zip)

Curated dataset for this study
- [hospitalA(BIDMC)_sepsis_curated.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalA(BIDMC)_sepsis_curated.rar)
- [hospitalA(BIDMC)_controls_curated.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalA(BIDMC)_controls_curated.rar)
- [hospitalB(Emory)_sepsis_curated.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalB(Emory)_sepsis_curated.rar)
- [hospital B_(Emory)_controls_curated.zip](https://github.com/pprahul/Vital-SEP/blob/main/hospitalB(Emory)_controls_curated.rar)

### Usage
The Algorithm is implemented as a set of following three python modules:

#### 1: Building and training the XGBoost models using Hospital A datasets.
Module: [Tele-SEP-train-model.py](https://github.com/pprahul/Vital-SEP/blob/main/Tele-SEP-train-model.py)

Parameters:
Each of the 15 sensor configurations (S<sub>i</sub>)
Each of the 16 timing tuples (W,L)

Output: AUROC for each (S<sub>i</sub>,W,L)

For each sensor configuration the highest AUC yielding model is chosen to be validated in the next function

#### 2: Validating the model using Hospital B datasets.
Module: [Tele-SEP-ModelLoadRunOnly.py](https://github.com/pprahul/Vital-SEP/blob/main/Tele-SEP-ModelLoadRunOnly.py)

Parameters: 
each of the 15 sensor configurations (S<sub>i</sub>)
Best performing timing tuple (W<sub>AUC</sub>,L<sub>AUC</sub>) corresponding to S<sub>i</sub>.
  
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

Output: AUC and its difference from that obtained in function 1 (for each sensor configuration)

#### 3: Choosing the best performing minimal sensor configuration
Module automatation being implemented 

Parameters:
	AUROC threshold value AUC<sub>min</sub>
	Lead time threshold value L<sub>min</sub>

Output: From the list of Sensor configurations arranged in ascending order based on number and complexity of vitals, choose the first configuration S<sub>min</sub> for which AUROC obtained in module 1 and corresponding lead time are greater than or equal to their respective threshold values AUC<sub>min</sub>  and L<sub>min</sub>.

### Setup and runtime
Modules 1,2 and 3 are run once at the setup time and a subset of the best performing pre-trained and validated models corresponding to various sensor configurations are also provided in the [repository](https://github.com/pprahul/Vital-SEP/tree/main/trained-models/XGBoost). During runtime, the following algorithm is used to predict sepsis for a new patient.

Parameters: 
	Patient’s wearable sensor configuration S<sub>p</sub>
	Patient_vitals = new patient data
	Lead time = 3,4,5,6 hours

Subroutines:
Choose the Tele-SEP model that satisfies the patient’s wearable sensor configuration S<sub>p</sub>. For the sensor configuration S<sub>p</sub>, retrieve four sets of models M<sub>p</sub>3, M<sub>p</sub>4, M<sub>p</sub>5, M<sub>p</sub>6 corresponding to the four lead times. 
From each set choose the best performing model M<sub>p</sub>3<sup>*</sup>, M<sub>p</sub>4<sup>*</sup>, M<sub>p</sub>5<sup>*</sup>, M<sub>p</sub>6<sup>*</sup>. Run these on Patient_vitals to compute the sepsis probabilities.

Output: The maximum of the four sepsis probabilities and the corresponding lead time resulting from the above computation 

