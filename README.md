# Hierarchical Machine Learning Model for in Hospital Moratality Rate Prediction
This project is the final work in the "Machine Learning in Healtcare - 097248" course.

## Abstract
In this work, we propose a Hierarchical ML model for ICU mortality rate prediction. We believe that the ramification of certain health-conditions can aid the model’s ability to predict the mortality rate of ICU patients. Our approach is a novel take on an existing task - in-hospital mortality prediction, and one we strongly believe can alter the decision making process of ICU care and treatment trajectory. This is a proof of concept that building hierarchical models that include more conditions than just cardiac conditions might be beneficial for making better classification models.

<p align="center">
<img align="center" src="https://user-images.githubusercontent.com/74211354/98663062-444aad80-2351-11eb-9d5a-c09be1edfaf3.png" width="50%"></img>
</p>

## Paper
Can be found in Project.pdf **add link**

## Repository 
* **src** - implementation of all neccesary modules for the project.
  * _hr_analysis.py_ - handle of signal loading, pre-processing and extraction of features from signals.
  * _load_data.py_ - sql handler for querying the MIMIC III dataset on an azure VM.
  * _load_sigs.py_ - utils file for hr_analysis.WaveForms class.
  * _prediction.py_ - module for training and evaluating different ML models.
  * _project.py_ - main file, sums up all the other files to a final implementation of the project's idea. Train and test ML models on predicting cardiac conditions within patients and later on predicting in-hospital mortality.
  * _queries.py_ - queries used to extract data from MIMIC III clinical dataset.
* **data** - relevant data for the training of the models and feature extraction.
  * _features_clinical.csv_ - clinical features for the patients in the sample.
  * _in_hospital_mortality.csv_ - table of dead patients and the admissions in which they died.
  * _matching.csv_ - matching table of patients with corresponding ECG signals entries.
  * _matching_numeric.csv_ - matching table of patients with corresponding numeric signal entries.
  * _wf_total.csv_ - features extracted from waveforms for all patients in the sample.
