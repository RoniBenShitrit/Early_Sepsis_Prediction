# Early_Sepsis_Prediction
Predicting sepsis using machine learning models
This work focused on predicting sepsis in patients. It involved analyzing the data distribution and missing values, performing feature engineering, testing various techniques and post-analysis. No data imputation was required for the selected model. Multiple models, including XGBoost, Random Forest, and AdaBoost, were trained and evaluated, with XGBoost emerging as the best performer based on F1 scores. 
<br>
#### Data
The dataset contains a total of 41 features collected for each patient and for every hour the patient spent in the ICU. These features include demographic information, vital signs, laboratory results, and other clinical information about patients who have been admitted to the ICU and the label feature. The label indicates whether the patient will be diagnosed with sepsis within six hours and is collected for each patient and for every hour. The dataset contained 30,000 samples, with 20,000 train samples and 10,000 test samples. 
<br> 
#### Task
For each patient, predict whether he will be diagnosed with sepsis within six hours.

