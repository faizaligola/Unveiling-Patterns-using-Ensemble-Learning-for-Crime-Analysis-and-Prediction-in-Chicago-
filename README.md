# Unveiling Patterns using Ensemble Learning for Crime Analysis and Prediction in Chicago
### Dataset
https://www.kaggle.com/code/onlyrohit/criminal-activity-hotspots-identification/data?select=2013.csv
## Libraries
The code imports essential Python libraries for data manipulation, visualization, and machine learning, including NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Plotly.
```
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")
```
## Loading Data
The code snippet handles the compilation of data from multiple CSV files for different years into a single DataFrame. A DataFrame is created by reading data from a CSV file for the year 2022.
```
# The dataset was available in different csv files for each year, we created a single dataframe of records and saved the data frame on drive. 
data_location = [
    'C:\\Users\\Project\\2020.csv',
    'C:\\Users\\Project\\2021.csv',
    'C:\\Users\\Project\\2019.csv',
    'C:\\Users\\Project\\2018.csv',
    'C:\\Users\\Project\\2017.csv',
    'C:\\Users\\Project\\2016.csv',
    'C:\\Users\\Project\\2015.csv',
    'C:\\Users\\Project\\2014.csv'
]
df = pd.read_csv('C:\\Users\\Project\\2022.csv')
```
The code defines a concise function, day_col(x), that takes a date as input and returns the corresponding day of the week as an integer. It utilizes the strftime method to extract the day of the week and convert it into an integer.
```
def day_col(x):
    return int(x.strftime("%w"))
```
## Exploratory Data Analysis
```
df.head()
```
![1](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/4cec3a7e-8490-4c36-a332-cbf7507f67bc)

This code snippet utilizes Matplotlib to create a line plot that visualizes the monthly trend of crimes based on the provided DataFrame. The plot is configured to show the count of crimes over each month.
```
plt.figure(figsize=[16,5])

df.resample('M').size().plot(legend=False)
plt.xlabel('Months')
plt.ylabel('Crimes')
plt.show()
```
![2](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/a26e3900-aa7d-4acc-995f-104a58b4fc28)
This code snippet aims to visualize the distribution of crime types in the DataFrame. It utilizes the value_counts method to count occurrences of each unique value in the 'Primary Type' column and then creates a bar plot to illustrate the frequency of each crime type.
```
value_counts = df['Primary Type'].value_counts()

value_counts.plot(kind='bar')

plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Most occuring crimes')
plt.show()
```
![3](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/4ae6d70d-2591-487a-bc5e-bb57eeec999c)
This code snippet focuses on visualizing the distribution of crimes based on different location descriptions in the DataFrame. It employs the value_counts method to count occurrences of each unique value in the 'Location Description' column, then creates a bar plot to display the top 15 locations with the highest crime counts.
```
value_counts = df['Location Description'].value_counts()[:15]

value_counts.plot(kind='bar')

plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Crimes at different locations')
plt.show()
```
![4](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/4241f487-5b80-4b6b-85bb-b639adb13450)

## Method: Transformation (Feature Engineering)
We focuses on extracting date and time-related features from the 'Date' column in the DataFrame. It creates new columns such as 'Hour,' 'Month,' 'Year,' and 'Day' to enhance the temporal analysis of crime data. Additionally, it groups the data based on various features and calculates the count of crimes, categorizing them into levels.
```
df.Date=pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')
df.index = pd.DatetimeIndex(df.Date)

df['Hour']=df['Date'].apply(lambda x:x.hour)
df['Month']=df['Date'].apply(lambda x:x.month)
df['Year']=df['Date'].apply(lambda x:x.year)
df['Day'] = df['Date'].apply(day_col)

df2 = df.groupby(['Month','Day','District','Hour'], as_index=False).agg({"Primary Type":"count"})
df2['Level'] = df2['Primary Type'].apply(lambda x: 0 if x <= 10 else 1 if x <= 20 else 2)
final_df = df2[['Month','Day','Hour','District','Primary Type','Level']]
```
## Train-Test split
```
X = final_df.iloc[:,0:4].values 
y = final_df.iloc[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
```
## Method: Scalng the training dataset
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
## Method: Statistical Analysis (chi-square)
This code snippet employs the chi2_contingency function from the scipy.stats module to perform a chi-square test of independence. The test is conducted on a contingency table created using the 'Month' and 'Level' columns from the final_df DataFrame. The value (963.73) represents the test statistic for the chi-square test. It measures the discrepancy between the observed and expected frequencies in the contingency table and the p-value (1.015e-189) indicates strong evidence against the null hypothesis of independence. There is a significant association between the 'Month' and 'Level' variables.

The results suggest that the distribution of crime levels across months is not independent. There is a statistically significant relationship, and further investigation may be warranted to understand the nature of this association.
```
cont_table = pd.crosstab(index=final_df['Month'], columns=final_df['Level'])

chi2, p, dof, expected = chi2_contingency(cont_table)

print('Chi-square statistic:', chi2)
print('P-value:', p)
```
![5](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/2de83dbf-e65a-4e36-a158-3a06b6a279e2)
```
cont_table.plot(kind='bar', stacked=True)

plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Distribution of Levels Across Months')
plt.legend(title='Level')
```
![6](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/2b7c51a9-3217-411e-97c8-4ab1984e635a)

## Method: Anomaly Detection
The application of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm for anomaly detection on a dataset containing latitude and longitude information. Anomalies are identified based on the clustering results.

This approach helps identify anomalies in spatial data based on their deviation from the surrounding clusters. In this specific example, there are 33 anomalies, constituting a 0.02% anomaly rate in the dataset.
```
data = pd.read_csv('C:\\Users\\Project\\2022.csv')

data.dropna(inplace=True)

X = data[["Latitude", "Longitude"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.1, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

anomaly_mask = (clusters == -1)

num_anomalies = sum(anomaly_mask)
anomaly_rate = num_anomalies / len(data)
print(f"Number of anomalies: {num_anomalies}")
print(f"Anomaly rate: {anomaly_rate:.2%}")
```
![7](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/303282d4-867d-400c-a874-960dccf87919)
```
anomalies = data[anomaly_mask]

print(anomalies)
```
![8](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/287240ac-a40e-403b-a7af-63cceb0f0e56)

## Printing the anomaly rows, showing only Latitude and Longitude columns
```
print(anomalies[["Latitude", "Longitude"]])
```
![9](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/481d7660-4f48-40fb-bb2e-5ca8e6b7b47e)

## Method : Using a Random Forest Classifier
### Grid search for parameter tuning
```
estimator_range = range(1, 101)

acc = []

for n in estimator_range:
    rf_model = RandomForestClassifier(n_estimators=n, random_state=0)
    rf_model.fit(X_train, y_train)
    score = rf_model.score(X_test, y_test)
    acc.append(score)

plt.plot(estimator_range, scores)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.show()
```
![10](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/b0e036d7-01b9-4d92-b7be-7afd0fdb149e)

### Random Forest
The Random Forest Classifier achieved an accuracy of approximately 69.58% on the test data, indicating the percentage of correctly predicted labels. R-Squared represents the proportion of variance in the target variable explained by the model. A higher R-squared value suggests better predictive performance.

MAE represents the average absolute difference between predicted and actual values. It is a measure of the model's average prediction error.

MSE represents the average squared difference between predicted and actual values. It quantifies the average magnitude of errors, with lower values indicating better model performance.
```
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 101)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:",(metrics.accuracy_score(y_test, y_pred)*100),"\n")
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(y_test,y_pred)
print("R-Squared test:", r2)
print("MAE test:", mae)
print("MSE test:", mse)
```
![11](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/0d4426af-c57b-4782-9493-e3c0888dfcc1)

### Random Forest with kfold
The mean accuracy across the 5 folds is approximately 74.07%, providing a more robust estimate of the classifier's performance compared to a single train-test split.
```
n_splits = 5

kf = KFold(n_splits=n_splits)

acc_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = RandomForestClassifier()
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc_scores.append(metrics.accuracy_score(y_test, y_pred))
mean_accuracy = np.mean(accuracy_scores)

print("Mean accuracy:", mean_accuracy)
```
## Method: Using a clustering algorithm
### Grid search to find optimal K value
```
score_list_test = []
score_list_train = []

for each in range(1,50):
    knn1 = KNeighborsClassifier(n_neighbors= each)
    knn1.fit(X_train, y_train)
    score_list_test.append(knn1.score(X_test,y_test))
    score_list_train.append(knn1.score(X_train,y_train))

# Plot
plt.plot(range(1, 50), score_list_test)
plt.plot(range(1, 50), score_list_train)
plt.xlabel("k values: orange - train, blue - test")
plt.ylabel("accuracy")
```
![13](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/2c5a6dbb-2c05-465a-9ecf-5dbdaf790531)

### KNN
The accuracy is approximately 59.71%, and additional regression metrics provide insights into the model's performance on the test data.
```
knn1 = KNeighborsClassifier(n_neighbors = 17)
knn1.fit(X_train, y_train)
y_pred = knn1.predict(X_test)

print("Accuracy:",(metrics.accuracy_score(y_test, y_pred)*100),"\n")
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(y_test,y_pred)
print("R-Squared test:", r2)
print("MAE test:", mae)
print("MSE test:", mse)
```
![14](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/d15ec218-2c64-459e-8767-73ef6500f0c3)

### KNN with Kfold
The averaged accuracy across the 10 folds is approximately 67.92%. This provides a more robust estimate of the KNN classifier's performance on the training data compared to a single train-test split.
```
model_1 = KNeighborsClassifier(n_neighbors=17)

kfold2 = KFold(n_splits=10, random_state=99, shuffle=True) 

cv_result1 = cross_val_score(model_1,
                            X_train,
                            y_train, 
                            cv = kfold2,
                            scoring = "accuracy")

print('Averaged Accuracy training k-fold - Model KNN:',cv_result1.mean())
y_fit1 = cross_val_predict(model_1,X_train,y_train, cv=10) 
```
![15](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/18509f89-1bc0-44ad-b7df-c9bb627db120)

## Method: Using a Decision Tree Regressor
```
train_sizes, train_scores, test_scores = learning_curve(
    estimator=DecisionTreeRegressor(max_depth=5),
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation error')
plt.title('Learning Curve')
plt.xlabel('Training set size')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()
```
![16](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/af861ef7-722f-4ca6-8b07-f952edbc9567)
The best-performing Decision Tree Regressor has a maximum depth of 8, achieving an accuracy of approximately 59.71%. The reported metrics provide insights into the model's performance on the test data.
```
final_r2 = 0
final_mae = 0
final_mse = 0
final_acc = 0 

for depth in range(1,15):
    regr_2 = DecisionTreeRegressor(max_depth=depth)

    regr_2.fit(X_train, y_train)

    y_2 = regr_2.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_2)
    mae = metrics.mean_absolute_error(y_test, y_2)
    rmse = np.sqrt(mse) 
    r2 = metrics.r2_score(y_test,y_2)
    acc = metrics.accuracy_score(y_test, y_pred)*100
    if acc > final_acc:
        final_r2 = r2
        final_mae = mae
        final_mse = mse
        final_acc = acc
print("R-Squared test:", final_r2)
print("MAE test:", final_mae)
print("MSE test:", final_mse)
print("Accuracy:",(metrics.accuracy_score(y_test, y_pred)*100),"\n")
```
![17](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/3bfc38c7-f14e-4997-9d2b-46f5f30c2c4f)

## Advance Method: Bagging Classifier
```
bagging_model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
bagging_model.fit(X_train, y_train)

importances = bagging_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in bagging_model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
```
![18](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/25330acb-e179-43fb-9d0a-f42a02da2577)

The AdaBoost Classifier achieved an accuracy of approximately 91.51% on the training set, and various regression metrics provide insights into its performance on the test data.
```
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=10), n_estimators=200, 
    algorithm="SAMME.R", learning_rate=0.5) 

ada_clf.fit(X_train, y_train)

y_pred_rf = ada_clf.predict(X_test)

print("Accuracy",ada_clf.score(X_train, y_train))

mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test,y_pred)
print("R-Squared:", r2)
print("MAE:", mae)
print("MSE:", mse)
```
![20](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/95cab646-4b2b-4ea3-b14d-2df7d1bdf0d9)

## Advance Method: Boosting ensemble technique
The Bagging Classifier achieved an accuracy of approximately 62.67% on the test set, and various regression metrics provide insights into its performance.
```
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features=4), n_estimators=500, 
    max_samples=1000, bootstrap=True, n_jobs=-1) 

bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)

print("Accuracy",metrics.accuracy_score(y_test, y_pred))
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test,y_pred)
print("R-Squared:", r2)
print("MAE:", mae)
print("MSE:", mse)
```
![21](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/56fa9a9a-9ab9-487a-ade6-d380297caf8d)

The mean accuracy across the 5 folds is approximately 74.07%. Additionally, various regression metrics provide insights into the model's performance on the last test set (fold).
```
n_splits = 5
kf = KFold(n_splits=n_splits)
accuracy_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 101)
    clf.fit(X_train, y_train)    
    y_pred = clf.predict(X_test)    
    accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
mean_accuracy = np.mean(accuracy_scores)
print("Mean accuracy:", mean_accuracy)


mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(y_test,y_pred)
print("R-Squared:", r2)
print("MAE:", mae)
print("MSE:", mse)
```
![19](https://github.com/faizaligola/Unveiling-Patterns-using-Ensemble-Learning-for-Crime-Analysis-and-Prediction-in-Chicago-/assets/80847944/00e83a0c-0ae5-4d8b-80d7-a173fc6b8d40)









