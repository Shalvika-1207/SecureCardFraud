#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# # Loading Data Sets

# In[2]:


#Train data
X_train = pd.read_csv("training_data.csv")

# Training Target Variables  
y_train = pd.read_csv("train_data_classlabels.csv")

# Test Data 
X_test = pd.read_csv("testing_data.csv") 


# # Data Analysis and Visualization

# In[3]:


X_train.head()


# In[4]:


X_train.describe()


# In[5]:


X_test.head()


# In[6]:


X_test.describe()


# In[7]:


X_train.isnull().sum()


# In[8]:


X_test.isnull().sum()


# In[9]:


# Impute missing values with the mean for numerical columns
X_test.fillna(X_test.mean(), inplace=True)


# In[10]:


X_test.isnull().sum()


# In[11]:


print('No Frauds', round(y_train['Class'].value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
print('Frauds', round(y_train['Class'].value_counts()[1]/len(y_train) * 100,2), '% of the dataset')


# In[12]:


y_train.value_counts()


# In[13]:


colors = ["#0101DF", "#DF0101"]
sns.countplot(x='Class', data=y_train, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()


# In[14]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = X_train['Amount'].values
time_val = X_train['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()


# # Scaling Time and Amount

# In[15]:


from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

X_train['scaled_amount'] = rob_scaler.fit_transform(X_train['Amount'].values.reshape(-1,1))
X_train['scaled_time'] = rob_scaler.fit_transform(X_train['Time'].values.reshape(-1,1))

X_train.drop(['Time','Amount'], axis=1, inplace=True)


# In[16]:


scaled_amount = X_train['scaled_amount']
scaled_time = X_train['scaled_time']

X_train.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
X_train.insert(0, 'scaled_amount', scaled_amount)
X_train.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

X_train.head()


# In[17]:


from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

X_test['scaled_amount'] = rob_scaler.fit_transform(X_test['Amount'].values.reshape(-1,1))
X_test['scaled_time'] = rob_scaler.fit_transform(X_test['Time'].values.reshape(-1,1))

X_test.drop(['Time','Amount'], axis=1, inplace=True)


# In[18]:


scaled_amount = X_test['scaled_amount']
scaled_time = X_test['scaled_time']

X_test.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
X_test.insert(0, 'scaled_amount', scaled_amount)
X_test.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

X_test.head()


# # Correlation matrix

# In[19]:


plt.figure(figsize=(6, 4))  # Set the size of the figure

# Entire DataFrame
corr = X_train.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20})
plt.title("Correlation Matrix", fontsize=14)

# Show the plot
plt.show()



# # Implementing various classifiers

# # 1.Gaussian Naive Bayes 

# In[20]:


from sklearn.naive_bayes import GaussianNB  # Naive Bayes Classifier
#from sklearn.linear_model import LogisticRegression # Logistic Regression Classifier
from sklearn.neighbors import KNeighborsClassifier  # K-nearest neighbors Classifier
from sklearn.model_selection import GridSearchCV  # Gridsearch for Parameter Tuning
from sklearn.pipeline import Pipeline  # Assembling steps to cross-validate together
from sklearn.metrics import classification_report  # For classification report
#from sklearn.svm import SVC  #For Support Vector Machine
from sklearn.model_selection import train_test_split # Train test spliting method
from sklearn.preprocessing import StandardScaler # For Standard Scaler 
from sklearn import datasets #importing datasets
import numpy as np

y_train = np.ravel(y_train)  # To convert array to size (n,), return a contiguous flattened array

"""### Gaussian Naive Bayes Classifier"""

clf1 = GaussianNB()
clf1_parameters = {
    'clf__priors': [[0.5,
                     0.5]],  # Prior probabilities (50 % data of each class)
    'clf__var_smoothing': np.logspace(
        0, -11, num=100
    )  # Portion of the largest variance of all features that is added to variances for calculation stability.
}
pipeline1 = Pipeline([
    ('clf', clf1),
])

# PARAMETER TUNING
grid1 = GridSearchCV(pipeline1, clf1_parameters, scoring='f1_macro', cv=10)
grid1.fit(X_train, y_train)
clf1 = grid1.best_estimator_
print("\nThe best parameters found for Gaussian NB are: \n",
      grid1.best_params_)

print("\nCLASSIFICATION REPORT FOR GAUSSIAN NAIVE BAYES BEST FIT\n\n")
print(classification_report(y_train, clf1.predict(X_train), labels=[0, 1]))


# In[21]:


from sklearn.metrics import classification_report, confusion_matrix


# In[22]:


print("\nCONFUSION MATRIX FOR GAUSSIAN NAIVE BAYES \n\n")
print(confusion_matrix(y_train, clf1.predict(X_train)))


# # Gaussian Naive Bayes with SMOTE

# In[23]:


from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline

# Define the SMOTE technique
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Define the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()

# Create a pipeline that includes SMOTE and the Gaussian Naive Bayes classifier
pipeline1 = Pipeline([
    ('smote', smote),            # Apply SMOTE to balance the classes
    ('clf', gnb_classifier),     # Gaussian Naive Bayes classifier
])

# PARAMETER TUNING
gnb_parameters = {
    'clf__priors': [[0.5, 0.5]],  # Prior probabilities (50 % data of each class)
    'clf__var_smoothing': np.logspace(0, -11, num=100)
}

# Use GridSearchCV with the pipeline including SMOTE
grid2 = GridSearchCV(pipeline1, gnb_parameters, scoring='f1_macro', cv=10)
grid2.fit(X_train, y_train)
clf2 = grid2.best_estimator_

print("\nThe best parameters found for Gaussian NB are: \n", grid2.best_params_)
print("\nCLASSIFICATION REPORT FOR GAUSSIAN NAIVE BAYES WITH SMOTE\n\n")
print(classification_report(y_train, clf2.predict(X_train), labels=[0, 1]))


# In[24]:


print("\nCONFUSION MATRIX FOR GAUSSIAN NAIVE BAYES WITH SMOTE\n\n")
print(confusion_matrix(y_train, clf2.predict(X_train)))


# # 2. K - Nearest Neighbour

# In[25]:


"""### K-Nearest Neighbor Classfier"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Define a search space
knn_parameters = {
    'clf__n_neighbors': [3, 5, 7],  #  number of neighbors
    'clf__metric': ['euclidean', 'manhattan'],
}

# Create a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()

# Create a pipeline that includes the K-Nearest Neighbors classifier
pipeline3 = Pipeline([
    ('clf', knn_classifier),
])

# Use RandomizedSearchCV 
randomized_search = RandomizedSearchCV(pipeline3, knn_parameters, scoring='f1_macro', cv=10, n_iter=10)
randomized_search.fit(X_train, y_train)
clf3 = randomized_search.best_estimator_

print("\nThe best parameters found for KNN are: \n", randomized_search.best_params_)


# In[26]:


print("\nCLASSIFICATION REPORT FOR KNN BEST FIT\n\n")
print(classification_report(y_train, clf3.predict(X_train), labels=[0, 1]))


# In[27]:


print("\nCONFUSION MATRIX FOR KNN\n\n")
print(confusion_matrix(y_train, clf3.predict(X_train)))


# # K - Nearest Neighbour with SMOTE

# In[28]:


from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline

# Define the SMOTE technique
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Define a search space for KNN
knn_parameters = {
    'clf__n_neighbors': [3, 5, 7],  # Reduced number of neighbors
    'clf__metric': ['euclidean', 'manhattan'],
}

# Create a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()

# Create a pipeline that includes SMOTE, KNN, and any other preprocessing steps
pipeline3 = Pipeline([
    ('smote', smote),           # Apply SMOTE to balance the classes
    ('clf', knn_classifier),    # K-Nearest Neighbors classifier
])

# Use RandomizedSearchCV 
randomized_search = RandomizedSearchCV(pipeline3, knn_parameters, scoring='f1_macro', cv=10, n_iter=10)
randomized_search.fit(X_train, y_train)
clf4 = randomized_search.best_estimator_

print("\nThe best parameters found for KNN are: \n", randomized_search.best_params_)


# In[29]:


print("\nCLASSIFICATION REPORT FOR KNN BEST FIT WITH SMOTE\n\n")
print(classification_report(y_train, clf4.predict(X_train), labels=[0, 1]))


# In[30]:


print("\nCONFUSION MATRIX FOR KNN\n\n")
print(confusion_matrix(y_train, clf4.predict(X_train)))


# # 3. Decision Tree 

# In[31]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Define a search space for Decision Tree hyperparameters
dt_parameters = {
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
}

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=123)

# Create a pipeline that includes the Decision Tree classifier
pipeline_dt = Pipeline([
    ('clf', dt_classifier),
])

# Use RandomizedSearchCV 
randomized_search_dt = RandomizedSearchCV(
    pipeline_dt,
    dt_parameters,
    scoring='f1_macro',
    cv=10,
    n_iter=5,  
    n_jobs=-1,  
)

randomized_search_dt.fit(X_train, y_train)
best_dt_model = randomized_search_dt.best_estimator_

print("\nThe best parameters found for Decision Tree are: \n", randomized_search_dt.best_params_)


# In[32]:


print("\nCLASSIFICATION REPORT FOR DECISION TREE BEST FIT \n\n")
print(classification_report(y_train, best_dt_model.predict(X_train), labels=[0, 1]))


# In[33]:


print("\nCONFUSION MATRIX FOR DECISION TREE\n\n")
print(confusion_matrix(y_train, best_dt_model.predict(X_train)))


# In[ ]:


"""### Support Vector Classifier"""
from sklearn.svm import SVC

# Define a search space for Support Vector Classifier
svc_params = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'clf__kernel': ['linear', 'rbf'], 'clf__gamma': ['scale', 'auto']}

# Create a Support Vector Classifier
svc_classifier = SVC()

# Create a pipeline that includes the Support Vector Classifier
pipeline_svc = Pipeline([
    ('clf', svc_classifier),
])

# Use RandomizedSearchCV for Support Vector Classifier
randomized_search_svc = RandomizedSearchCV(pipeline_svc, svc_params, scoring='f1_macro', cv=10, n_iter=10)
randomized_search_svc.fit(X_train, y_train)
clf_svc = randomized_search_svc.best_estimator_

print("\nThe best parameters found for Support Vector Classifier are: \n", randomized_search_svc.best_params_)
print("\nCLASSIFICATION REPORT FOR Support Vector Classifier\n\n")
print(classification_report(y_train, clf_svc.predict(X_train), labels=[0, 1]))


# In[ ]:


"""### Logistic Regression Classifier"""
from sklearn.linear_model import LogisticRegression

# Define a search space for Logistic Regression
log_reg_params = {"clf__penalty": ['l1', 'l2'], 'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Create a Logistic Regression classifier
log_reg_classifier = LogisticRegression()

# Create a pipeline that includes the Logistic Regression classifier
pipeline_log_reg = Pipeline([
    ('clf', log_reg_classifier),
])

# Use RandomizedSearchCV for Logistic Regression
randomized_search_log_reg = RandomizedSearchCV(pipeline_log_reg, log_reg_params, scoring='f1_macro', cv=10, n_iter=10)
randomized_search_log_reg.fit(X_train, y_train)
clf_log_reg = randomized_search_log_reg.best_estimator_

print("\nThe best parameters found for Logistic Regression are: \n", randomized_search_log_reg.best_params_)
print("\nCLASSIFICATION REPORT FOR Logistic Regression\n\n")
print(classification_report(y_train, clf_log_reg.predict(X_train), labels=[0, 1]))



# # 4. Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Define a search space for Random Forest hyperparameters
rf_parameters = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=123)

# Use RandomizedSearchCV 
randomized_search_rf = RandomizedSearchCV(
    rf_classifier,
    rf_parameters,
    scoring='f1_macro',
    cv=10,
    n_iter=5,  
    n_jobs=-1,  
)

randomized_search_rf.fit(X_train, y_train)
best_rf_model = randomized_search_rf.best_estimator_

print("\nThe best parameters found for Random Forest are: \n", randomized_search_rf.best_params_)



# In[35]:


print("\nCLASSIFICATION REPORT FOR RANDOM FOREST BEST FIT \n\n")
print(classification_report(y_train, best_rf_model.predict(X_train), labels=[0, 1]))


# In[36]:


print("\nCONFUSION MATRIX FOR RANDOM FOREST\n\n")
print(confusion_matrix(y_train, best_rf_model.predict(X_train)))


# # Best Model after comparing results is RANDOM FOREST

# # Predicting labels for test data after comparing results

# In[37]:


y_test_predrf = best_rf_model.predict(X_test)

# Convert the predicted labels to integer
y_test_predrf = y_test_predrf.astype(int)

# Create a DataFrame to display the predicted labels
predicted_dfrf = pd.DataFrame({'Predicted Labels': y_test_predrf})

# Set option to display all rows
pd.set_option('display.max_rows', None)

# Print the DataFrame
print(predicted_dfrf)

# Reset the option to the default value
pd.reset_option('display.max_rows')


# # Saving the Class Labels predicted by Random Forest

# In[42]:


# we can see from the classification reports the best model IS RANDOM FOREST among GAUSIAN NAIVE BAYES, KNN ,DECISION TREE AND RANDOM FOREST 
#Hence, saving the predicted labels by RANDOM FOREST CLASSIFIER.
file_path = r"C:\Users\win11\Downloads\Shalvika_Srivastav_Test_Target_Labels.txt"

# Open the file in write mode and write the labels
with open(file_path, "w") as file:
    for label in y_test_predrf:
        file.write(str(label) + "\n")

print("Predicted labels have been saved to", file_path)


# In[ ]:




