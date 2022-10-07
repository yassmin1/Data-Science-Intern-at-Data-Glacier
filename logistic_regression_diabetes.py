# %% [markdown]
# ## Pima Indians Diabetes Case study
# Diabetes is one of the most frequent diseases worldwide and the number of diabetic patients is growing over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes. 
# 
# Individuals with diabetes face a risk of developing some secondary health issues such as heart diseases and nerve damage. Thus, early detection and treatment of diabetes can prevent complications and assist in reducing the risk of severe health problems. 
# Even though it's incurable, it can be managed by treatment and medication.
# 
# Researchers at the Bio-Solutions lab want to get better understanding of this disease among women and are planning to use machine learning models that will help them to identify patients who are at risk of diabetes. 
# 
# We will use logistic regression to model the "Pima Indians Diabetes" data set. In particular, all patients here are females at least 21 years old of Pima Indian heritage. This model will predict which people are likely to develop diabetes.

# %% [markdown]
# ## Data Description:
# 
# * Pregnancies: Number of times pregnant
# * Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
# * BloodPressure: Diastolic blood pressure (mm Hg)
# * SkinThickness: Triceps skinfold thickness (mm)
# * Insulin: 2-Hour serum insulin (mu U/ml)
# * BMI: Body mass index (weight in kg/(height in m)^2)
# * Pedigree: Diabetes pedigree function - A function that scores likelihood of diabetes based on family history.
# * Age: Age in years
# * Class: Class variable (0: the person is not diabetic or 1: the person is diabetic)

# %% [markdown]
# ### Import necessary libraries

# %%
# To filter the warnings
import warnings

warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to split data
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
# To build linear model for statistical analysis and prediction


# To get diferent metric scores
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, roc_auc_score,recall_score
import pickle



# %% [markdown]
# ### Read the dataset

# %%
data = pd.read_csv("pima-indians-diabetes.csv")

# %%
# defining columns where we need to replace 0 with NaN
cols = ["Glucose", "BMI", "Pedigree"]

# %%
# replacing 0 with NaN
data[cols] = data[cols].replace(0, np.nan)

# %%
# Let's impute missing values using mean value
data[cols] = data[cols].fillna(data[cols].mean())

# %% [markdown]
# ### Splitting data into train and test

# %%
X = data.drop(["Class"], axis=1)
Y = data["Class"]

# %%
#X.sample(4)

# %%
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1,stratify=Y)

# %% [markdown]
# **The Stratify argument maintains the original distribution of classes in the target variable while splitting the data into train and test sets.**

# %% [markdown]
# ### Fitting Logistic Regression model

# %%
lg=LogisticRegression(penalty='elasticnet',l1_ratio=0.9,max_iter=500,solver='saga',class_weight='balanced')
lg.fit(X_train,y_train)

# %%
pickle.dump(lg, open('model.pkl','wb'))

# %%
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1	,100.0,	47.9,0.137]]))


