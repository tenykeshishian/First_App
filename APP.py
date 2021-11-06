# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 09:37:14 2021

@author: Lenovo
"""

import numpy as np
import pandas as pd
from numpy.random import seed
import streamlit as st
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import seaborn as sns



# Heading
st.title('Probability Of Having Diabetes') 
st.subheader('press this > button on upper left corner to enter the values')
#st.text('author : Teny')          
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')

# Importing data
data = pd.read_csv('C:/Users/Lenovo/.spyder-py3/diabetes.csv')
print('\n\n',data.head(5))

'''
Diabetes pedigree function : 
    a function which scores likelihood of 
    diabetes based on family history
'''
st.write(data.describe())

# Data Preprocessing
column_names = data.columns.tolist()
print('\n\n',column_names)


# Checking for Nan Values
print('\n\n',data.info())
print('\n\n',data.isnull().sum())
correlation_matrix = data.corr(method='pearson')
print('\n\n',correlation_matrix)
# Each attribute correlation with final diagnosis
print('\n\n',correlation_matrix['Outcome'].sort_values())

# histogram
data.hist(bins=50,figsize=(25,20))
plt.show()

# Removing duplicated values
data.drop_duplicates(keep='first',inplace=True)
x = data.drop(['Outcome'],axis=1)
y = data.Outcome

# Spliting data into train and test datasets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
x_train2 = x_train.copy()
print('\n\n',x_train2.shape)

# Handling missing values
# To make sure that we don't mistake missing values
# as being 0, we will replace them with NaN
names = ['Glucose','BloodPressure','SkinThickness',
         'Insulin','BMI','DiabetesPedigreeFunction','Age']
for name in names:
    x_train2[name].replace(0,np.nan,inplace=True)
print('\n\n',x_train2.isnull().sum())

# Replacing NaN values with median of the corresponding attribute
glucose_med = x_train2['Glucose'].median()
blood_pressure_med = x_train2['BloodPressure'].median()
skin_thickness_med = x_train2['SkinThickness'].median()
insulin_med = x_train2['Insulin'].median()
bmi_med = x_train2['BMI'].median()
age_med = x_train2['Age'].median()
dpf_med = x_train2['DiabetesPedigreeFunction'].median()

x_train2['Glucose'].fillna(glucose_med,inplace=True)
x_train2['BloodPressure'].fillna(blood_pressure_med,inplace=True)
x_train2['SkinThickness'].fillna(skin_thickness_med,inplace=True)
x_train2['Insulin'].fillna(insulin_med,inplace=True)
x_train2['BMI'].fillna(bmi_med,inplace=True)
x_train2['Age'].fillna(age_med,inplace=True)
x_train2['DiabetesPedigreeFunction'].fillna(dpf_med,inplace=True)

print('\n\n',x_train2.isnull().sum())

# creating a sidebar slider for user data input
def user_data():
    pregnancies = st.sidebar.slider('Pregnancies',0,17,3,key='a3')
    glucose = st.sidebar.slider('Glucose',0,200,120,key='b')
    bp = st.sidebar.slider('Blood Pressure',0,122,70,key='c')
    skinthickness = st.sidebar.slider('Skin Thickness',0,100,20,key='d')
    insulin = st.sidebar.slider('Insulin',0,846,79,key='e')
    bmi = st.sidebar.slider('BMI',0,67,20,key='f')
    dpf = st.sidebar.slider('Diabetes Pedigree Function',0.0,2.4,0.47,key='g')
    age = st.sidebar.slider('Age',21,88,33,key='h')
    
    user_data_dict = {
        'pregnancies':pregnancies,
        'glucose':glucose,
        'bp':bp,
        'skinthickness':skinthickness,
        'bmi':bmi,
        'insulin':insulin,
        'dpf':dpf,
        'age':age}
    data2 = pd.DataFrame(user_data_dict,index=[0])
    return data2

# User data
UserData = user_data()
st.subheader('Patient Data')
st.write(UserData)


# Rescaling 
# initialize min_max scaler
mm_scaler = MinMaxScaler()
col = x_train2.columns.tolist()
print('\n',col)
x_train2[col]=mm_scaler.fit_transform(x_train2[col])
x_train2.sort_index(inplace=True)
print('\n',x_train2.head(5))


x_train3 = x_train2.copy()
S_scaler = StandardScaler()
col = x_train3.columns.tolist()
print('\n',col)
x_train3[col]=S_scaler.fit_transform(x_train3[col])
x_train3.sort_index(inplace=True)

# Support vector machine
#model = SVC(kernel='sigmoid',C=0.1)
model = BaggingClassifier(base_estimator=SVC(),
                       n_estimators=10, random_state=0)
model.fit(x_train3,y_train)
user_result = model.predict(UserData)

output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'


# Random Forest Classifier
Model2 = RandomForestClassifier(n_estimators=90,criterion='entropy')
Model2.fit(x_train2,y_train)
user_result2 = Model2.predict(UserData)
output2=''
if user_result2[0]==0:
  output2 = 'You are not Diabetic'
else:
  output2 = 'You are Diabetic'


# Logistic Regression

model3 = LogisticRegression(solver='liblinear',penalty='l2',max_iter=800)
model3.fit(x_train,y_train)
user_result3 = model3.predict(UserData)
output3=''
if user_result3[0]==0:
  output3 = 'You are not Diabetic'
else:
  output3 = 'You are Diabetic'

# Deep learning      
seed(1)
model4 = Sequential()
model4.add(Dense(300,input_dim=8,activation='relu'))
model4.add(Dense(180,activation='relu'))
model4.add(Dense(1,activation='sigmoid'))
model4.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model4.fit(x_train2,y_train,batch_size=70,epochs=220,shuffle=False)

_,accuracy = model4.evaluate(x_train2,y_train)
# Model accuracy
print('Model accuracy : %.2f'%(accuracy*100))
# Model Prediction
user_result4=model4.predict(UserData)
output4=''
if user_result4[0]==0:
    output4 = 'You are not Diabetic'
else:
    output4 = 'Yo are Diabetec'



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = data, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = UserData['age'], y = UserData['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = data, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = UserData['age'], y = UserData['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = data, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = UserData['age'], y = UserData['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = data, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = UserData['age'], y = UserData['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = data, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = UserData['age'], y = UserData['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = data, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = UserData['age'], y = UserData['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = data, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = UserData['age'], y = UserData['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)


st.title('Result:')
#st.subheader('SVM Accuracy:')
#st.write(str(accuracy_score(y_test, model.predict(x_test))*100)+'%')
#st.subheader(output)
#st.write('-------------------------------------')

#st.subheader('Random Forest Accuracy:')
#st.write(str(accuracy_score(y_test, Model2.predict(x_test))*100)+'%')
#st.subheader(output2)
#st.write('-------------------------------------')

st.subheader('Logistic Regression Accuracy:')
st.write(str(accuracy_score(y_test, model3.predict(x_test))*100)+'%')
st.subheader(output3)
st.write('-------------------------------------')

st.subheader('Deep learning Accuracy:')
st.write('Model accuracy : %.2f'%(accuracy*100))
st.subheader(output4)