# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.finally execute the program and display the output 

~~~
## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vignesh s
RegisterNumber: 212223230240

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset.head()
dataset.info()

dataset = dataset.drop('sl_no', axis=1);
dataset.info()

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes


dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x = dataset.iloc[:,:-1]
x

y=dataset.iloc[:,-1]
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score, confusion_matrix
cf = confusion_matrix(y_test, y_pred)
cf

accuracy=accuracy_score(y_test,y_pred)
accuracy

```

## Output:
## HEAD 
![image](https://github.com/user-attachments/assets/c5918b2c-53e0-4aee-a2cb-98b0fdb997c0)

## INFO 
![image](https://github.com/user-attachments/assets/ba978d8a-4ac9-474d-9165-7a3e802f92c1)

## changing into Category:

![image](https://github.com/user-attachments/assets/694b43fa-f0e7-40cb-9d44-4446536e6024)

## Changing into codes:
![image](https://github.com/user-attachments/assets/1944f757-59a8-4b72-8ce4-b13db0802bbc)

## Value of X:
![image](https://github.com/user-attachments/assets/bf0d7e95-0f46-4876-87d6-63c9d7d04742)
## Value of Y:
![image](https://github.com/user-attachments/assets/82fd491e-9912-40cf-8fa0-4a59f64adbf8)
## Y Prediction:
![image](https://github.com/user-attachments/assets/1cd3632b-b111-4959-b98d-c8315bf42a7e)

## Confusion Matrix:
![image](https://github.com/user-attachments/assets/7eb8fbe8-cba1-4803-92a2-78437867e98f)
## Accuracy:
![image](https://github.com/user-attachments/assets/acf3c404-2a0f-4ca9-ad60-1dba756f1840)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
