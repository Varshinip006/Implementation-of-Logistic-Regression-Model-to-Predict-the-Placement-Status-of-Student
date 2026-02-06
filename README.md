# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset using the Pandas library.

2.Create a copy of the dataset and remove unnecessary columns like serial number and salary.

3.Check the dataset for missing values and duplicate records.

4.Convert all categorical attributes into numerical form using Label Encoding.

5.Separate the dataset into independent features (X) and target variable (status).

6.Split the dataset into training and testing sets using an 80:20 ratio.

7.Initialize the Logistic Regression model with a suitable solver.

8.Train the model using the training dataset.

9.Predict the placement status using the test dataset.

10.Evaluate the model performance using accuracy score and classification report.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Priya Varshini P
RegisterNumber: 212224240119

import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

data1

x = data1.iloc[:, :-1]
y = data1["status"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])


```

## Output:
<img width="1403" height="380" alt="image" src="https://github.com/user-attachments/assets/b75c59a8-00b9-4838-8481-ba8806ca21b6" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
