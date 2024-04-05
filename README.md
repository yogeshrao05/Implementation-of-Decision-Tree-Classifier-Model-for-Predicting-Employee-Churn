# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yogesh rao S D
RegisterNumber:  212222110055
*/
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![318881670-53611d16-6446-4710-9231-b1a5653eef1f](https://github.com/sivabalan28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497347/8b1d0738-a695-4c7d-a9fe-34947fb6fefe)
![318881803-a8131f83-0152-4df1-bfe0-dcd3e24c2a94](https://github.com/sivabalan28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497347/cc5bdc15-fa5b-42fd-a7e6-5949354e6809)
![318881995-d29a789e-8d05-465e-bcef-bea718bda013](https://github.com/sivabalan28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497347/3c8ce3ac-65d9-4cff-922c-01c81d3302fb)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
