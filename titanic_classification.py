#importing libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from warnings import filterwarnings
filterwarnings(action='ignore')

#loading datasets
pd.set_option('display.max_columns',10,'display.width',1000)
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

#displaying shape
train.shape
test.shape

#checking for null values
train.isnull().sum()
test.isnull().sum()

#description of dataset
train.describe(include="all")
train.groupby('Survived').mean()
train.corr()
male_ind=len(train[train['Sex']=='male'])
print("No of males in titanic:",male_ind)
female_ind=len(train[train['Sex']=='female'])
print("No of females in titanic:",female_ind)

#plotting
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
gender=['Male','Female']
index=[577,314]
ax.bar(gender,index)
plt.xlabel("Gender")
plt.ylabel("No of people onboarding ship")
plt.show()

alive=len(train[train['Survived']==1])
dead=len(train[train['Survived']==0])
train.groupby('Sex')[['Survived']].mean()

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
status=['Survived','Dead']
ind=[alive,dead]
ax.bar(status,ind)
plt.xlabel("Status")
plt.show()

plt.figure(1)
train.loc[train['Survived']==1, 'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people according to ticket class in which people survived')

plt.figure(2)
train.loc[train['Survived']==0, 'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people according to ticket class in which people couldn\'t survive')

plt.figure(1)
age=train.loc[train.Survived==1,'Age']
plt.title('The histogram of the age groups of the people that had survived')
plt.hist(age, np.arrange(0,100,10))
plt.xticks(np.arrange(0,100,10))

plt.figure(2)
age=train.loc[train.Survived==0,'Age']
plt.title('The histogram of the age groups of the people that couldn\'t survive')
plt.hist(age, np.arrange(0,100,10))
plt.xticks(np.arrange(0,100,10))

train[["SibSp","Survived"]].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[["Pclass","Survived"]].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[["Age","Survived"]].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=True)
train[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
l=['C=Cherbourg','Q=Queenstown','S=Southampton']
s=[0.553571,0.389610,0.336957]
ax.pie(s,labels=1,autopct='%1.2f%%')
plt.show()

test.describe(include="all")

#dropping useless columns
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)
train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)

column_train=['Age'='Pclass','SibSp','Parch','Fare','Sex','Embarked']
x=train[column_train]
y=train['Survived']
x['Age'].isnull().sum()
x['Pclass'].isnull().sum()
x['SibSp'].isnull().sum()
x['Parch'].isnull().sum()
x['Fare'].isnull().sum()
x['Sex'].isnull().sum()
x['Embarked'].isnull().sum()

#filling missing values
x['Age']=x['Age'].fillna(x['Age'].median())
x['Age'].isnull.sum()
x['Embarked']=train['Embarked'].fillna(method='pad')
x['Embarked'].isnull.sum()
d={'male':0, 'female':1}
x['Sex']=x['Sex'].apply(lambda x:d[x])
x['Sex'].head()
e={'C':0,'Q':1,'S':2}
x['Embarked']=x['Embarked'].apply(lambda x:e[x])
x['Embarked'].head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=7)

#using logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_mat=confusion_matrix(y_test,y_pred)
print(confusion_mat)

#using support vector
from sklearn.svm import SVC
model1=SVC()
model1.fit(x_train,y_train)

pred_y=model1.predict(x_test)

from sklearn.metrics import accuracy_score
print("acc=",accuracy_score(y_test,pred_y))

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat=confusion_matrix(y_test,pred_y)
print(confusion_mat)
print(classification_report(y_test,pred_y))

#using knn neighbors
from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier(n_neighbors=S)
model2.fit(x_train,y_train)
y_pred2=model2.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy score:",accuracy_score(y_test,y_pred2))

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat=confusion_matrix(y_test,y_pred2)
print(confusion_mat)
print(classification_report(y_test,y_pred2))