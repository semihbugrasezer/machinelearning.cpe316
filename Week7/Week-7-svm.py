#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np


# In[2]:


data=pd.read_csv(r"C:\Users\hasan\Desktop\SPRING SEMESTER\IML\ML-LAB\WEEK-7\data.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.drop(["Unnamed: 32","id"],axis=1,inplace=True)


# In[6]:


M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]


# In[7]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius mean")
plt.ylabel("area mean")
plt.show()


# In[8]:


data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]


# In[9]:


y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


# In[10]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[11]:


x.head()


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[13]:


from sklearn.svm import SVC
svc=SVC(random_state=42)
svc.fit(x_train,y_train)


# In[14]:


svc.score(x_test,y_test)


# In[15]:


train_accuracy=[]
test_accuracy=[]
for i in range(1,100):
    svm=SVC(C=i)
    svm.fit(x_train,y_train)
    train_accuracy.append(svm.score(x_train,y_train))
    test_accuracy.append(svm.score(x_test,y_test))
plt.plot(range(1,100),train_accuracy,label="training accuracy")
plt.plot(range(1,100),test_accuracy,label="testing accuracy")
plt.xlabel("c values")
plt.ylabel("accuracy")
plt.show()


# In[16]:


print("best accuracy is {} when c ={}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# In[ ]:




