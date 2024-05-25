#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[4]:


data = pd.read_csv(r"C:\Users\hasan\Desktop\SPRING SEMESTER\IML\ML-LAB\WEEK-8\data.csv")


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.drop(["Unnamed: 32","id"],axis=1, inplace=True)


# In[8]:


M = data[data.diagnosis=="M"]
B = data[data.diagnosis=="B"]


# In[9]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant") 
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[10]:


data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis] 


# In[11]:


y = data.diagnosis.values 


# In[12]:


x_data= data.iloc[:,1:3].values 


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size=0.3,random_state=1)


# In[14]:


from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
tree_classification=DecisionTreeClassifier(random_state=1,criterion='entropy')
tree_classification.fit(x_train,y_train)


# In[16]:


y_head=tree_classification.predict(x_test)


# In[17]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_head)
accuracy


# In[18]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_head)


# In[19]:


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,fmt='.0f',linewidths=0.5,linecolor="red",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_head")
plt.show()


# In[ ]:




