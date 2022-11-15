#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#We retreive the data from the database here


# In[2]:


main=pd.read_csv("C:/Users/Crisp/OneDrive/Documents/WeatherForecast.csv",engine="python",error_bad_lines=False)
print(main.shape)
data=main.copy()


# In[ ]:


#Here we drop the following datasets


# In[3]:


data =data.dropna()
data.drop(["Formatted Date","Daily Summary","Summary","Loud Cover"],axis=1,inplace=True)
print(data.shape)


# In[4]:


data["Precip Type"][data["Precip Type"]=="rain"]=1
data["Precip Type"][data["Precip Type"]=="snow"]=0
data["Precip Type"].unique()


# In[5]:


import missingno as mnso


# In[ ]:


#Here we check for missing values


# In[6]:


mnso.bar(data)


# In[7]:


data.isnull().sum()


# In[ ]:


#Here we train the model 


# In[8]:


from sklearn.model_selection import train_test_split # class to split data 

data_copy=data.copy()
print(data.head())


Y =data["Precip Type"]
X=data.drop(["Precip Type"],axis=1)
print(X.head())


X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)


# In[ ]:


#Here we get the maximum and minimum value from the train data and the test data


# In[9]:


from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Temperature (C)'].values.reshape(-1,1))
X_train_temp = scale.transform(X_train['Temperature (C)'].values.reshape(-1,1))
X_test_temp = scale.transform(X_test['Temperature (C)'].values.reshape(-1,1))
print('Max:',X_train_temp.shape)
print('Max:',X_test_temp.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[10]:


scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Apparent Temperature (C)'].values.reshape(-1,1))
X_train_atemp = scale.transform(X_train['Apparent Temperature (C)'].values.reshape(-1,1))
X_test_atemp = scale.transform(X_test['Apparent Temperature (C)'].values.reshape(-1,1))
print('Max:',X_train_atemp.shape)
print('Max:',X_test_atemp.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[11]:


scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Humidity'].values.reshape(-1,1))
X_train_humid = scale.transform(X_train['Humidity'].values.reshape(-1,1))
X_test_humid = scale.transform(X_test['Humidity'].values.reshape(-1,1))
print('Max:',X_train_humid.shape)
print('Max:',X_test_humid.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[12]:


scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Wind Speed (km/h)'].values.reshape(-1,1))
X_train_wind = scale.transform(X_train['Wind Speed (km/h)'].values.reshape(-1,1))
X_test_wind = scale.transform(X_test['Wind Speed (km/h)'].values.reshape(-1,1))
print('Max:',X_train_wind.shape)
print('Max:',X_test_wind.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[13]:


scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Wind Bearing (degrees)'].values.reshape(-1,1))
X_train_bwind = scale.transform(X_train['Wind Bearing (degrees)'].values.reshape(-1,1))
X_test_bwind = scale.transform(X_test['Wind Bearing (degrees)'].values.reshape(-1,1))
print('Max:',X_train_bwind.shape)
print('Max:',X_test_bwind.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[14]:


scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Visibility (km)'].values.reshape(-1,1))
X_train_visi = scale.transform(X_train['Visibility (km)'].values.reshape(-1,1))
X_test_visi = scale.transform(X_test['Visibility (km)'].values.reshape(-1,1))
print('Max:',X_train_visi.shape)
print('Max:',X_test_visi.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[15]:


scale = MinMaxScaler()
 # reshape because of minmax take column and scale
scale.fit(X_train['Pressure (millibars)'].values.reshape(-1,1))
X_train_pres = scale.transform(X_train['Pressure (millibars)'].values.reshape(-1,1))
X_test_pres = scale.transform(X_test['Pressure (millibars)'].values.reshape(-1,1))
print('Max:',X_train_pres.shape)
print('Max:',X_test_pres.shape)
print('Min:',y_train.shape)
print('Min:',y_test.shape)


# In[16]:


from scipy.sparse import hstack
import scipy.sparse as sp
import numpy as np

train = hstack((sp.csr_matrix(X_train_temp),X_train_atemp,X_train_humid,X_train_wind,X_train_bwind,X_train_visi,X_train_pres)).tocsr()
test = hstack((sp.csr_matrix(X_test_temp),X_test_atemp,X_test_humid,X_test_wind,X_test_bwind,X_test_visi,X_test_pres)).tocsr()


train= train.astype(np.float)
test= test.astype(np.float)
train=train.toarray()
test=test.toarray()
y_train=y_train.astype(np.float)
y_test=y_test.astype(np.float)

print(train.shape)
print(test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


import keras as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# In[ ]:


#Here we specify the size of the model


# In[24]:


model=Sequential([
    Dense(16,input_shape=(7,),activation='relu'),
    Dense(32,activation='relu',use_bias=True),
    Dense(2,activation='softmax')
])


# In[25]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[26]:


#lets visaulaize the model
print(model)


# In[27]:


history=model.fit(x=train,y=y_train,epochs=50,shuffle='True',verbose=2)


# In[ ]:


#Here we test the accuracy of the model


# In[28]:


results = model.evaluate(X_test, y_test, batch_size=10)
print('test loss, test acc:', results)


# In[36]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix


# In[62]:


predictions = model.predict(X_test)


# In[42]:


from keras.layers import Dropout
from keras.layers import BatchNormalization

modeld = Sequential()
modeld.add(Dropout(0.2, input_shape=(7,)))
modeld.add(Dense(16, activation='relu'))
modeld.add(Dropout(0.2))
modeld.add(BatchNormalization())
modeld.add(Dense(32, activation='relu'))
modeld.add(Dense(2, activation='softmax'))


# In[43]:


modeld.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[44]:


history=modeld.fit(x=train,y=y_train,epochs=0,shuffle='True',verbose=2)


# In[ ]:


#Here we test the accuracy of the test module


# In[48]:


resultsd = modeld.evaluate(X_test, y_test, batch_size=10)
print('test loss, test acc:', resultsd)


# In[ ]:


model.save("waethermodel.hdf5")


# In[ ]:


from keras.models import load_model
lmodel = load_model("waethermodel.hdf5")


# In[47]:


Input_hidden_weights = model.layers[0].get_weights()[0]
Hidden_output_weights = model.layers[1].get_weights()[0]
print(Input_hidden_weights)
print(len(Input_hidden_weights))
print("\n############################################################")
print(Hidden_output_weights)
len(Hidden_output_weights)


# In[51]:


## Predicting new weather patterns
new_weather = scale.transform(X_train['Pressure (millibars)'].values.reshape(-1,1))
print(f' Pressure Prediction:/n{new_weather}')
print('/nNew Pressure Prediction:',model.predict(X_test))


# In[59]:


## Predicting new weather patterns
new_weather = scale.transform(X_train['Temperature (C)'].values.reshape(-1,1))
print(f'Temperature Prediction:/n{new_weather}')
print('/nNew Tempereature Prediction:',model.predict(X_test))


# In[61]:


## Predicting new weather patterns
new_weather =  scale.transform(X_train['Humidity'].values.reshape(-1,1))
print(f'Humidity Prediction:/n{new_weather}')
print('/nNew Humidity Prediction:',model.predict(X_test))


# In[ ]:




