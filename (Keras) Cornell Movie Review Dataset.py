
# coding: utf-8

# Based on https://github.com/yaringal/BayesianRNN/blob/master/Example/sentiment_lstm_regression.py

# In[1]:


import numpy as np
from tensorflow.contrib.keras.python.keras.optimizers import SGD, RMSprop, Adagrad
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers.embeddings import Embedding
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM, GRU, SimpleRNN
from tensorflow.contrib.keras.python.keras.regularizers import l2
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from yaringal_callbacks import ModelTest
from yaringal_dataset import loader

get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (10, 5)

# Global params:
NB_WORDS = 20000
SKIP_TOP = 0
TEST_SPLIT = 0.2
INIT_SEED = 2017
GLOBAL_SEED = 2018
MAXLEN = 200
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512


# In[2]:


dataset = loader(INIT_SEED, MAXLEN, NB_WORDS, SKIP_TOP, TEST_SPLIT)

X_train, X_test, Y_train, Y_test = dataset.X_train, dataset.X_test, dataset.Y_train, dataset.Y_test
mean_y_train, std_y_train = dataset.mean_y_train, dataset.std_y_train


# In[3]:


def get_model(idrop=0.2, edrop=0.1, odrop=0.25, rdrop=0.2, weight_decay=1e-4, lr=1e-3):
    model = Sequential()
    model.add(Embedding(NB_WORDS, 128, embeddings_regularizer=l2(weight_decay),
                        input_length=MAXLEN)) 
    if edrop:
        model.add(Dropout(edrop))
    model.add(LSTM(128, kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                   bias_regularizer=l2(weight_decay), dropout=idrop, recurrent_dropout=rdrop))
    if odrop:
        model.add(Dropout(odrop))
    model.add(Dense(1, kernel_regularizer=l2(weight_decay),
                    bias_regularizer=l2(weight_decay)))
    optimizer = Adam(lr)
    model.compile(loss='mse', metrics=["mse"], optimizer=optimizer)
    return model


# ## Normal Variational LSTM (w/o Embedding Dropout)

# In[4]:


print('Build model...')
model = get_model(rdrop=0.25, odrop=0.25, edrop=0, idrop=0.25, weight_decay=1e-4, lr=1e-3)


# In[5]:


modeltest_1 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=1, verbose=0, T=4,
                        mean_y_train=mean_y_train, std_y_train=std_y_train,
                        loss='euclidean', batch_size=TEST_BATCH_SIZE)


# In[6]:


hisotry_1 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=250, callbacks=[modeltest_1])


# ## Standard LSTM w/o Dropout

# In[7]:


print('Build model...')
model = get_model(edrop=0, rdrop=0, odrop=0, idrop=0, weight_decay=1e-10, lr=1e-3)


# In[8]:


modeltest_2 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=1, verbose=0, T=2,
                        mean_y_train=mean_y_train, std_y_train=std_y_train,
                        loss='euclidean', batch_size=TEST_BATCH_SIZE)


# In[9]:


hisotry_2 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=250, callbacks=[modeltest_2])


# ## LSTM with Standard Dropout (different mask at differnt time steps)

# In[10]:


print('Build model...')
model = get_model(edrop=0.3, rdrop=0, odrop=0.3, idrop=0, weight_decay=1e-4, lr=1e-3)


# In[11]:


modeltest_3 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=1, verbose=0, T=4,
                        mean_y_train=mean_y_train, std_y_train=std_y_train,
                        loss='euclidean', batch_size=TEST_BATCH_SIZE)


# In[12]:


hisotry_3 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=250, callbacks=[modeltest_3])


# ## Visualizations

# In[14]:


plt.title("Log Loss Comparison")
plt.plot(np.arange(len(modeltest_1.history)), [x[0] ** 0.5 for x in modeltest_1.history], label="variational")
plt.plot(np.arange(len(modeltest_2.history)), [x[0] ** 0.5 for x in modeltest_2.history], "g-", label="no dropout")
plt.plot(np.arange(len(modeltest_3.history)), [x[0] ** 0.5 for x in modeltest_3.history], "y-", label="naive dropout")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")


# In[ ]:




