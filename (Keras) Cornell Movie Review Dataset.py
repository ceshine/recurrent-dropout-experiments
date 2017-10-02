
# coding: utf-8

# # (Keras) Cornell Movie Review Dataset

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
plt.rcParams["figure.figsize"] = (8, 5)

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
                        test_every_X_epochs=2, verbose=0, T=10,
                        mean_y_train=mean_y_train, std_y_train=std_y_train,
                        loss='euclidean', batch_size=TEST_BATCH_SIZE)


# In[6]:


history_1 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=200, callbacks=[modeltest_1])


# In[24]:


print("Best RMSE: {:.4f} Best Epoch: {}".format(
    np.min([x[1] ** 0.5 for x in modeltest_1.history]), 
    (np.argmin([x[1] ** 0.5 for x in modeltest_1.history]) + 1)*2
))


# ## Standard LSTM w/o Dropout

# In[10]:


print('Build model...')
model = get_model(edrop=0, rdrop=0, odrop=0, idrop=0, weight_decay=1e-10, lr=1e-3)


# In[11]:


modeltest_2 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=2, verbose=0, T=1,
                        mean_y_train=mean_y_train, std_y_train=std_y_train,
                        loss='euclidean', batch_size=TEST_BATCH_SIZE)


# In[12]:


history_2 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=200, callbacks=[modeltest_2]
)


# In[25]:


print("Best RMSE: {:.4f} Best Epoch: {}".format(
    np.min([x[1] ** 0.5 for x in modeltest_2.history]), 
    (np.argmin([x[1] ** 0.5 for x in modeltest_2.history]) + 1)*2
))


# ## LSTM with Standard Dropout (different mask at differnt time steps)

# In[13]:


print('Build model...')
model = get_model(edrop=0.3, rdrop=0, odrop=0.3, idrop=0, weight_decay=1e-4, lr=1e-3)


# In[14]:


modeltest_3 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=2, verbose=0, T=10,
                        mean_y_train=mean_y_train, std_y_train=std_y_train,
                        loss='euclidean', batch_size=TEST_BATCH_SIZE)


# In[15]:


history_3 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=200, callbacks=[modeltest_3])


# In[26]:


print("Best RMSE: {:.4f} Best Epoch: {}".format(
    np.min([x[1] ** 0.5 for x in modeltest_3.history]), 
    (np.argmin([x[1] ** 0.5 for x in modeltest_3.history]) + 1)*2
))


# ## Visualizations

# In[40]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Raw MSE Comparison - Training Set")
plt.plot(np.arange(len(history_1.history["mean_squared_error"])), 
         history_1.history["mean_squared_error"], label="variational")
plt.plot(np.arange(len(history_2.history["mean_squared_error"])), 
         history_2.history["mean_squared_error"], "g-", label="no dropout")
plt.plot(np.arange(len(history_3.history["mean_squared_error"])), 
         history_3.history["mean_squared_error"], "y-", label="naive dropout")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("Raw MSE")
plt.subplot(1, 2, 2)
plt.title("(MC - Approx) Histogram")
plt.hist([x[1] ** 0.5 - x[0] ** 0.5 for x in modeltest_1.history], alpha=0.5, label="varational")
plt.hist([x[1] ** 0.5 - x[0] ** 0.5 for x in modeltest_3.history], alpha=0.5, label="navie dropout")
plt.legend(loc='best')
plt.xlabel("Difference in Raw MSE")
plt.ylabel("Count")
plt.xticks(fontsize=8, rotation=0)


# In[39]:


plt.title("RMSE Comparison - Validation Set")
plt.plot(np.arange(len(modeltest_1.history)), [x[1] ** 0.5 for x in modeltest_1.history], "b-", label="variational(mc)")
plt.plot(np.arange(len(modeltest_2.history)), [x[0] ** 0.5 for x in modeltest_2.history], "g-", label="no dropout")
plt.plot(np.arange(len(modeltest_3.history)), [x[1] ** 0.5 for x in modeltest_3.history], "y-", label="naive dropout(mc)")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("RMSE")

