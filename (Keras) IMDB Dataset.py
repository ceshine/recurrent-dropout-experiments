
# coding: utf-8

# # (Keras) IMDB Dataset

# In[1]:


import numpy as np
from tensorflow.contrib.keras.python.keras.optimizers import SGD, RMSprop, Adagrad
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers.embeddings import Embedding
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM, GRU, SimpleRNN
from tensorflow.contrib.keras.python.keras.regularizers import l2
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.datasets import imdb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
MAXLEN = 80
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512
WEIGHT_DECAY = 1e-4


# In[2]:


np.random.seed(100)


# In[3]:


(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=NB_WORDS)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)


# In[4]:


def get_model(idrop=0.2, edrop=0.1, odrop=0.25, rdrop=0.2, weight_decay=WEIGHT_DECAY):
    model = Sequential()
    model.add(Embedding(NB_WORDS, 128, embeddings_regularizer=l2(weight_decay),
                        input_length=MAXLEN))  # , batch_input_shape=(batch_size, maxlen)))
    if edrop:
        model.add(Dropout(edrop))
    model.add(LSTM(128, kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                   bias_regularizer=l2(weight_decay), dropout=idrop, recurrent_dropout=rdrop))
    if odrop:
        model.add(Dropout(odrop))
    model.add(Dense(1, kernel_regularizer=l2(weight_decay),
                    bias_regularizer=l2(weight_decay), activation='sigmoid'))
    optimizer = Adam(1e-3)
    model.compile(loss='binary_crossentropy', metrics=["binary_accuracy"], optimizer=optimizer)
    return model


# ## Normal Variational LSTM (w/o Embedding Dropout)
# All models in this notebook do not have embedding dropout as Keras does not have such layer.

# In[5]:


print('Build model...')
model = get_model(idrop=0.25, edrop=0, odrop=0.25, rdrop=0.25, weight_decay=1e-4)


# In[6]:


modeltest_1 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=1, verbose=0,
                        loss='binary', batch_size=TEST_BATCH_SIZE)


# In[7]:


history_1 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=20, callbacks=[modeltest_1])


# In[11]:


best_epoch = np.argmin([x[1] for x in modeltest_1.history[:18]]) + 1
print("Best Loss: {:.4f} Acc: {:.2f}% Best Epoch: {}".format(
    modeltest_1.history[best_epoch-1][1], 
    modeltest_1.history[best_epoch-1][3] * 100, 
    best_epoch
))


# In[12]:


plt.title("Log Loss Comparison")
plt.plot(np.arange(len(modeltest_1.history)), [x[0] for x in modeltest_1.history], label="std")
plt.plot(np.arange(len(modeltest_1.history)), [x[1] for x in modeltest_1.history], "g-", label="mc")
plt.legend(loc='best')


# In[13]:


plt.title("Accuracy Comparison")
plt.plot(np.arange(0, len(modeltest_1.history)), [x[2] for x in modeltest_1.history], label="std")
plt.plot(np.arange(0, len(modeltest_1.history)), [x[3] for x in modeltest_1.history], "g-", label="mc")
plt.legend(loc='best')


# ## Standard LSTM
# I choose to keep a very low weight decay because assigning zero seems to cause some problems.

# In[14]:


print('Build model...')
model = get_model(edrop=0, rdrop=0, odrop=0, idrop=0, weight_decay=1e-10)


# In[15]:


modeltest_2 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=1, verbose=0, T=1,
                        loss='binary', batch_size=TEST_BATCH_SIZE)


# In[17]:


history_2 = model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=20, callbacks=[modeltest_2])


# In[25]:


best_epoch = np.argmin([x[1] for x in modeltest_2.history]) + 1
print("Best Loss: {:.4f} Acc: {:.2f}% Best Epoch: {}".format(
    modeltest_2.history[best_epoch-1][1], 
    modeltest_2.history[best_epoch-1][3] * 100, 
    best_epoch
))


# ## LSTM with Standard Dropout (different mask at differnt time steps)

# In[20]:


print('Build model...')
model = get_model(edrop=0.25, rdrop=0, odrop=0.25, idrop=0, weight_decay=1e-4)


# In[21]:


modeltest_3 = ModelTest(X_test, Yt=Y_test,
                        test_every_X_epochs=1, verbose=0, T=10,
                        loss='binary', batch_size=TEST_BATCH_SIZE)


# In[22]:


history_3 =model.fit(
    X_train, Y_train,
    verbose=2,
    shuffle=True,
    # validation_data=[X_test, Y_test],
    batch_size=BATCH_SIZE, epochs=20, callbacks=[modeltest_3])


# In[24]:


best_epoch = np.argmin([x[1] for x in modeltest_3.history[:19]]) + 1
print("Best Loss: {:.4f} Acc: {:.2f}% Best Epoch: {}".format(
    modeltest_3.history[best_epoch-1][1], 
    modeltest_3.history[best_epoch-1][3] * 100, 
    best_epoch
))


# ## Visualizations

# In[40]:


bins = np.arange(-0.1, 0.035, 0.01)


# In[53]:


len(history_2.history["binary_accuracy"])


# In[54]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Accuracy Comparison - Training Set")
plt.plot(np.arange(len(history_2.history["binary_accuracy"])), 
         np.array(history_1.history["binary_accuracy"][:20]) * 100, label="variational")
plt.plot(np.arange(len(history_2.history["binary_accuracy"])), 
         np.array(history_2.history["binary_accuracy"]) * 100, "g-", label="no dropout")
plt.plot(np.arange(len(history_3.history["binary_accuracy"])), 
         np.array(history_3.history["binary_accuracy"]) * 100, "y-", label="naive dropout")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.subplot(1, 2, 2)
plt.title("(MC - Approx) Histogram")
plt.hist([x[1] - x[0]  for x in modeltest_1.history[:17]], bins=bins, alpha=0.5, label="varational")
plt.hist([x[1] - x[0]  for x in modeltest_3.history[:17]], bins=bins, alpha=0.5, label="navie dropout")
plt.legend(loc='best')
plt.xlabel("Difference in Loss")
plt.ylabel("Count")
plt.xticks(fontsize=8, rotation=0)


# In[60]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Log Loss Comparison - Validation Set")
plt.plot(np.arange(len(modeltest_2.history)), [x[1] for x in modeltest_1.history[:20]], "b-", label="variational(mc)")
plt.plot(np.arange(len(modeltest_2.history)), [x[1] for x in modeltest_2.history], "g-", label="no dropout")
plt.plot(np.arange(len(modeltest_3.history)), [x[1] for x in modeltest_3.history], "y-", label="naive dropout(mc)")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("Log Loss")
plt.subplot(1, 2, 2)
plt.title("Accuracy Comparison - Validation Set")
plt.plot(np.arange(len(modeltest_2.history)), [x[3] * 100 for x in modeltest_1.history[:20]], "b-", label="variational(mc)")
plt.plot(np.arange(len(modeltest_2.history)), [x[3] * 100 for x in modeltest_2.history], "g-", label="no dropout")
plt.plot(np.arange(len(modeltest_3.history)), [x[3] * 100 for x in modeltest_3.history], "y-", label="naive dropout(mc)")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("Accuracy (%)")


# In[ ]:




