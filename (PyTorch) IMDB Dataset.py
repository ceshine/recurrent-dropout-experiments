
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorflow.contrib.keras.python.keras.datasets import imdb
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

from yaringal_dataset import loader
from weight_drop import WeightDrop
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout

get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (10, 5)

# Global params:
NB_WORDS = 20000
SKIP_TOP = 0
TEST_SPLIT = 0.2
INIT_SEED = 2017
GLOBAL_SEED = 2018
MAXLEN = 80
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512


# In[2]:


(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=NB_WORDS)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)


# In[3]:


class Model(nn.Module):
    def __init__(self, nb_words, hidden_size=128, embedding_size=128, n_layers=1,
                 wdrop=0.25, odrop=0.25, edrop=0.1, idrop=0.25, variational=False,
                 standard_dropout=False, batch_first=True):
        super(Model, self).__init__()
        self.standard_dropout = standard_dropout
        self.lockdrop = LockedDropout(batch_first=batch_first)
        self.odrop = odrop
        self.idrop = idrop
        self.edrop = edrop
        self.n_layers = n_layers
        self.embedding = nn.Embedding(nb_words, embedding_size)
        self.rnns = [
            nn.LSTM(embedding_size if l == 0 else hidden_size,
                   hidden_size, num_layers=1, batch_first=batch_first)
            for l in range(n_layers)
        ]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop, variational=variational)
                         for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, X):
        emb = embedded_dropout(self.embedding, X, dropout=self.edrop if self.training else 0)
        if self.standard_dropout:
            raw_output = F.dropout(emb, p=self.idrop, training=self.training)
        else:
            raw_output = self.lockdrop(emb, self.idrop)
        new_hidden, new_cell_state = [], []
        for l, rnn in enumerate(self.rnns):
            raw_output, (new_h, new_c) = rnn(raw_output)
            if self.standard_dropout:
                raw_output = F.dropout(raw_output, p=self.odrop, training=self.training)
            else:
                raw_output = self.lockdrop(raw_output, self.odrop)         
            new_hidden.append(new_h)
            new_cell_state.append(new_c)
        hidden = torch.cat(new_hidden, 0)
        cell_state = torch.cat(new_cell_state, 0)
        final_output = self.output_layer(raw_output)
        return final_output[:, -1, 0], hidden, cell_state


# In[4]:


MC_ROUNDS = 10
def fit(model, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=30):
    epoch_losses = []
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        indices = torch.randperm(len(X_train)).cuda()
        losses, acc = [], []
        model.train()
        for i in range(0, len(X_train), BATCH_SIZE): #tqdm_notebook(range(0, len(X_train), BATCH_SIZE)):
            optimizer.zero_grad()
            pred, _, _ = model(Variable(X_train_tensor[indices[i:(i+BATCH_SIZE)]]))
            # print(pred.size())
            loss = criterion(
                pred,
                Variable(Y_train_tensor[indices[i:(i+BATCH_SIZE)]], requires_grad=False)
            )
            acc.append(
                torch.eq(
                    (F.sigmoid(pred).data > 0.5).float(), 
                    Y_train_tensor[indices[i:(i+BATCH_SIZE)]]
                )
            )
            losses.append(loss.data.cpu()[0])
            loss.backward()
            optimizer.step()
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        # Standard dropout approximation
        losses, acc=[], []
        model.eval()    
        for i in range(0, len(X_test), TEST_BATCH_SIZE):
            pred_test, _, _ = model(Variable(X_test_tensor[i:(i+TEST_BATCH_SIZE)], volatile=True))
            # print(pred.size())
            loss = F.binary_cross_entropy_with_logits(
                pred_test, Variable(Y_test_tensor[i:(i+TEST_BATCH_SIZE)]))
            acc.append(
                torch.eq(
                    (F.sigmoid(pred_test).data > 0.5).float(), 
                    Y_test_tensor[i:(i+TEST_BATCH_SIZE)]
                )
            )
            losses.append(loss.data.cpu()[0])      
        std_test_acc = torch.mean(torch.cat(acc).float())
        std_test_loss = np.mean(losses)
        # MC dropout
        losses, acc = [], []
        model.train()
        for i in range(0, len(X_test), TEST_BATCH_SIZE):
            pred_list = []
            for j in range(MC_ROUNDS):
                pred_test, _, _ = model(Variable(X_test_tensor[i:(i+TEST_BATCH_SIZE)], volatile=True))
                pred_list.append(pred_test.unsqueeze(0))
            pred_all = torch.mean(torch.cat(pred_list, 0), 0)
            loss = F.binary_cross_entropy_with_logits(
                pred_all, Variable(Y_test_tensor[i:(i+TEST_BATCH_SIZE)]))
            acc.append(
                torch.eq(
                    (F.sigmoid(pred_test).data > 0.5).float(), 
                    Y_test_tensor[i:(i+TEST_BATCH_SIZE)]
                )
            )
            losses.append(loss.data.cpu()[0])      
        mc_test_acc = torch.mean(torch.cat(acc).float())            
        mc_test_loss = np.mean(losses)
        epoch_losses.append([
            train_loss, std_test_loss, mc_test_loss,
            train_acc, std_test_acc, mc_test_acc])
        print("Epoch: {} Train: {:.4f}/{:.2f}%, Val Std: {:.4f}/{:.2f}%, Val MC: {:.4f}/{:.2f}%".format(
            epoch, train_loss, train_acc*100, std_test_loss, std_test_acc*100, mc_test_loss, mc_test_acc*100))
    return epoch_losses


# In[5]:


Y_train_tensor =  torch.from_numpy(Y_train).float().cuda()
Y_test_tensor =  torch.from_numpy(Y_test).float().cuda()
X_train_tensor =  torch.from_numpy(X_train).long().cuda()
X_test_tensor =  torch.from_numpy(X_test).long().cuda()


# ## Weight Dropped LSTM (w Embedding Dropout)

# In[6]:


model_1 = Model(NB_WORDS, wdrop=0.25, odrop=0.25, edrop=0.25, idrop=0.25)
model_1.cuda()
optimizer = torch.optim.Adam([
            {'params': model_1.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
        ],)
epoch_losses_1 = fit(
    model_1, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=15)


# ## No Dropout

# In[7]:


model_2 = Model(NB_WORDS, wdrop=0, odrop=0, edrop=0, idrop=0)
model_2.cuda()
optimizer = torch.optim.Adam(model_2.parameters(), lr=1e-4)
epoch_losses_2 = fit(model_2, optimizer, X_train_tensor, Y_train_tensor, 
                     X_test_tensor, Y_test_tensor, n_epochs=15)


# ## Naive Dropout (w/o Embedding Dropout)

# In[8]:


model_3 = Model(NB_WORDS, wdrop=0, odrop=0.25, edrop=0, idrop=0.25, standard_dropout=True)
model_3.cuda()
optimizer = torch.optim.Adam(model_3.parameters(), lr=1e-4)
epoch_losses_3 = fit(model_3, optimizer, X_train_tensor, Y_train_tensor, 
                     X_test_tensor, Y_test_tensor, n_epochs=15)


# ## Variational LSTM

# In[9]:


model_4 = Model(NB_WORDS, wdrop=0.25, odrop=0.25, edrop=0.25, idrop=0.25, variational=True)
model_4.cuda()
optimizer = torch.optim.Adam(model_4.parameters(), lr=2e-4)
epoch_losses_4 = fit(model_4, optimizer, X_train_tensor, Y_train_tensor, 
                     X_test_tensor, Y_test_tensor, n_epochs=15)


# ## Visualizations

# In[22]:


plt.title("Log Loss Comparison - Training Set")
plt.plot(np.arange(len(epoch_losses_1)), [x[0] for x in epoch_losses_1], label="weight dropped")
plt.plot(np.arange(len(epoch_losses_2)), [x[0] for x in epoch_losses_2], "g-", label="no dropout")
plt.plot(np.arange(len(epoch_losses_3)), [x[0] for x in epoch_losses_3], "y-", label="naive dropout")
plt.plot(np.arange(len(epoch_losses_4)), [x[0] for x in epoch_losses_4], "m-", label="variational")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")


# In[23]:


plt.title("Accuracy Comparison  - Training Set")
plt.plot(np.arange(len(epoch_losses_1)), [x[3] for x in epoch_losses_1], label="weight dropped")
plt.plot(np.arange(len(epoch_losses_2)), [x[3] for x in epoch_losses_2], "g-", label="no dropout")
plt.plot(np.arange(len(epoch_losses_3)), [x[3] for x in epoch_losses_3], "y-", label="standard")
plt.plot(np.arange(len(epoch_losses_4)), [x[3] for x in epoch_losses_4], "m--", label="variational")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")


# In[20]:


plt.title("Log Loss Comparison - Validation Set")
plt.plot(np.arange(len(epoch_losses_1)), [x[1] for x in epoch_losses_1], label="weight dropped")
plt.plot(np.arange(len(epoch_losses_2)), [x[1] for x in epoch_losses_2], "g-", label="no dropout")
plt.plot(np.arange(len(epoch_losses_3)), [x[1] for x in epoch_losses_3], "y-", label="naive dropout")
plt.plot(np.arange(len(epoch_losses_4)), [x[1] for x in epoch_losses_4], "m--", label="variational")
plt.plot(np.arange(len(epoch_losses_4)), [x[2] for x in epoch_losses_4], "m-", label="variational(mc)")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")


# In[19]:


plt.title("Accuracy Comparison  - Validation Set")
plt.plot(np.arange(len(epoch_losses_1)), [x[4] for x in epoch_losses_1], label="weight dropped")
plt.plot(np.arange(len(epoch_losses_2)), [x[4] for x in epoch_losses_2], "g-", label="no dropout")
plt.plot(np.arange(len(epoch_losses_3)), [x[4] for x in epoch_losses_3], "y-", label="standard")
plt.plot(np.arange(len(epoch_losses_4)), [x[4] for x in epoch_losses_4], "m--", label="variational")
plt.plot(np.arange(len(epoch_losses_4)), [x[5] for x in epoch_losses_4], "m-", label="variational(mc)")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")

