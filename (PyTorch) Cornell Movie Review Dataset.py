
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import numpy as np
import matplotlib.pyplot as plt

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
MAXLEN = 200
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512


# In[2]:


dataset = loader(INIT_SEED, MAXLEN, NB_WORDS, SKIP_TOP, TEST_SPLIT)

X_train, X_test, Y_train, Y_test = dataset.X_train, dataset.X_test, dataset.Y_train, dataset.Y_test
mean_y_train, std_y_train = dataset.mean_y_train, dataset.std_y_train


# In[3]:


def inverse_transform(v):
    return v * std_y_train + mean_y_train 


# In[4]:


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


# In[5]:


MC_ROUNDS = 10
def fit(model, optimizer, X_train_tensor, Y_train_tensor, 
        X_test_tensor, Y_test_tensor, n_epochs=30, mc_dropout=True):
    epoch_losses = []
    criterion = torch.nn.MSELoss()
    for epoch in range(n_epochs):
        indices = torch.randperm(len(X_train)).cuda()
        losses, losses_raw = [], []
        model.train()
        for i in range(0, len(X_train), BATCH_SIZE): #tqdm_notebook(range(0, len(X_train), BATCH_SIZE)):
            optimizer.zero_grad()
            pred, _, _ = model(Variable(X_train_tensor[indices[i:(i+BATCH_SIZE)]]))
            loss_raw = criterion(
                pred,
                Variable(Y_train_tensor[indices[i:(i+BATCH_SIZE)]], requires_grad=False)
            )
            loss = F.mse_loss(inverse_transform(pred),
                              inverse_transform(
                                  Variable(Y_train_tensor[indices[i:(i+BATCH_SIZE)]], requires_grad=False)))
            losses_raw.append(loss_raw.data.cpu()[0])
            losses.append(loss.data.cpu()[0])
            loss_raw.backward()
            optimizer.step()
        train_loss = np.mean(losses)** 0.5
        train_loss_raw = np.mean(losses_raw)
        # Standard dropout approximation
        losses, losses_raw = [], []
        model.eval()    
        for i in range(0, len(X_test), TEST_BATCH_SIZE):
            pred_test, _, _ = model(Variable(X_test_tensor[i:(i+TEST_BATCH_SIZE)], volatile=True))
            loss_raw = F.mse_loss(pred_test, Variable(Y_test_tensor[i:(i+TEST_BATCH_SIZE)]))
            loss = F.mse_loss(inverse_transform(pred_test),
                              inverse_transform(Variable(Y_test_tensor[i:(i+TEST_BATCH_SIZE)])))
            losses_raw.append(loss_raw.data.cpu()[0])
            losses.append(loss.data.cpu()[0])
        std_test_loss = np.mean(losses) ** 0.5
        std_test_loss_raw = np.mean(losses_raw)
        if mc_dropout:
            # MC dropout
            losses, losses_raw = [], []
            model.train()
            for i in range(0, len(X_test), TEST_BATCH_SIZE):
                pred_list = []
                for j in range(MC_ROUNDS):
                    pred_test, _, _ = model(Variable(X_test_tensor[i:(i+TEST_BATCH_SIZE)], volatile=True))
                    pred_list.append(pred_test.unsqueeze(0))
                pred_all = torch.mean(torch.cat(pred_list, 0), 0)
                loss_raw = F.mse_loss(pred_all, Variable(Y_test_tensor[i:(i+TEST_BATCH_SIZE)]))
                loss = F.mse_loss(inverse_transform(pred_all),
                                  inverse_transform(Variable(Y_test_tensor[i:(i+TEST_BATCH_SIZE)])))        
                losses_raw.append(loss_raw.data.cpu()[0])
                losses.append(loss.data.cpu()[0])    
            mc_test_loss = np.mean(losses) ** 0.5
            mc_test_loss_raw = np.mean(losses_raw)
            epoch_losses.append([
                train_loss, std_test_loss, mc_test_loss,
                train_loss_raw, std_test_loss_raw, mc_test_loss_raw
            ])
            print("Epoch: {} Train: {:.4f}/{:.4f}, Val Std: {:.4f}/{:.4f}, Val MC: {:.4f}/{:.4f}".format(
                epoch, train_loss, std_test_loss_raw, std_test_loss, std_test_loss_raw, mc_test_loss, mc_test_loss_raw))
        else:
            epoch_losses.append([train_loss, std_test_loss, mc_test_loss])
            print("Epoch: {} Train: {:.4f}/{:.4f}, Val Std: {:.4f}/{:.4f}".format(
                epoch, train_loss, std_test_loss_raw, std_test_loss, std_test_loss_raw))
    return epoch_losses


# In[6]:


Y_train_tensor =  torch.from_numpy(Y_train).float().cuda()
Y_test_tensor =  torch.from_numpy(Y_test).float().cuda()
X_train_tensor =  torch.from_numpy(X_train).long().cuda()
X_test_tensor =  torch.from_numpy(X_test).long().cuda()


# ## Weight Dropped LSTM (w Embedding Dropout)

# In[22]:


model_1 = Model(NB_WORDS + dataset.index_from, wdrop=0.05, odrop=0.1, edrop=0.2, idrop=0.1)
model_1.cuda()
optimizer = torch.optim.Adam([
            {'params': model_1.parameters(), 'lr': 1e-4, 'weight_decay': 2e-5}
        ],)
epoch_losses_1 = fit(
    model_1, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=50)


# ## No Dropout

# In[8]:


model_2 = Model(NB_WORDS + dataset.index_from, wdrop=0, odrop=0, edrop=0, idrop=0)
model_2.cuda()
optimizer = torch.optim.Adam([
            {'params': model_2.parameters(), 'lr': 1e-4}
        ],)
epoch_losses_2 = fit(
    model_2, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=50)


# ## Naive Dropout (w/o Embedding Dropout)

# In[9]:


model_3 = Model(NB_WORDS + dataset.index_from, 
                wdrop=0, odrop=0.2, edrop=0, idrop=0.2, standard_dropout=True)
model_3.cuda()
optimizer = torch.optim.Adam([
            {'params': model_3.parameters(), 'lr': 1e-4, 'weight_decay': 2e-5}
        ],)
epoch_losses_3 = fit(
    model_3, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=50)


# ## Variational LSTM

# In[25]:


model_4 = Model(NB_WORDS + dataset.index_from, wdrop=0.02, odrop=0.1, edrop=0.1, idrop=0.1, variational=True)
model_4.cuda()
optimizer = torch.optim.Adam([
            {'params': model_4.parameters(), 'lr': 1e-4, 'weight_decay': 2e-5}
        ],)
epoch_losses_4 = fit(
    model_4, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=50)


# ## Variational LSTM w/o Recurrent Dropout

# In[26]:


model_5 = Model(NB_WORDS + dataset.index_from, wdrop=0, odrop=0.1, edrop=0.2, idrop=0.1)
model_5.cuda()
optimizer = torch.optim.Adam([
            {'params': model_5.parameters(), 'lr': 1e-4, 'weight_decay': 2e-5}
        ],)
epoch_losses_5 = fit(
    model_5, optimizer, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, n_epochs=50)


# ## Visualizations

# In[31]:


plt.title("RMSE Comparison - Training Set")
plt.plot(np.arange(len(epoch_losses_1)), [x[0] for x in epoch_losses_1], label="weight dropped")
plt.plot(np.arange(len(epoch_losses_2)), [x[0] for x in epoch_losses_2], "g-", label="no dropout")
plt.plot(np.arange(len(epoch_losses_3)), [x[0] for x in epoch_losses_3], "y-", label="naive dropout")
plt.plot(np.arange(len(epoch_losses_4)), [x[0] for x in epoch_losses_4], "m-", label="variational")
plt.plot(np.arange(len(epoch_losses_5)), [x[0] for x in epoch_losses_5], "c-", label="variational-2")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")


# In[33]:


plt.title("RMSE Comparison - Validation Set")
plt.plot(np.arange(len(epoch_losses_1)), [x[1] for x in epoch_losses_1], label="weight dropped")
plt.plot(np.arange(len(epoch_losses_2)), [x[1] for x in epoch_losses_2], "g-", label="no dropout")
plt.plot(np.arange(len(epoch_losses_3)), [x[1] for x in epoch_losses_3], "y-", label="naive dropout")
plt.plot(np.arange(len(epoch_losses_4)), [x[1] for x in epoch_losses_4], "m-", label="variational")
plt.plot(np.arange(len(epoch_losses_4)), [x[2] for x in epoch_losses_4], "m--", label="variational(mc)")
plt.plot(np.arange(len(epoch_losses_5)), [x[1] for x in epoch_losses_5], "c-", label="variational-2")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("logloss")


# In[ ]:




