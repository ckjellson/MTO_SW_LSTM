import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict
import random

class MTO_SW_LSTM(nn.Module):
    """
    LSTM class

    Parameters:
        window_size (int): Size of sliding window
        hidden_size (int): Number of hidden nodes in LSTM
        num_layers  (int): Number of LSTM layers
        n_features  (int): Number of values at each timestep
        stride      (int): Stride length, 0<stride<=window_size
        bsize       (int): Batch size during training
        device      (str): "cuda" or "cpu"
        bidir       (bool): Bidirectional LSTM
        nout        (list): List of integers [h1, h2, ..., hn, hout] for size of output DNN
        dropout     (int): Dropout for LSTM
        dropout2    (int): Dropout for output DNN
    
    """
    def __init__(self,window_size,hidden_size,num_layers,n_features,stride,bsize,device,bidir,nout,dropout,dropout2):
        super(MTO_SW_LSTM, self).__init__()
        
        # Initiate RNN
        self.rnn = nn.LSTM(input_size=n_features*window_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=bidir, dropout=dropout, batch_first=True)
        
        # Initiate output DNN
        units = []
        units.append(('fc0', nn.Linear(hidden_size, nout[0])))
        for i in range(len(nout)-2):
            units.append(('do'+str(i), nn.Dropout(dropout2)))
            units.append(('af'+str(i), nn.Tanh()))
            units.append(('lin'+str(i), nn.Linear(nout[i],nout[i+1])))
        units.append(('do'+str(len(nout)), nn.Dropout(dropout2)))
        units.append(('af'+str(len(nout)), nn.Tanh()))
        units.append(('lin'+str(len(nout)), nn.Linear(nout[-2],nout[-1])))
        self.dnn = nn.Sequential(OrderedDict(units))
        # Choose activation function:
        self.af = nn.Sigmoid()
        
        # Settings
        self.bidir = bidir
        self.stride = stride
        self.ws = window_size
        self.device = device
        self.bsize = bsize
        if bidir:
            self.nb_lstm_layers = num_layers*2
        else:
            self.nb_lstm_layers = num_layers
        self.nb_lstm_units = hidden_size
        self.to(self.device)
    
    '''
    Initialize hidden parameters in LSTM
    
    Parameters:
        using_gpu (bool):
        nseq      (int): Number of input sequences
    '''
    def init_hidden(self, using_gpu, nseq):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        # Choose torch.ones, or torch.zeros or torch.randn:
        hidden_a = torch.ones(self.nb_lstm_layers, nseq, self.nb_lstm_units)
        hidden_b = torch.ones(self.nb_lstm_layers, nseq, self.nb_lstm_units)

        if using_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    '''
    Forward pass
    
    Parameters:
        x           (list): List of sequence and label correspondences, [(X,y), ...], size(X)=(n_timesteps, n_features)
        using_gpu   (bool):
        nseq        (int): Number of input sequences
    '''
    def forward(self, x, using_gpu, nseq):
        
        self.hidden = self.init_hidden(using_gpu, nseq)
        
        lengths = [x_.size()[0] for x_ in x]
        
        maxlen = max(lengths)
        for i in range(len(lengths)):
            if (lengths[i]-self.ws)%self.stride==0:
                lengths[i] = lengths[i]
            else:
                lengths[i] = lengths[i]+(self.stride-(lengths[i]-self.ws)%self.stride)
        
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
        toadd = self.stride-maxlen%self.stride
        
        if toadd<self.stride:
            x_padded = torch.cat((x_padded,torch.zeros((x_padded.size()[0],toadd,x_padded.size()[2])).float().to(self.device)),dim=1)
        
        x_padded = torch.stack([torch.flatten(x_padded[:,i*self.stride:i*self.stride+self.ws,:],start_dim=1,end_dim=2) for i in range(np.int64((x_padded.size()[1]-self.ws)/self.stride+1))], dim=1)
        newlengths = []
        for i in range(len(lengths)):
            if lengths[i]>=self.ws:
                newlengths.append(np.int64((lengths[i]-self.ws)/self.stride+1))
            else:
                newlengths.append(1)
        lengths = newlengths
        b, s, n = x_padded.shape
        
        # pack padded sequence
        x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        out, self.hidden = self.rnn(x_padded, self.hidden)
        
        # unpack the feature vector
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        ndim = 1
        if self.bidir:
            ndim = 2
        out = out.view(b, s, ndim, self.nb_lstm_units)
        
        # many-to-one rnn, get the last result
        y = torch.stack([out[i,np.array(lengths[i])-1, -1, :] for i in range(len(lengths))],dim=0)
        y = self.af(self.dnn(y))
        
        return y
    
    '''
    Train model
    
    Parameters:
        train_data  (list): List of sequence and label correspondences, [(X,y), ...], size(X)=(n_timesteps, n_features)
        val_data    (list): List of sequence and label correspondences, [(X,y), ...], size(X)=(n_timesteps, n_features)
        epochs      (int):
        bsize       (int): Batch size
        learning_rate (float):
        using_gpu   (bool):
        testnbr     (int): How often to test on val_data
    '''
    def train_model(self, train_data, val_data, epochs, bsize, learning_rate, using_gpu, testnbr):
        nseq = len(train_data)
        
        optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        time0 = time.time()
        running_loss_list= []
        val_loss_list = []
                
        for ee in range(epochs):
            #print("Running epoch ", ee+1)
            
            idxs = list(range(nseq))
            random.shuffle(idxs)
            
            self.train()
            
            # defining gradient in each epoch as 0
            if using_gpu:
                for param in self.parameters():
                    param.grad = None
            else:
                optimizer.zero_grad()
            
            running_loss = 0
            
            for bb in range(np.int64(np.floor(nseq/bsize))):
                
                x_train = [train_data[i][0] for i in idxs[bb*bsize:(bb+1)*bsize]]
                y_train = torch.cat([train_data[i][1] for i in idxs[bb*bsize:(bb+1)*bsize]],dim=0)
                
                nbatchseq = len(x_train)
                
                out = self.forward(x_train,using_gpu,nbatchseq)
                
                loss = criterion(out, y_train)
                
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
            
            # Calculate losses and print epoch information
            if (ee+1)%testnbr==0 or ee==0:
                self.eval()
                x_val = [obj[0] for obj in val_data]
                y_val = torch.cat([obj[1] for obj in val_data],dim=0)
                mean_train_loss = running_loss/np.floor(nseq/bsize)
                running_loss_list.append(mean_train_loss)
                with torch.no_grad():
                    nvalseq = len(x_val)
                    out = self.forward(x_val,using_gpu,nvalseq)
                    loss = criterion(out, y_val)
                    val_loss_list.append(loss.item())
                print("Epoch {} - Training loss: {} - Validation loss: {}".format(ee+1, mean_train_loss, val_loss_list[-1]))
        print("Finished training in ", time.time()-time0, " seconds")
        return None