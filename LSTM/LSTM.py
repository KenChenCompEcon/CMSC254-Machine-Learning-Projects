import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

bob_train = open('bobsue-data/bobsue.seq2seq.train.tsv').read().lower().split('\n')[:-1]
bob_test = open('bobsue-data/bobsue.seq2seq.test.tsv').read().lower().split('\n')[:-1]
bob_dev = open('bobsue-data/bobsue.seq2seq.dev.tsv').read().lower().split('\n')[:-1]
voc = open('bobsue-data/bobsue.voc.txt').read().lower().split('\n')[:-1]

# Set the training set
x_train = []; y_train = []
for p in bob_train:
    pair = p.split('\t')
    x_train.append(pair[0]); y_train.append(pair[1])
    
print("The size of the vocabulary dictionary is: {}".format(len(voc)))

# Impelent the uniform distribution assignment
np.random.seed(25400)
voc_w2n = {}
idx = np.eye(len(voc), len(voc)).tolist()
for i in range(len(idx)):
    voc_w2n[voc[i]] = idx[i]

x_train_num = []; y_train_num = []
for i in range(len(x_train)):
    x_sen = torch.FloatTensor([voc_w2n[word] for word in x_train[i].split()])
    y_sen = torch.FloatTensor([voc_w2n[word] for word in y_train[i].split()])
    x_train_num.append(x_sen); y_train_num.append(y_sen)
    
class MyLSTM(nn.Module):
  
    def __init__(self, n_in, n_out):
        super(MyLSTM, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.fc_enco = nn.Linear(n_in+n_out, n_out)
        self.ic_enco = nn.Linear(n_in+n_out, n_out)
        self.oc_enco = nn.Linear(n_in+n_out, n_out)
        self.gc_enco = nn.Linear(n_in+n_out, n_out)
        self.fc_deco = nn.Linear(n_in+n_out, n_out)
        self.ic_deco = nn.Linear(n_in+n_out, n_out)
        self.oc_deco = nn.Linear(n_in+n_out, n_out)
        self.gc_deco = nn.Linear(n_in+n_out, n_out)

    def forward(self, c_old, h_old, x, mode):
        tensor = torch.cat([x, h_old])
        if mode=='encode':
            f = torch.sigmoid(self.fc_enco(tensor))
            i = torch.sigmoid(self.ic_enco(tensor))
            o = torch.sigmoid(self.oc_enco(tensor))
            g = torch.tanh(self.gc_enco(tensor))
            c = f*c_old + i*g
            h = o*torch.tanh(c)
        if mode=='decode':
            f = torch.sigmoid(self.fc_deco(tensor))
            i = torch.sigmoid(self.ic_deco(tensor))
            o = torch.sigmoid(self.oc_deco(tensor))
            g = torch.tanh(self.gc_deco(tensor))
            c = f*c_old + i*g
            h = o*torch.tanh(c)
        return c, h

sen_length = [len(sen) for sen in y_train_num]
print("The maximal length of all sentences is: {}".format(max(sen_length)))

def compute_loss(h_lst, y_sen):
    l_pred = len(h_lst); l_true = len(y_sen)
    if l_pred>=l_true:
        loss = sum([nn.MSELoss()(h_lst[i], y_sen[i]) for i in range(l_true)])/l_true
    else:
        h_lst = h_lst + [torch.zeros(y_sen[0].shape)]*(l_true-l_pred)
        loss = sum([nn.MSELoss()(h_lst[i], y_sen[i]) for i in range(l_true)])/l_true
    return loss

# Train the model with the training data set
hparams = {
    'learning_rate': 0.01,
    'epochs': 10,
    'logint': 1000
}

# Model Training
model = MyLSTM(len(voc), len(voc))
#model.to(DEVICE)
#mse = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=hparams['learning_rate'], momentum=0.9, weight_decay=1e-3)
epoch_losses = []
for i in range(hparams['epochs']):
    for j in range(len(x_train_num)):
        x_sen = x_train_num[j]
        y_sen = y_train_num[j]
        c_old = torch.zeros(len(voc))
        h_old = torch.zeros(len(voc))
        for m in range(len(x_sen)):
            x = x_sen[m]
            c_old, h_old = model(c_old, h_old, x, 'encode')
        h_lst = []; count = 0; word = '<s>'; x = torch.FloatTensor(voc_w2n[word])
        while count<30 and word!='<\s>':
            c_old, h_old = model(c_old, h_old, x, 'decode')
            h_lst.append(h_old); count+=1
            word = voc[x.max(0)[1]]
            x = torch.FloatTensor(voc_w2n[word])
        loss = compute_loss(h_lst, y_sen)*100
        opt.zero_grad()
        loss.backward()
        opt.step()
        if j%hparams['logint']==0:
            print("At epoch {}, # iteration {}, the loss is {}".format(i, j, loss))
    epoch_losses.append(loss)
    print('Epoch {:4} | Loss {:.3f}'.format(i, loss))
    print("-----------------------------------------------")
