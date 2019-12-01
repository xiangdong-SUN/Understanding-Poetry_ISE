import torch
import csv
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F

from model import RNNSequenceClassifier

Gutenberg = []
corpus = []
# normal version
with open('../Poetry/corpus_to_be_labeled.csv') as t:
    lines = csv.reader(t)
    next(lines)
    for line in lines:
        Gutenberg.append([line[0].strip()])
print('Poetry dataset size: ', len(Gutenberg))

# net = RNNSequenceClassifier()
# net.eval()
net = torch.load('../models/LSTMSuffixElmoAtt_Poetry_fold_0_epoch_23.pt')
net.eval()

corpus.append(torch.FloatTensor(Gutenberg))

torch.no_grad()

predict = net(corpus, 200)
# predict = F.softmax(net(corpus))

print(predict)
 