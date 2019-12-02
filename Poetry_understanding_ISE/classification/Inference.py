import torch
import csv
import torch.nn as nn
from util import get_num_lines, get_vocab, embed_sequence, get_word2idx_idx2word, get_embedding_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F

from util import get_num_lines, get_vocab, embed_sequence, get_word2idx_idx2word, get_embedding_matrix
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model import RNNSequenceClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import csv
import h5py
import random
import math
import numpy as np
import matplotlib

Gutenberg = []

with open('../Poetry/corpus_to_be_labeled.csv') as t:
    lines = csv.reader(t)
    next(lines)
    for line in lines:
        # Gutenberg.append([line[1].strip(), int(float(line[2])), int(float(line[3]))])
        Gutenberg.append([line[1].strip(), int(line[2])])

print('Poetry dataset size: ', len(Gutenberg))


# vocab is a set of words
vocab = get_vocab(Gutenberg)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
elmos_poetry = None
# suffix_embeddings: number of suffix tag is 2, and the suffix embedding dimension is 50
suffix_embeddings = nn.Embedding(2, 50)

'''
2. 2
embed the datasets
'''
# random.seed(0)  # set a seed
# random.shuffle(raw_poetry)
# for example in raw_poetry:
    # print(example[0])
    # print(example[1])
    # print(example[2])
    # print("-" * 10)


embedded_Gutenberg = [[embed_sequence(example[0], example[1], word2idx, glove_embeddings,
                                   elmos_poetry, suffix_embeddings)]
                   for example in Gutenberg]


print(embed_sequence)
print(embedded_Gutenberg)

from model import RNNSequenceClassifier



# net = RNNSequenceClassifier()
# net.eval()
net = torch.load('../models/LSTMSuffixElmoAtt_Poetry_fold_0_epoch_23.pt')
net.eval()

# corpus = torch.FloatTensor(embedded_Gutenberg)

torch.no_grad()

# predict = net(corpus, 100)
predict = F.softmax(net(embedded_Gutenberg))


print(predict)
 