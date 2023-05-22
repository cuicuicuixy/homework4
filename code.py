import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
# import time
from scipy.sparse import csr_matrix
# from tensorboardX import SummaryWriter

#读取文本文件中的内容，将读取的内容转换为字符串类型，输出前100个字符的内容。
path = r"E:\DLNL\homework4\神雕选段.txt"
with open(path, 'r', encoding='ANSI') as f:
    data = f.readlines()
data=''.join(data)#读取的列表转换为字符串
print(data[:100])
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
X_train = csr_matrix((len(data), len(chars)), dtype=np.int)
char_id = np.array([chars.index(c) for c in data])
X_train[np.arange(len(data)), char_id] = 1
y_train = np.roll(char_id,-1)
X_train.shape
y_train.shape
def get_batch(X_train, y_train, seq_length):
    X = X_train
    y = torch.from_numpy(y_train).long()
    for i in range(0, len(y), seq_length):   
        id_stop = i+seq_length if i+seq_length < len(y) else len(y)
        yield([torch.from_numpy(X[i:id_stop].toarray().astype(np.float32)), 
               y[i:id_stop]])

class nn_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, X, hidden):
        _, hidden = self.lstm(X, hidden)
        output = self.out(hidden[0])
        return output, hidden
    
    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size)
               )
     
hidden_size = 256
seq_length = 25
rnn = nn_LSTM(vocab_size, hidden_size, vocab_size)
#设置损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)

def train(X_batch, y_batch):
    h_prev = rnn.initHidden()
    optimizer.zero_grad()
    batch_loss = torch.tensor(0, dtype=torch.float)
    
    for i in range(len(X_batch)):
        y_score, h_prev = rnn(X_batch[i].view(1,1,-1), h_prev)
        loss = loss_fn(y_score.view(1,-1), y_batch[i].view(1))
        batch_loss += loss
    batch_loss.backward()
    optimizer.step()

    return y_score, batch_loss/len(X_batch)

def sample_chars(rnn, X_seed, h_prev, length=20):
    '''Generate text using trained model'''
    X_next = X_seed
    results = []
    with torch.no_grad():
        for i in range(length):        
            y_score, h_prev = rnn(X_next.view(1,1,-1), h_prev)
            y_prob = nn.Softmax(0)(y_score.view(-1)).detach().numpy()
            y_pred = np.random.choice(chars,1, p=y_prob).item()
            results.append(y_pred)
            X_next = torch.zeros_like(X_seed)
            X_next[chars.index(y_pred)] = 1
    return ''.join(results)


all_losses = []
print_every = 100



def compute_ngram_overlap(text, n=3):
    """
    计算文本中n-gram的重复率
    :param text: 生成的文本
    :param n: n-gram的大小
    :return: 重复率
    """
    ngrams = set()
    overlap = 0
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        if ngram in ngrams:
            overlap += 1
        ngrams.add(ngram)
    return overlap / (len(text) - n + 1)

for epoch in range(20):    
    for batch in get_batch(X_train, y_train, seq_length):
        X_batch, y_batch = batch
        _, batch_loss = train(X_batch, y_batch)
        all_losses.append(batch_loss.item())
        if len(all_losses)%print_every==1:
            print(f'----\nRunning Avg Loss:{np.mean(all_losses[-print_every:])} at iter: {len(all_losses)}\n----')
            te=sample_chars(rnn, X_batch[0], rnn.initHidden(), 600)
            print(te)
            print('重复率：{}',compute_ngram_overlap(te, n=3))   

print(sample_chars(rnn, X_batch[20], rnn.initHidden(), 200))
               
