import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader,Dataset
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import math
s = nn.Softmax()
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=600):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model-2)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model-2, 2).float() * (-math.log(10000.0) / (d_model-2)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.mask = torch.cat((torch.zeros(300,1),torch.ones(300,1))).unsqueeze(0)
        self.start = torch.zeros(600,1).unsqueeze(0)
        self.mask = torch.cat((self.start,self.mask),2)
        pe = torch.cat((pe,self.mask),2)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
   
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = s(attn_logits)
    values = torch.matmul(attention, v)
    return values, attention
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0, sequence_length = 300):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.input_dim = input_dim
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x
class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.input_dim = block_args["input_dim"]
        self.sequence_length = block_args["sequence_length"]
        self.positional_encoding = PositionalEncoding(d_model=block_args["input_dim"])
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
        self.fc = nn.Linear(block_args["input_dim"]*block_args["sequence_length"]*2, 1)
    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        for l in self.layers:
            x = l(x, mask=mask)
        
        return x,torch.sigmoid(self.fc(x.reshape(-1,self.input_dim* self.sequence_length*2)))

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

SEQ_LEN = 300
VOCAB_SIZE = 6
NUM_TRAINING_STEPS = 25000
BATCH_SIZE = 64
# This function generates data samples as described at the beginning of the
# script
def get_data_sample(batch_size=1):
    random_seq = torch.randint(low=0, high=VOCAB_SIZE - 1,
                               size=[batch_size,SEQ_LEN, SEQ_LEN + 2])                         
    ############################################################################
    # TODO: Calculate the ground truth output for the random sequence and store
    # it in 'gts'.
    ############################################################################
    gts = torch.empty(batch_size,dtype=float)
    for batch in range(batch_size):
        a = random_seq[batch][0]
        b = random_seq[batch][1]
        gts[batch] = ((torch.bincount(random_seq[batch])[a].item()-1)-(torch.bincount(random_seq[batch])[b].item()-1))
        
    
    # Ensure that GT is non-negative
    ############################################################################
    # TODO: Why is this needed?
    ############################################################################
    gts += SEQ_LEN
    return random_seq, gts.type(torch.FloatTensor)

# # Instantiate the network, loss function, and optimizer
# # inputs, labels = get_data_sample(BATCH_SIZE)
# net = TransformerEncoder(num_layers = 3,input_dim =128,num_heads =4, dim_feedforward = 20)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
# # Train the network
# num = 0
# for i in range(NUM_TRAINING_STEPS):
    
#     inputs = torch.randint(low=0, high=VOCAB_SIZE - 1,size=[BATCH_SIZE, 600,128],dtype=torch.float)
#     labels = torch.randint(low=0, high=1,size=[BATCH_SIZE, 1])
#     optimizer.zero_grad()
#     _,outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     # print(f'labels:{labels}')
#     # print(f'output:{torch.argmax(outputs, axis=-1)}')
#     accuracy = (torch.argmax(outputs, axis=-1) == labels).float().mean()

#     if i % 100 == 0:
#         for name,p in net.named_parameters():
#             if p.requires_grad == True:
#                 num+=1
#         print('[%d/%d] loss: %.3f, accuracy: %.3f' %
#               (i , NUM_TRAINING_STEPS - 1, loss.item(), accuracy.item()))
#     if i == NUM_TRAINING_STEPS - 1:
#         print('Final accuracy: %.3f, expected %.3f' %
#               (accuracy.item(), 1.0))