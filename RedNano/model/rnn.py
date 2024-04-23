import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.nn as nn
from utils.constants import use_cuda
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from .util import Full_NN
device = 'cuda' if use_cuda else 'cpu'

# Creating BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5, device=0):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        
        h0 = autograd.Variable(torch.randn(self.num_layers * 2, x.size(0), self.hidden_size))
        c0 = autograd.Variable(torch.randn(self.num_layers * 2, x.size(0), self.hidden_size))
        if use_cuda:
            h0, c0 = h0.cuda(self.device), c0.cuda(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)  #(N, 2*hidden_size)
        return out

class BiLSTMtrain(nn.Module):    # for train
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5, device=0):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        
        h0 = autograd.Variable(torch.randn(self.num_layers * 2, x.size(0), self.hidden_size))
        c0 = autograd.Variable(torch.randn(self.num_layers * 2, x.size(0), self.hidden_size))
        if use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        
        out, _ = self.lstm(x, (h0, c0))
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)  #(N, 2*hidden_size)
        return out

# Creating RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        #self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = autograd.Variable(torch.randn(self.num_layers*2, x.size(0), self.hidden_size)).to(device)
    # Forward prop
        out, _ = self.rnn(x, h0)
        #out = out.reshape(out.shape[0], -1)
        #out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout_rate)
        #self.fc = nn.Linear(hidden_size * sequence_length, hidden_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # Set initial hidden and cell states
        h0 = autograd.Variable(torch.randn(self.num_layers*2, x.size(0), self.hidden_size)).to(device)
        c0 = autograd.Variable(torch.randn(self.num_layers*2, x.size(0), self.hidden_size)).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return out


# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length=1):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        #self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        self.gru.flatten_parameters()
        h0 = autograd.Variable(torch.randn(self.num_layers*2, x.size(0), self.hidden_size)).to(device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        #out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        #out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class LSTM_attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, sequence_length=1):
        super(LSTM_attn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True,dropout=dropout_rate)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        #self.fc = nn.Linear(hidden_size * sequence_length, hidden_size)

    def attention_net(self, lstm_output):
        # Attention过程
        u = torch.tanh(torch.matmul(lstm_output, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = lstm_output * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        attn_output = torch.sum(scored_x, dim=1) #加权求和

        #state = lstm_output.permute(1, 0, 2)
        return attn_output

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = autograd.Variable(torch.randn(self.num_layers*2, x.size(0), self.hidden_size)).to(device)
        c0 = autograd.Variable(torch.randn(self.num_layers*2, x.size(0), self.hidden_size)).to(device)

        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        lstm_output,  _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        #out = self.fc(out)
        attn_output = self.attention_net(lstm_output)
        return attn_output

import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
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


import torch
import torch.nn as nn

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights