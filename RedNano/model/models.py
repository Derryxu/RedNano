import torch
from torch import nn
from .resnet import Resnet2, Resnet3, Conv1d
from .util import Full_NN, FlattenLayer, one_hot_embedding
# from .rnn import  RNN_LSTM, BiLSTM
from utils.constants import min_cov

class ReadLevelModel(nn.Module):

    def __init__(self, model_type, dropout_rate=0.5, hidden_size=128,
                 seq_len=5, signal_lens=65, embedding_size=4, device=0):
        super(ReadLevelModel, self).__init__()
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model_type = model_type
        self.raw, self.basecall = False, False
        if self.model_type in ["raw_signal", "comb_basecall_raw"]:
            self.raw = True
        if self.model_type in ["basecall", "comb_basecall_raw"]:
            self.basecall = True

        if self.basecall:
            self.embed = nn.Embedding(4, embedding_size)
            self.resnet = Resnet3(in_channels=embedding_size+4, out_channels=256)
        
        if self.raw:
            # self.bilstm = BiLSTM(4+1, hidden_size, num_layers, dropout_rate=dropout_rate, device=device)
            self.resnet_r = Resnet2(in_channels=4+1, out_channels=256)
        
        if self.raw and self.basecall:
            self.full = Full_NN(input_size=256+256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)
        elif self.raw: # only raw
            self.full = Full_NN(input_size=256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)
        else:  # only basecall
            self.full = Full_NN(input_size=256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)


    def forward(self, features):
        # (N * 20, 5+65, 5) -> (N * 20, 1)?
        kmer = features[:, 0, :]
        qual = features[:, 1, :]
        mis = features[:, 2, :]
        ins = features[:, 3, :]
        dele = features[:, 4, :]
        signals = torch.transpose(features[:, 5:, :], 1, 2)

        if self.basecall:
            y_kmer_embed = self.embed(kmer.long())
            qual = torch.reshape(qual, (-1, self.seq_len, 1)).float()
            mis = torch.reshape(mis, (-1, self.seq_len, 1)).float()
            ins = torch.reshape(ins, (-1, self.seq_len, 1)).float()
            dele = torch.reshape(dele, (-1, self.seq_len, 1)).float()
            y = torch.cat((y_kmer_embed, qual, mis, ins, dele), 2)  # (N, 5, 8)
            
            y = torch.transpose(y, 1, 2)
            y = self.resnet(y)
            # print("resnet output ", y.shape)  # (N, 2*hidden_size)
        
        if self.raw:
            signals = signals.float()
            signals_len = signals.shape[2]
            kmer_embed = one_hot_embedding(kmer.long(), signals_len) # torch.Size([N, seq_len*signal_len, 4])
            signals_ex = signals.reshape(signals.shape[0], -1, 1)
            # print("signals_ex: ", signals_ex.shape)  # (N, seq_len*signal_len, 1)
            # print("kmer_embed: ", kmer_embed.shape)  # (N, seq_len*singal_len, 4)
            x = torch.cat((kmer_embed, signals_ex), -1)  # (N, L, C)
            x = torch.transpose(x, 1, 2)
            x = self.resnet_r(x)

        ##################### Full connect layer
        if self.raw and self.basecall:
            z= torch.cat((x, y), 1)
            z = self.full(z)
        elif self.raw: # only raw
            z = self.full(x)
        else: # basecall
            z = self.full(y)

        ##################### sigmoid
        out_ = self.sigmoid(z)
        return out_


class SiteLevelModel(nn.Module):

    def __init__(self, model_type, dropout_rate=0.5, hidden_size=128, 
                 seq_len=5, signal_lens=65, embedding_size=4, device=0):
        super(SiteLevelModel, self).__init__()
        self.read_level_model = ReadLevelModel(model_type, dropout_rate, hidden_size, seq_len, signal_lens, 
                                               embedding_size, device=device)
    
    def get_read_level_probs(self, features):  # flattened features (N, 70, 5)
        return self.read_level_model(features)

    def forward(self, features):
        # (N, 20, 5+65, 5) -> (N, 20, 1) -> (N, 1)
        features = features.view(-1, features.shape[2], features.shape[3])
        probs = self.read_level_model(features).view(-1, min_cov)
        return 1 - torch.prod(1 - probs, axis=1)
