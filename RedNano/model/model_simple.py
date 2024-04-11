import torch
from torch import nn
from .resnet import Resnet, Conv1d
from .util import Full_NN, FlattenLayer, one_hot_embedding
from .rnn import  RNN_LSTM, LSTM_attn

class Model(nn.Module):

    def __init__(self, model_type, dropout_rate=0.5,  hidden_size=512, num_hidden=128,  seq_lenr=5, seq_lene=11, 
                 signal_lens=65, embedding_size=4, num_layers = 2):
        super(Model, self).__init__()
        self.seq_lenr = seq_lenr
        self.seq_lene = seq_lene
        self.fc = FlattenLayer()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model_type = model_type
        self.raw, self.basecall, self.signalFea = False, False, False
        if self.model_type in ["raw_signal", "comb_basecall_raw", "comb_basecallns_raw", "comb_signalFea_raw", "comb_basecall_signalFea_raw"]:
            self.raw = True
        if self.model_type in ["basecall", "comb_basecall_signalFea", "comb_basecall_raw", "comb_basecallns_raw", "comb_basecall_signalFea_raw"]:
            self.basecall = True
        if self.model_type in ["signalFea", "comb_basecall_signalFea", "comb_signalFea_raw","comb_basecall_signalFea_raw"]:
            self.signalFea = True
        self.no_seq = False
        if self.model_type in ["comb_basecallns_raw"]:
            self.no_seq = True

        # ["basecall", "signalFea", "raw_signal", \
        # "comb_basecall_signalFea","comb_basecall_raw",\
        # "comb_signalFea_raw","comb_basecall_signalFea_raw"]

        if self.basecall or self.signalFea:
            if not self.no_seq:
                self.embed = nn.Embedding(4, embedding_size)
                self.resnet = Resnet(in_channels=(embedding_size+4))
                if self.basecall and self.signalFea:
                    self.resnet = Resnet(in_channels=(embedding_size+4+4))
            else:
                self.resnet = Resnet(in_channels=4)
                if self.basecall and self.signalFea:
                    self.resnet = Resnet(in_channels=(4+4))
            self.full = Full_NN(input_size=hidden_size, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)
        
        if self.raw:
            self.cnn = Conv1d(in_channels=(num_hidden*2), out_channels=hidden_size)
            self.bilstm =RNN_LSTM((5), num_hidden, num_layers, dropout_rate=0.2)
            self.full = Full_NN(input_size=hidden_size, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)
            if self.model_type != "signalFea":
                self.full_concat = Full_NN(input_size=hidden_size*2, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)


    def forward(self, kmere, kmerr, base_means, base_meadian, base_stds, base_signal_lens, signals, qual, mis, ins, dele):

        if self.basecall or self.signalFea:
            if not self.no_seq:
                y_kmere_embed = self.embed(kmere.long())
            if self.basecall:
                qual = torch.reshape(qual, (-1, self.seq_lene, 1)).float()
                mis = torch.reshape(mis, (-1, self.seq_lene, 1)).float()
                ins = torch.reshape(ins, (-1, self.seq_lene, 1)).float()
                dele = torch.reshape(dele, (-1, self.seq_lene, 1)).float()
                if not self.no_seq:
                    y = torch.cat((y_kmere_embed, qual, mis, ins, dele),2)
                else:
                    y = torch.cat((qual, mis, ins, dele),2)
            
            if self.signalFea :
                base_means = torch.reshape(base_means, (-1, self.seq_lene, 1)).float()
                base_meadian = torch.reshape(base_meadian, (-1, self.seq_lene, 1)).float()
                base_stds = torch.reshape(base_stds, (-1, self.seq_lene, 1)).float()
                base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_lene, 1)).float()
                if not self.no_seq:
                    y = torch.cat((y_kmere_embed, base_means, base_meadian, base_stds, base_signal_lens),2)
                else:
                    y = torch.cat((base_means, base_meadian, base_stds, base_signal_lens),2)
            
            if self.basecall and self.signalFea:
                y = torch.cat((y_kmere_embed, base_means, base_meadian, base_stds, base_signal_lens, qual, mis, ins, dele),2)
            
            y = torch.transpose(y, 1, 2)
            y_out = self.resnet(y)
            y_out = self.fc(y_out)
        
        if self.raw:
            signals = signals.float()
            signals_len = signals.shape[2]
            y_kmerr_embed = one_hot_embedding(kmerr.long(), signals_len) # torch.Size([N, 5, siglen ,4])
            signals_ex = signals.view(signals.shape[0], -1, 1)
            x = torch.cat((y_kmerr_embed, signals_ex), -1)
            # x = torch.transpose(x, 1, 2)
            x_out =self.bilstm(x)
            x_out = torch.transpose(x_out, 1, 2)
            x_out = self.cnn(x_out)
            x_out = self.fc(x_out)

        ##################### Full connect layer
        if self.raw and (self.basecall or self.signalFea):
            z= torch.cat((x_out, y_out), 1)
            z = self.fc(z)
            z = self.full_concat(z)
        elif self.raw: # only raw
            z = self.fc(x_out)
            z = self.full(z)
        else: # basecall or signalFea
            z = self.fc(y_out)
            z = self.full(z)

        ##################### sigmoid
        out_ = self.sigmoid(z)
        return out_