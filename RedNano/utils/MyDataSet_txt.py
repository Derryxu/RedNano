from torch.utils.data import Dataset
import linecache
import os
import numpy as np

from utils.constants import base2code, min_cov, max_seq_len


def _parse_one_read_level_feature(feature, kmerb, kmere, kmer):
    words = feature.strip().split("||")
    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[4].split(";")])[kmerb:kmere]
    qual = np.array([int(x) for x in words[5].split(",")])[kmerb:kmere]
    mis = np.array([int(x) for x in words[6].split(",")])[kmerb:kmere]
    ins = np.array([int(x) for x in words[7].split(",")])[kmerb:kmere]
    dele = np.array([int(x) for x in words[8].split(",")])[kmerb:kmere]
    # assert len(qual) == len(mis) == len(ins) == len(dele) == len(kmer)
    return np.concatenate((kmer.reshape(-1, len(kmer)), qual.reshape(-1, len(kmer)), 
                           mis.reshape(-1, len(kmer)), ins.reshape(-1, len(kmer)), 
                           dele.reshape(-1, len(kmer)), k_signals.transpose()), 
                           axis=0)


def generate_features_line(line, kmerb, kmere, sampleing=True):
    words = line.strip().split("\t")
    sampleinfo = "\t".join(words[0:3])
    coverage = int(words[3])
    kmer = np.array([base2code[x] for x in words[4]])[kmerb:kmere]
    features = np.array(words[5:-1])
    assert len(features) == coverage
    if sampleing:
        features = features[np.random.choice(coverage, min_cov, replace=False)]
    features = np.array([_parse_one_read_level_feature(x, kmerb, kmere, kmer) for x in features])
    label = int(words[-1])
    return sampleinfo, features, label


class MyDataSetTxt(Dataset):
    def __init__(self, filename, seq_len=5, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self.seq_len=seq_len
        center_pos = max_seq_len // 2
        self.kmerb, self.kmere = center_pos - self.seq_len//2, center_pos + self.seq_len//2 + 1
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = generate_features_line(line, self.kmerb, self.kmere)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass


# FeaData2 ======================================================
# ChunkDataset hasn't being accepted
# https://github.com/pytorch/pytorch/pull/26547

# https://github.com/pytorch/text/issues/130
# https://github.com/pytorch/text/blob/0b4718d7827b7f278cd3169af7f2587c1f663a27/torchtext/datasets/unsupervised_learning.py
def generate_offsets(filename):
    offsets = []
    with open(filename, "r") as rf:
        offsets.append(rf.tell())
        while rf.readline():
            offsets.append(rf.tell())
    return offsets


class MyDataSetTxt2(Dataset):
    def __init__(self, filename, offsets, linenum, seq_len=5, transform=None):
        self._filename = os.path.abspath(filename)
        self._total_data = linenum

        self.seq_len=seq_len
        center_pos = max_seq_len // 2
        self.kmerb, self.kmere = center_pos - self.seq_len//2, center_pos + self.seq_len//2 + 1
        
        self._transform = transform

        self._offsets = offsets
        # self._data_stream = open(self._filename, 'r')
        self._current_offset = 0

    def __getitem__(self, idx):
        offset = self._offsets[idx]
        # self._data_stream.seek(offset)
        # line = self._data_stream.readline()
        with open(self._filename, "r") as rf:
            rf.seek(offset)
            line = rf.readline()
        output = generate_features_line(line, self.kmerb, self.kmere)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data

    def close(self):
        # self._data_stream.close()
        pass
