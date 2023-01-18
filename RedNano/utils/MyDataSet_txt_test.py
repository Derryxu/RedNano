from torch.utils.data import Dataset
import linecache
import os
import numpy as np

base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
code2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def signal_features_line(line):
    words = line.strip().split("\t")
    sampleinfo = "\t".join(words[0:6])
    kmers = np.array([base2code[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_median = np.array([float(x) for x in words[8].split(",")])
    base_stds = np.array([float(x) for x in words[9].split(",")])
    base_signal_lens = np.array([int(x) for x in words[10].split(",")])
    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[11].split(";")])
    qual = np.array([int(x) for x in words[12].split(",")])
    mis = np.array([int(x) for x in words[13].split(",")])
    ins = np.array([int(x) for x in words[14].split(",")])
    dele = np.array([int(x) for x in words[15].split(",")])
    return sampleinfo, kmers, base_means, base_median, base_stds, base_signal_lens, k_signals, qual, mis, ins, dele


class MyDataSetTxt(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        # self.max_subreads = max_subreads
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = signal_features_line(line)
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
    def __init__(self, filename, offsets, linenum, transform=None):
        self._filename = os.path.abspath(filename)
        self._total_data = linenum
        self._transform = transform

        self._offsets = offsets
        self._data_stream = open(self._filename, 'r')
        self._current_offset = 0

    def __getitem__(self, idx):
        offset = self._offsets[idx]
        self._data_stream.seek(offset)
        line = self._data_stream.readline()
        # with open(self._filename, "r") as rf:
        #     rf.seek(offset)
        #     line = rf.readline()
        output = signal_features_line(line)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data

    def close(self):
        self._data_stream.close()
