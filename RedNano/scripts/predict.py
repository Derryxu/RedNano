import torch, os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import sys
sys.path.append(os.getcwd())

# from model.train_model_resnet_all import Model
# from model.train_model_res_5mer import Model
from model.model_simple import Model

#from model.model_hdf5 import Model


from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from utils.MyDataSet_txt_test import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--model", default=None, required=True, help="model file for test")
    
    parser.add_argument("--test_option", default=0, type=int, required=True, help='input dataset option: [0] --test_file  [1] --test_file_dir --  [2] --test_hdf5_dir ')
    
    parser.add_argument("--test_file", default=None, required=False, help="features file")
    parser.add_argument("--test_file_dir", default=None, required=False, help="features files dir")
    parser.add_argument("--test_hdf5_dir", default=None, required=False, help="features hdf5 files dir")
    
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--num_iterations", default=1, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--rnn_hid", default=512, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)

    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--seq_lens", default=11, type=int)
    parser.add_argument("--signal_lens", default=65, type=int)
    parser.add_argument("--embedding_size", default=4, type=int)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--model_type", default='comb_basecall_raw', type=str, 
                        choices=["basecall", "signalFea", "raw_signal", "comb_basecall_signalFea","comb_basecall_raw","comb_signalFea_raw","comb_basecall_signalFea_raw"],
                        required=False, help="module for train:[basecall, signalFea, raw_signal, comb_basecall_signalFea, comb_basecall_raw, comb_signalFea_raw, comb_basecall_signalFea_raw]")
                        
    parser.add_argument("--output_file_dir", default='./', type=str, help="floder for output]")
    parser.add_argument("--num_workers", default=2, type=int)

    return parser

def loss_function():
    return torch.nn.MSELoss()

def test_epoch(model, test_dl, kmer_beg, kmer_end, n_iters=1):
    print("test epoch")
    model.eval()
    all_y_true = None
    all_y_pred = []
    sampleinfo_ = []
    kmer_ = []

    with torch.no_grad():
        #for i, batch_features_all in enumerate(test_dl):
        for batch_features_all in tqdm(test_dl):
            sampleinfo, kmer, base_means, base_median, base_stds, base_signal_lens, signals, qual, mis, ins, dele = \
                batch_features_all[0], batch_features_all[1][:, kmer_beg:kmer_end].cuda(), batch_features_all[2][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[3][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[4][:, kmer_beg:kmer_end].cuda(), batch_features_all[5][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[6][:, kmer_beg:kmer_end, :].cuda(), batch_features_all[7][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[8][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[9][:, kmer_beg:kmer_end].cuda(), batch_features_all[10][:, kmer_beg:kmer_end].cuda()

            y_pred  = model(kmer, base_means, base_median, base_stds, base_signal_lens, signals, qual, mis, ins, dele)

            y_pred = y_pred.cpu().numpy()
            sampleinfo_.extend(sampleinfo)
            kmer_.extend(kmer.cpu().numpy())

            if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
                all_y_pred.extend(y_pred.flatten())
            else:
                all_y_pred.extend(y_pred[:, 1])
        # sampleinfo
    all_y_pred = np.array(all_y_pred).flatten()
    test_results = {}
    test_results['sampleinfo'] = sampleinfo_
    test_results['kmer'] = kmer_
    test_results['y_pred'] = all_y_pred

    return test_results


def split_sampleinfo(sampleinfo):
    infos = sampleinfo.split("\t")
    return infos

code2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def split_kmer(kmer):
    strs = "".join([code2base[x] for x in kmer])
    return strs
    
def listdir(path):
    file_list = []
    for file in os.listdir(path):
        file_list.append(os.path.join(path, file))
    return file_list


def test(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_iterations = args.num_iterations
    batch_size = args.batch_size

    model_path = args.model
    model = Model(args.model_type, args.dropout_rate, args.hidden_size, args.rnn_hid,  args.seq_lens, args.signal_lens, args.embedding_size, args.rnn_n_layers)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])

    if torch.cuda.device_count() > 1:
        print(device, "Use", torch.cuda.device_count(), 'gpus')
    model.to(device)
    print(model)

    print("********** data loader")
    if args.test_option==0:
        file_list = [args.test_file]
        print(len(file_list))
    elif args.test_option==1:
        file_list = listdir(args.test_file_dir)
    

    print("*********** Model Test ... ")
    kmer_beg, kmer_end = 5 - args.seq_lens//2, 5 + args.seq_lens//2 + 1
    for file in tqdm(file_list):
        test_dataset = MyDataSetTxt(file)
        test_dl = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        test_results_epoch = test_epoch(model, test_dl, kmer_beg, kmer_end, 1)

        # 将sampleinfo保存在文件中
        sampleinfo = test_results_epoch['sampleinfo']
        output_data = pd.DataFrame([split_sampleinfo(sa) for sa in sampleinfo])
        output_data.columns = ['chrom', 'pos', 'alignstrand', 'loc_in_re', 'readname', 'strand']
        output_data['kmer'] = [split_kmer(kmer) for kmer in test_results_epoch['kmer']]
        output_data['pred'] = list(test_results_epoch['y_pred'])
        wf_file = os.path.join(args.output_file_dir, os.path.splitext(os.path.basename(file))[0] + ".reads.output.txt")
        output_data.to_csv(wf_file, sep='\t', index=False) 

def main():
    args = argparser().parse_args()
    test(args)

if __name__ == '__main__':
    main()
