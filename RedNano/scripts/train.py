import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import time
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.getcwd())
from model.model_simple import Model
from utils.MyDataSet_txt import *
from utils.hct_sample import *
from utils.pytorchtools import EarlyStopping

use_cuda = torch.cuda.is_available()


def loss_function():
    # return torch.nn.CrossEntropyLoss()
    return torch.nn.BCELoss()

def get_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc

def get_accuracy(y_true, y_pred):
    pred = y_pred.copy()
    for i in range(len(pred)):
        if pred[i] < 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    return accuracy_score(y_true, pred)

def train_epoch(model, train_dl, optimizer, loss_func, kmer_beg, kmer_end, clip_grad=None):
    print("train epoch")
    model.train()
    train_loss_list = []
    all_y_true = []
    all_y_pred = []
    loss_results = {}
    for batch_features_all in tqdm(train_dl):
        sampleinfo, kmer, base_means, base_median, base_stds, base_signal_lens, signals, qual, mis, ins, dele, labels = \
            batch_features_all[0], batch_features_all[1][:, kmer_beg:kmer_end].cuda(), batch_features_all[2][:, kmer_beg:kmer_end].cuda(), \
            batch_features_all[3][:,kmer_beg:kmer_end].cuda(), \
            batch_features_all[4][:, kmer_beg:kmer_end].cuda(), batch_features_all[5][:, kmer_beg:kmer_end].cuda(), \
            batch_features_all[6][:, kmer_beg:kmer_end, :].cuda(), batch_features_all[7][:, kmer_beg:kmer_end].cuda(), \
            batch_features_all[8][:, kmer_beg:kmer_end].cuda(), \
            batch_features_all[9][:, kmer_beg:kmer_end].cuda(), batch_features_all[10][:, kmer_beg:kmer_end].cuda(), batch_features_all[11].cuda()

        y_true = labels.flatten()
        y_pred = model(kmer, base_means, base_median, base_stds, base_signal_lens, signals, qual, mis, ins, dele)
        y_pred = y_pred.squeeze(-1)
        loss = loss_func(y_pred.to(torch.float), y_true.to(torch.float))  # sigmoid
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        train_loss_list.append(loss.item())
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        all_y_true.extend(y_true)
        if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
            all_y_pred.extend(y_pred.flatten())
        else:
            all_y_pred.extend(y_pred[:, 1])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    loss_results['avg_loss'] = np.mean(np.array(train_loss_list))
    loss_results['roc_auc'] = get_roc_auc(all_y_true, all_y_pred)
    loss_results['pr_auc'] = get_pr_auc(all_y_true, all_y_pred)
    loss_results['acc'] = get_accuracy(all_y_true, all_y_pred)
    return loss_results


def val_epoch(model, val_dl, loss_func, kmer_beg, kmer_end, n_iters=1):
    print("val epoch")
    model.eval()
    all_y_true = None
    all_y_pred = []
    val_loss_list = []

    with torch.no_grad():
        for _ in range(n_iters):
            y_true_tmp = []
            y_pred_tmp = []
            for batch_features_all in tqdm(val_dl):
                # for i, batch_features_all in enumerate(val_dl):
                sampleinfo, kmer, base_means, base_median, base_stds, base_signal_lens, signals, qual, mis, ins, dele, labels = \
                    batch_features_all[0], batch_features_all[1][:, kmer_beg:kmer_end].cuda(), batch_features_all[2][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[3][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[4][:, kmer_beg:kmer_end].cuda(), batch_features_all[5][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[6][:, kmer_beg:kmer_end, :].cuda(), batch_features_all[7][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[8][:, kmer_beg:kmer_end].cuda(), \
                    batch_features_all[9][:, kmer_beg:kmer_end].cuda(), batch_features_all[10][:, kmer_beg:kmer_end].cuda(), batch_features_all[
                        11].cuda()

                y_true = labels.flatten()
                y_pred = model(kmer, base_means, base_median, base_stds, base_signal_lens, signals, qual, mis, ins,
                               dele)
                y_pred = y_pred.squeeze(-1)
                loss = loss_func(y_pred.to(torch.float), y_true.to(torch.float))  # sigmoid
                val_loss_list.append(loss.item())

                y_true = y_true.cpu().numpy()
                y_pred = y_pred.cpu().numpy()

                if all_y_true is None:  # all_y_true只用记录一次就行
                    y_true_tmp.extend(y_true)

                if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
                    y_pred_tmp.extend(y_pred.flatten())
                else:
                    y_pred_tmp.extend(y_pred)

            if all_y_true is None:
                all_y_true = y_true_tmp
            all_y_pred.append(y_pred_tmp)

    all_y_pred = np.array(all_y_pred)
    all_y_true = np.array(all_y_true)
    y_pred_avg = np.mean(all_y_pred, axis=0)

    all_y_true = np.array(all_y_true).flatten()
    all_y_pred = np.array(all_y_pred).flatten()

    val_results = {}
    val_results['y_pred'] = y_pred_avg
    val_results['y_true'] = all_y_true
    val_results['roc_auc'] = get_roc_auc(all_y_true, y_pred_avg)
    val_results['pr_auc'] = get_pr_auc(all_y_true, y_pred_avg)

    y_pred_avg, all_y_true = np.array(y_pred_avg), np.array(all_y_true)
    val_results['avg_loss'] = np.mean(np.array(val_loss_list)) 
    val_results['acc'] = get_accuracy(all_y_true, y_pred_avg) 
    return val_results

def train(args):
    if use_cuda:
        device = 'cuda' if use_cuda else 'cpu'
        print("GPU is available!")
        print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("GPU is not available!")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('************* Model loader')
    model = Model(args.model_type, args.dropout_rate, args.hidden_size, args.rnn_hid, args.seq_lens, args.signal_lens,
                  args.embedding_size, args.rnn_n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = loss_function()

    if torch.cuda.device_count() > 1: # mul gpus
        model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()

    if args.resume and os.path.exists(args.resume):
        print('********** Resume model from '+ args.resume)
        ckpt = args.resume
        checkpoint = torch.load(ckpt)
        try:
            model.load_state_dict(checkpoint['net'])
        except RuntimeError:
            model.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    model_dir = os.path.abspath(args.save_dir).rstrip("/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=model_dir)

    print("********** data loader")
    if args.train_option==0:
        train_dataset = MyDataSetTxt(args.train_file)
        validate_dataset = MyDataSetTxt(args.valid_file)
    elif args.train_option==1:
        exit()
        # hdf5 file ********
    elif args.train_option==2 and args.sample=="no_sample":
        train_filelist, val_filelist = no_sample_txt(args.train_val_pos_file, args.train_val_neg_file)
        train_dataset = MyDataSetTxt(train_filelist)
        validate_dataset = MyDataSetTxt(val_filelist)
    elif args.train_option==2 and args.sample=='unm_undersample':
        train_filelist, val_filelist = undersample_txt(args.train_val_pos_file, args.train_val_neg_file)
        train_dataset = MyDataSetTxt(train_filelist)
        validate_dataset = MyDataSetTxt(val_filelist)
    else:
        return False


    train_dl = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                           prefetch_factor=2)
    val_dl = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         prefetch_factor=2)

    train_results = {}
    val_results = {}
    print("*********** Model Train ... ")
    kmer_beg, kmer_end = 5 - args.seq_lens//2, 5 + args.seq_lens//2 + 1
    for epoch in range(args.epochs):
        # trian beginning
        train_results_epoch = train_epoch(model, train_dl, optimizer, loss_func, kmer_beg, kmer_end, args.clip_grad)
        # val beginning
        val_results_epoch = val_epoch(model, val_dl, loss_func, kmer_beg, kmer_end, 1)

        print("Epoch:[{epoch}/{n_epoch}] \t ".format(epoch=epoch, n_epoch=args.epochs))
        print("Train Loss:{loss:.2f}\t "
              "Train ROC AUC: {roc_auc:.3f}\t "
              "Train PR AUC: {pr_auc:.3f}\t "
              "Train ACC: {acc:.3f}".format(loss=train_results_epoch["avg_loss"],
                                            roc_auc=train_results_epoch["roc_auc"],
                                            pr_auc=train_results_epoch["pr_auc"],
                                            acc=train_results_epoch["acc"], ))

        print("Val Loss:{loss:.2f} \t "
              "Val ROC AUC: {roc_auc:.3f}\t "
              "Val PR AUC: {pr_auc:.3f}\t"
              "Val ACC: {acc:.3f}".format(loss=val_results_epoch["avg_loss"],
                                          roc_auc=val_results_epoch["roc_auc"],
                                          pr_auc=val_results_epoch["pr_auc"],
                                          acc=val_results_epoch["acc"]))

        # early-stopping
        early_stopping(val_results_epoch['avg_loss'], model, optimizer)
        # 若满足 early stopping 要求
        # if early_stopping.early_stop:
        #     print("Early stopping, epoch : ", epoch, "best----epoch=", curr_best_epoch, " -----acc= ", curr_val_bestAcc,
        #           "  -----loss= ", curr_val_min_loss)
        #     # 结束模型训练
        #     break
    linecache.clearcache()

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    
    parser.add_argument("--train_file", default=None, required=False)
    parser.add_argument("--valid_file", default=None, required=False)

    parser.add_argument("--train_hdf5_dir", default=None, required=False)
    parser.add_argument("--valide_hdf5_dir", default=None, required=False)

    parser.add_argument("--train_val_pos_file", default=None, required=False)
    parser.add_argument("--train_val_neg_file", default=None, required=False)
    
    parser.add_argument("--tratio_train_val", default=0.75, type=float, required=False, help='input dataset option: [0] --train_file --valide_hdf5_dir  [1] --train_hdf5_dir --  [2] --train_val_pos_file --train_val_neg_file')
    parser.add_argument("--train_option", default=0, type=int, required=True, help='input dataset option: [0] --train_file --valide_hdf5_dir  [1] --train_hdf5_dir --  [2] --train_val_pos_file --train_val_neg_file')

    parser.add_argument("--sample", type=str, help="sample [no_sample for option[0,1], unm_undersample for option[2]]")

    parser.add_argument("--save_dir", default=None, required=True, help='model.pt save dir')

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--weight_decay", dest="weight_decay", default=1e-5, type=float)
    parser.add_argument("--patience", default=5, type=int, help="early-stopping patience")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--clip_grad", default=0.5, type=float)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--rnn_hid", default=128, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--seq_lens", default=5, type=int)
    parser.add_argument("--signal_lens", default=65, type=int)
    parser.add_argument("--embedding_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--resume", type=str, help="train resume file: smodel.pt ")
    parser.add_argument("--model_type", default='comb_basecall_raw', type=str, 
                        choices=["basecall", "signalFea", "raw_signal", "comb_basecall_signalFea","comb_basecall_raw","comb_signalFea_raw","comb_basecall_signalFea_raw"],
                        required=False, help="module for train:[basecall, signalFea, raw_signal, comb_basecall_signalFea, comb_basecall_raw, comb_signalFea_raw, comb_basecall_signalFea_raw]")
    return parser


def main():
    args = argparser().parse_args()
    for arg in vars(args):
        print(arg, ': ', getattr(args, arg))  # getattr() 函数是获取args中arg的属性值
    train(args)


if __name__ == '__main__':
    main()
