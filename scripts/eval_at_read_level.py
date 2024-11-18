#! /usr/bin/python
"""
evaluate in read level
"""
import argparse
import os
import random
from collections import namedtuple
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product


# num_sites = [10000, 100000, 200000, 1000000000]
num_sites = [100000, 200000, 1000000000]
CallRecord = namedtuple('CallRecord', ['chrom', 'pos', 'strand',
                                       'read_name', 
                                       'kmer', 'prob1',
                                       'predicted_label',
                                       'is_true_methylated'])


def get_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def get_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc


def sample_sites(filename, is_methylated, probcf, sampleids=None, motifs=None, rm_amiguous=False):
    all_crs = list()
    rf = open(filename)
    next(rf)
    skip_cnt = 0
    cnt = 0
    repeat_cnt = 0
    read_sampleids = set()
    for line in rf:
        cnt += 1
        words = line.strip().split("\t")
        
        kmer = words[6]
        if motifs is not None and kmer not in motifs:
            skip_cnt += 1
            continue

        readid = words[4]
        chrom = words[0]
        pos = int(words[1])
        sampid = "\t".join([chrom, str(pos), readid])  # chrom, pos, holeid
        if sampleids is not None:
            if sampid not in sampleids:
                skip_cnt += 1
                continue
        if sampid in read_sampleids:
            repeat_cnt += 1
            continue

        prob1 = float(words[7])
        if not rm_amiguous:
            label = 1 if prob1 >= probcf else 0
        else:
            if prob1 < probcf and 1 - prob1 < probcf:
                skip_cnt += 1
                continue
            label = 1 if prob1 >= 0.5 else 0
        
        read_sampleids.add(sampid)
        all_crs.append(CallRecord(chrom, pos, words[2], 
                                  words[4], kmer, prob1,
                                  label,
                                  is_methylated))
    sys.stderr.write('there are {} cpg candidates totally, {} cpgs kept, {} cpgs left, '
                     '{} cpgs repeat\n'.format(cnt, len(all_crs), skip_cnt, repeat_cnt))
    rf.close()
    random.shuffle(all_crs)
    return all_crs


def sample_sites_nanom6a(filename, is_methylated, probcf):
    all_crs = list()
    rf = open(filename)
    skip_cnt = 0
    cnt = 0
    repeat_cnt = 0
    read_sampleids = set()
    for line in rf:
        cnt += 1
        words = line.strip().split("\t")

        prob1 = float(words[1])
        label = 1 if prob1 >= probcf else 0
        
        readid = words[2].split("|")[0].split(".")[0]
        all_crs.append(CallRecord("chrom", -1, "strand", 
                                  readid, "kmer", prob1,
                                  label,
                                  is_methylated))
    sys.stderr.write('there are {} cpg candidates totally, {} cpgs kept, {} cpgs left, '
                     '{} cpgs repeat\n'.format(cnt, len(all_crs), skip_cnt, repeat_cnt))
    rf.close()
    random.shuffle(all_crs)
    return all_crs


def get_sampleids(sampleids_file):
    sampleids = set()
    with open(sampleids_file, "r") as rf:
        for line in rf:
            if not line.startswith("#"):
                sampleids.add(line.strip())
    return sampleids


def _parse_motif(motif_str):
    """
    use iupac tables to parse motif to multiple motifs
    """
    if motif_str is None:
        return None
    motif_str = motif_str.upper()
    if motif_str == "RRACH":
        center_motifs = [['A', 'G'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
        return set(["".join(x) for x in product(*(center_motifs))])
    elif motif_str == "DRACH":
        center_motifs = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
        return set(["".join(x) for x in product(*(center_motifs))])
    else:
        raise ValueError("unknown motif: {}".format(motif_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read-level evaluation')
    parser.add_argument('--unmethylated', type=str, required=True)
    parser.add_argument('--methylated', type=str, required=True)
    parser.add_argument('--prob_cf', required=False, type=float, default=0.5, 
                        help='prob cutoff to judge if a site is methylated, default 0.5. range [0, 1].')
    parser.add_argument('--round', type=int, default=5, required=False,
                        help="number of repeated tests for random sampling")
    parser.add_argument('--result_file', type=str, required=False, default=None, 
                        help='the result file are going to save')
    parser.add_argument('--sampleids_file_u', type=str, default=None, required=False,
                        help='the file contains unmethylated ids of sites to be tested')
    parser.add_argument('--sampleids_file_m', type=str, default=None, required=False,
                        help='the file contains methylated ids of sites to be tested')
    parser.add_argument('--seed', type=int, default=1234, help="seed")
    parser.add_argument('--motif', type=str, required=False, default=None,
                        help='the motif of sites to be tested, RRACH, DRACH, or None')
    parser.add_argument('--nanom6a', action="store_true", required=False, help="nanom6a read-level input")
    parser.add_argument('--rm_amiguous', action="store_true", required=False, help="rm ambiguous sites")

    args = parser.parse_args()

    random.seed(args.seed)
    prob_cf = args.prob_cf
    motifs = _parse_motif(args.motif)

    sample_ids_u = get_sampleids(args.sampleids_file_u) if args.sampleids_file_u is not None else None
    sample_ids_m = get_sampleids(args.sampleids_file_m) if args.sampleids_file_m is not None else None

    result_file = os.path.abspath(args.result_file) if args.result_file is not None else None
    pr_writer = open(result_file, 'w') if result_file is not None else sys.stdout
    pr_writer.write("# motif: {}\n".format(args.motif))
    pr_writer.write("tested_type\tTP\tFN\tTN\tFP\t"
                    "accuracy\trecall\tspecificity\tprecision\t"
                    "fallout\tmiss_rate\tFDR\tNPV\tAUC\tAUPR\tsamplenum\tprob_cf\tnum_rounds\n")
    
    if args.nanom6a:
        unmethylated_sites = sample_sites_nanom6a(args.unmethylated, False, float(prob_cf))
        methylated_sites = sample_sites_nanom6a(args.methylated, True, float(prob_cf))
    else: # default 
        unmethylated_sites = sample_sites(args.unmethylated, False, float(prob_cf), sample_ids_u, motifs, args.rm_amiguous)
        methylated_sites = sample_sites(args.methylated, True, float(prob_cf), sample_ids_m, motifs, args.rm_amiguous)

    for site_num in num_sites:
        num_rounds = args.round
        if site_num >= len(methylated_sites) and site_num >= len(unmethylated_sites):
            num_rounds = 1
        metrics = []
        for roundidx in range(num_rounds):
            random.shuffle(methylated_sites)
            random.shuffle(unmethylated_sites)
            tested_sites = methylated_sites[:site_num] + unmethylated_sites[:site_num]

            tp = 0
            fp = 0
            tn = 0
            fn = 0

            called = 0
            correct = 0

            y_truelabel = []
            y_scores = []

            for s in tested_sites:
                tp += s.predicted_label and s.is_true_methylated
                fp += s.predicted_label and not s.is_true_methylated
                tn += not s.predicted_label and not s.is_true_methylated
                fn += not s.predicted_label and s.is_true_methylated

                y_truelabel.append(s.is_true_methylated)
                y_scores.append(s.prob1)

            sys.stderr.write("{}, {}, {}, {}\n".format(tp, fn, tn, fp))
            precision, recall, specificity, accuracy = 0, 0, 0, 0
            fall_out, miss_rate, fdr, npv, = 0, 0, 0, 0
            auroc = 0
            aupr = 0
            if len(tested_sites) > 0:
                accuracy = float(tp + tn) / len(tested_sites)
                if tp + fp > 0:
                    precision = float(tp) / (tp + fp)
                    fdr = float(fp) / (tp + fp)  # false discovery rate
                else:
                    precision = 0
                    fdr = 0
                if tp + fn > 0:
                    recall = float(tp) / (tp + fn)
                    miss_rate = float(fn) / (tp + fn)  # false negative rate
                else:
                    recall = 0
                    miss_rate = 0
                if tn + fp > 0:
                    specificity = float(tn) / (tn + fp)
                    fall_out = float(fp) / (fp + tn)  # false positive rate
                else:
                    specificity = 0
                    fall_out = 0
                if tn + fn > 0:
                    npv = float(tn) / (tn + fn)  # negative predictive value
                else:
                    npv = 0
                # auroc = roc_auc_score(np.array(y_truelabel), np.array(y_scores))
                auroc = get_roc_auc(np.array(y_truelabel), np.array(y_scores))  # same as above
                aupr = get_pr_auc(np.array(y_truelabel), np.array(y_scores))
            metrics.append([tp, fn, tn, fp, accuracy, recall, specificity, precision,
                            fall_out, miss_rate, fdr, npv, auroc, aupr, len(tested_sites)])
        sys.stderr.write("\n")
        # cal mean
        metrics = np.array(metrics, dtype=float)

        metrics_mean = np.mean(metrics, 0)
        mean_tpfntnfp = "\t".join([str(round(x, 1)) for x in metrics_mean[:4]])
        mean_perf = "\t".join([str(round(x, 4)) for x in metrics_mean[4:14]])
        mean_numlen = str(round(metrics_mean[14]))
        pr_writer.write("\t".join([str(site_num), mean_tpfntnfp, mean_perf, mean_numlen,
                                    str(prob_cf), str(num_rounds)]) + "\n")

        metrics_std = np.std(metrics, 0)
        std_tpfntnfp = "\t".join([str(round(x, 1)) for x in metrics_std[:4]])
        std_perf = "\t".join([str(round(x, 4)) for x in metrics_std[4:14]])
        std_numlen = str(round(metrics_std[14]))
        pr_writer.write("\t".join([str(site_num) + "_std", std_tpfntnfp, std_perf, std_numlen,
                                    str(prob_cf), str(num_rounds)]) + "\n")
        pr_writer.flush()

    pr_writer.close()
