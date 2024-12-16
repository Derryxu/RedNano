#! /usr/bin/env python
import argparse
import os


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def read_position_file(positionfp, col_chrom=0, col_pos=1, one_based=False, header=False):
    posstrs = set()
    with open(positionfp, 'r') as rf:
        if header:
            next(rf)
        for line in rf:
            words = line.strip().split("\t")
            if one_based:
                try:
                    posstrs.add(' '.join([words[col_chrom], str(int(words[col_pos]) - 1)]))
                except ValueError:
                    continue
            else:
                posstrs.add(' '.join([words[col_chrom], words[col_pos]]))
    return posstrs


def filter_one_signal_feature_file(sf_fp, positions, wfp, label, chrom_col=1, pos_col=2, ext_label=False, not_in=False):
    wf = open(wfp, 'w')
    with open(sf_fp, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            chromsome, loc = words[chrom_col-1], int(words[pos_col-1])
            posstr_tmp = ' '.join(list(map(str, [chromsome, loc])))
            if (not_in and posstr_tmp not in positions) or (not not_in and posstr_tmp in positions):
                if ext_label:
                    wf.write("\t".join(words + [label]) + "\n")
                else:
                    wf.write("\t".join(words[:-1] + [label]) + "\n")
    wf.close()


def filter_one_signal_feature_file_append(sf_fp, positions, wfp, label, chrom_col=1, pos_col=2, ext_label=False, not_in=False):
    wf = open(wfp, 'a')
    with open(sf_fp, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            chromsome, loc = words[chrom_col], int(words[pos_col])
            posstr_tmp = ' '.join(list(map(str, [chromsome, loc])))
            if (not_in and posstr_tmp not in positions) or (not not_in and posstr_tmp in positions):
                if ext_label:
                    wf.write("\t".join(words + [label]) + "\n")
                else:
                    wf.write("\t".join(words[:-1] + [label]) + "\n")
    wf.close()
# ==========================================================================


def main():
    parser = argparse.ArgumentParser(description='extract samples with interested ref_positions '
                                                 'from signal feature file')
    parser.add_argument('--sf_path', type=str, required=True,
                        help='the sample signal_features_file path needed to be filtered, or a directory')
    parser.add_argument('--unique_fid', type=str, required=False,
                        default='.tsv',
                        help='unique str of all to be processed files')
    parser.add_argument('-p', "--pos_fp",
                        help="the directory of position file, per line: chromosome\tpos_in_forward_strand",
                        type=str, required=True)
    parser.add_argument('--midfix', type=str, required=False,
                        default="filtered",
                        help='a str to id the output file')
    parser.add_argument('--label', type=str, required=False,
                        default="1", choices=["0", "1"],
                        help="methy label of the extracted samples")
    parser.add_argument('--ext_label', action='store_true', default=False, 
                        help="extract the label from the last column of the signal feature file")

    parser.add_argument('--chrom_col', type=int, default=0,
                        required=False, help="the col of the chroms in the signal feature file, "
                                             "default 0")
    parser.add_argument('--pos_col', type=int, default=1,
                        required=False, help="the col of the poses in the signal feature file, "
                                             "default 1")
    parser.add_argument('--p_chrom_col', type=int, default=0, required=False,
                        help="the col of the chroms in the position file, default 0")
    parser.add_argument('--p_pos_col', type=int, default=1, required=False,
                        help="the col of the poses in the position file, default 1")
    parser.add_argument('--p_one_based', action='store_true', default=False,
                        help="whether the position file is one_based or not, default False")
    parser.add_argument('--not_in', action='store_true', default=False,
                        help="extract the samples not in the position file")

    args = parser.parse_args()
    sf_fp = args.sf_path  # signal features
    unique_fid = args.unique_fid
    positionfp = args.pos_fp  # position file
    midfix = args.midfix
    label = args.label
    ext_label = args.ext_label
    not_in = args.not_in

    chrom_col = args.chrom_col
    pos_col = args.pos_col
    p_chrom_col = args.p_chrom_col
    p_pos_col = args.p_pos_col
    p_one_based = args.p_one_based

    positions = read_position_file(positionfp, p_chrom_col, p_pos_col, p_one_based)
    print('there are {} positions to be chosen'.format(len(positions)))
    if os.path.isdir(sf_fp):
        wfp = sf_fp.strip('/') + '.' + midfix + '.tsv'
        wf = open(wfp, 'w')
        wf.close()
        for sfile in os.listdir(sf_fp):
            if sfile.find(unique_fid) != -1:
                read_file = sf_fp + '/' + sfile
                filter_one_signal_feature_file_append(read_file, positions, wfp, label,
                                                      chrom_col, pos_col, ext_label, not_in)
                print('done with file {}'.format(read_file))
    else:
        fname, fext = os.path.splitext(sf_fp)
        wfp = fname + '.' + midfix + fext
        filter_one_signal_feature_file(sf_fp, positions, wfp, label, chrom_col, pos_col, ext_label, not_in)


if __name__ == '__main__':
    main()
