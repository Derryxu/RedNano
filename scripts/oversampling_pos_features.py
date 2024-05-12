import argparse
import numpy as np


def count_line_num(sl_filepath, fheader=False, printlog=True):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    if printlog:
        print('done count the lines of file {}'.format(sl_filepath))
    return count


def oversample_file(ifile, ofile, vfile):
    ifile_lines = count_line_num(ifile, fheader=False)
    vfile_lines = count_line_num(vfile, fheader=False)
    if ifile_lines >= vfile_lines:
        print('no need to oversample')
    else:
        idx_oversample = np.random.choice(np.arange(ifile_lines), vfile_lines - ifile_lines, replace=True)
        idx_oversample = np.sort(idx_oversample)
        count = 0
        idx_idx = 0
        with open(ifile, 'r') as rf:
            with open(ofile, 'w') as wf:
                for line in rf:
                    wf.write(line)
                    for idx in idx_oversample[idx_idx:]:
                        if count == idx:
                            wf.write(line)
                        else:
                            break
                        idx_idx += 1
                    count += 1
        # check
        assert count_line_num(ofile, fheader=False) == vfile_lines
        print('done oversampling')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file, ori feature file", required=True)
    parser.add_argument("--output", help="output file, oversampled feature file", required=False)
    parser.add_argument("--vfile", help="file that the input file should be oversampled to match", required=True)

    args = parser.parse_args()

    oversample_file(args.input, args.output, args.vfile)


if __name__ == "__main__":
    main()
