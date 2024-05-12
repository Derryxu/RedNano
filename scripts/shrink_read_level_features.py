import random
import argparse
import numpy as np


def shrink_file(input_file, output_file, kept_num, min_num, random_seed):
    assert kept_num > min_num
    np.random.seed(random_seed)
    wf = open(output_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            words = line.strip().split('\t')
            coverage = int(words[3])
            if coverage < min_num:
                kept_indices = np.random.choice(coverage, min_num-coverage, replace=False)
                kept_indices.sort()
                new_line = '\t'.join([words[0], words[1], words[2], str(min_num), words[4]])
                for i in range(coverage):
                    new_line += '\t' + words[5 + i]
                for i in kept_indices:
                    new_line += '\t' + words[5 + i]
                new_line += '\t' + words[-1]
                wf.write(new_line + '\n')
            elif coverage <= kept_num:
                wf.write(line)
            else:
                kept_indices = np.random.choice(coverage, kept_num, replace=False)
                kept_indices.sort()
                new_line = '\t'.join([words[0], words[1], words[2], str(kept_num), words[4]])
                for i in kept_indices:
                    new_line += '\t' + words[5 + i]
                new_line += '\t' + words[-1]
                wf.write(new_line + '\n')
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--kept_num', type=int, default=50)
    parser.add_argument('--min_num', type=int, default=20)
    args = parser.parse_args()

    shrink_file(args.input, args.output, args.kept_num, args.min_num, args.seed)


if __name__ == '__main__':
    main()
