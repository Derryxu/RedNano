import argparse


def oversample_file(ifile, ofile, fold):
    with open(ifile, 'r') as rf:
        with open(ofile, 'w') as wf:
            for line in rf:
                for i in range(fold):
                    wf.write(line)
    # check
    print('done oversampling')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file, ori feature file", required=True)
    parser.add_argument("--output", help="output file, oversampled feature file", required=False)
    parser.add_argument("--fold", default=10, type=int, help="oversample folds, default 10", required=False)

    args = parser.parse_args()

    oversample_file(args.input, args.output, args.fold)


if __name__ == "__main__":
    main()
