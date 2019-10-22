#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pyknp import Juman
import mojimoji
import tqdm


def parse_argument():
    parser = argparse.ArgumentParser("", add_help=True)
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    if args.model:
        option = "--model=" + args.model
    else:
        option = None

    jumanpp = Juman(option=option)
    with open(args.input_file, "r") as r, \
            open(args.output_file, "w") as w:
        for row in tqdm.tqdm(r):
            row = row.strip()
            if not row:
                continue
            row = row.replace(" ", "")
            result = jumanpp.analysis(mojimoji.han_to_zen(row))
            w.write(" ".join([mrph.midasi for mrph in result.mrph_list()])
                    + "\n")


if __name__ == '__main__':
    main()
