#!/usr/bin/python
import os
import argparse
from nltk.stem.wordnet import WordNetLemmatizer
import lemmatize_snts

def main(args):
    dep_file = os.path.join(args.input_dir, "dep")
    token_wf = open(os.path.join(args.input_dir, 'token'), 'w')
    pos_wf = open(os.path.join(args.input_dir, 'pos'), 'w')
    lemma_wf = open(os.path.join(args.input_dir, 'lemma'), 'w')
    toks = []
    poss = []

    special_symbols = set()
    special_token_map = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}

    der_lemma = lemmatize_snts.initialize_lemma(args.lemma_file)

    lmtzer = WordNetLemmatizer()
    for line in open(dep_file):
        fields = line.strip().split()
        if len(fields) < 2: #A new sent
            lemmas = lemmatize_snts.lemmatize_sentence(toks, poss, der_lemma, lmtzer)
            assert len(toks) == len(lemmas)
            print >> token_wf, (' '.join(toks))
            print >> pos_wf, (' '.join(poss))
            print >> lemma_wf, (' '.join(lemmas))
            toks = []
            poss = []
            continue

        curr_tok = fields[1].strip()

        for sp in special_token_map:
            if sp in curr_tok:
                curr_tok = curr_tok.replace(sp, special_token_map[sp])

        toks.append(curr_tok)
        if curr_tok[0] == '-' and curr_tok[-1] == '-':
            if len(curr_tok) > 2:
                special_symbols.add(curr_tok)

        poss.append(fields[4].strip())

    token_wf.close()
    pos_wf.close()
    print special_symbols


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--input_dir", type=str, help="the input directory, containing token and pos tags")
    argparser.add_argument("--lemma_file", type=str, help="celex lemma file", required=False)

    args = argparser.parse_args()
    main(args)

