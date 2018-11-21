#!/usr/bin/python
import os
from nltk.stem.wordnet import WordNetLemmatizer
import argparse
from nltk.corpus import wordnet as wn
from collections import defaultdict
def initialize_lemma(lemma_file):
    lemma_map = defaultdict(set)
    with open(lemma_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            word = fields[0]
            lemma = fields[1]
            if word == lemma:
                continue
            lemma_map[word].add(lemma)
    return lemma_map

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

#Use the lemma dict to find the lemma of the word
def get_celex_lemma(word, pos, der_lemma=None):
    if is_adverb(pos) and der_lemma is not None and word in der_lemma:
        return list(der_lemma[word])[0]
    return word

def get_wordnet_lemma(word, pos, lmtzer=WordNetLemmatizer()):
    try:
        if is_noun(pos):
            return lmtzer.lemmatize(word, wn.NOUN)
        if is_verb(pos):
            return lmtzer.lemmatize(word, wn.VERB)
        if is_adjective(pos):
            return lmtzer.lemmatize(word, wn.ADJ)
    except:
        print word
        return None
    return None

def lemmatize_sentence(toks, poss, der_lemma=None, lmtzer=WordNetLemmatizer()):
    assert len(toks) == len(poss)

    lemmas = []
    for (word, pos) in zip(toks, poss):
        wn_lem = get_wordnet_lemma(word, pos, lmtzer)
        if wn_lem:
            lemmas.append(wn_lem.encode("ascii"))
        else:
            celex_lem = get_celex_lemma(word, pos, der_lemma)
            lemmas.append(celex_lem)
    return lemmas

def main(args):
    der_file = os.path.join(args.lemma_dir, 'der.lemma')
    der_lemma = initialize_lemma(der_file)

    tok_file = os.path.join(args.input_dir, "token")
    pos_file = os.path.join(args.input_dir, "pos")

    tok_seqs = [line.strip().split() for line in open(tok_file, 'r')]
    pos_seqs = [line.strip().split() for line in open(pos_file, 'r')]

    result_file = os.path.join(args.input_dir, 'lemma')
    lmtzer = WordNetLemmatizer()
    with open(result_file, 'w') as wf:
        for (toks, poss) in zip(tok_seqs, pos_seqs):
            assert len(toks) == len(poss)

            lemmas = []
            for (word, pos) in zip(toks, poss):
                wn_lem = get_wordnet_lemma(word, pos, lmtzer)
                if wn_lem:
                    lemmas.append(wn_lem.encode("ascii"))
                else:
                    celex_lem = get_celex_lemma(word, pos, der_lemma)
                    lemmas.append(celex_lem)

                print >>wf, ' '.join(lemmas)
        wf.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--input_dir", type=str, help="the input directory, containing token and pos tags")
    argparser.add_argument("--lemma_dir", type=str, help="celex directory", required=False)

    args = argparser.parse_args()
    main(args)
