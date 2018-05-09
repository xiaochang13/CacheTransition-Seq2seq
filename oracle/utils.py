#!/usr/bin/python
import re
import sys
from collections import defaultdict
from enum import Enum
class Tokentype(Enum):
    WORD = 1
    LEM = 2
    POS = 3
    CONCEPT = 4
    DEP = 5
    ARC = 6
    CATEGORY = 7

class OracleType(Enum):
    AAAI = 1
    CL = 2

class FeatureType(Enum):
    SHIFTPOP = 1
    ARCBINARY = 2
    ARCCONNECT = 3
    PUSHIDX = 4

shiftpop_feat_num = 14
cache_feat_num = 22
pushidx_feat_num = 30

NULL = "-NULL-"
UNKNOWN = "-UNK-"

symbols = set("'\".-!#&*|\\/")
re_symbols = re.compile("['\".\'`\-!#&*|\\/@=\[\]]")
special_token_map = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}

def allSymbols(s):
    return re.sub(re_symbols, "", s) == ""

def equals(s1, s2):
    s2_sub = s2.replace("-RRB-", ")").replace("-LRB-", "(")
    s2_sub = re.sub(re_symbols, "", s2_sub)
    s1 = s1.replace("\xc2\xa3", "#")
    s1_sub = re.sub(re_symbols, "", s1)

    return s1_sub.lower() == s2_sub.lower()

#Search s in seq from index
def searchSeq(s, seq, index):
    for i in xrange(index, len(seq)):
        if s == "@-@":
            if "".join(seq[i:i+3]) == "@-@":
                return (i, i+3)
        if s == "\"":
            if "".join(seq[i:i+2]) == "\'\'":
                return (i, i+2)
            if seq[i] == "\'\'":
                return (i, i+1)
        if allSymbols(s):
            if seq[i] == s:
                return (i, i+1)
        if allSymbols(seq[i]):
            continue
        for j in xrange(i+1, len(seq)+1):
            if allSymbols(seq[j-1]):
                continue
            curr_seq = "".join(seq[i:j])
            if equals(s, curr_seq):
                return (i, j)
    return (-1, -1)

def loadTokens(input_file):
    ret = []
    with open(input_file, 'r') as input_f:
        for line in input_f:
            ret.append(line.strip().split())
        input_f.close()
    return ret

def loadDepTokens(dep_file):
    ret = []
    with open(dep_file, "r") as dep_f:
        sent_idx = 0
        tok_seq = []
        for line in dep_f:
            splits = line.strip().split("\t")
            if len(splits) < 10:
                if tok_seq:
                    ret.append(tok_seq)
                tok_seq = []
                sent_idx += 1
            else:
                word = splits[1]
                for sp in special_token_map:
                    if sp in word:
                        word = word.replace(sp, special_token_map[sp])
                # if word in special_token_map:
                #     word = special_token_map[word]
                tok_seq.append(word)
        dep_f.close()
    return ret

#Based on some heuristic tokenize rules, map the original toks to the tokenized result
#Also map the toks to the new alignment toks
def mergeToks(orig_toks, tokenized_seq, all_alignments, sent_index):
    new_alignment = defaultdict(list)
    matched_index = 0
    triple_list = []
    for index in all_alignments:
        for (start, end, wiki_label) in all_alignments[index]:
            triple_list.append((index, (start, end, wiki_label)))
    sorted_alignments = sorted(triple_list, key=lambda x: (x[1][0], x[1][1]))

    visited = set()
    for (index, (start, end, wiki_label)) in sorted_alignments:
        for i in xrange(start, end):
            if i in visited:
                continue
            if i < end:
                curr_span = "".join(orig_toks[i: end])
                if allSymbols(curr_span):
                    print curr_span, wiki_label
                    break

                (new_start, new_end) = searchSeq(curr_span, tokenized_seq, matched_index)
                if new_start == -1:
                    print ("Something wrong here in %d" % sent_index)
                    print curr_span
                    print orig_toks
                    print tokenized_seq
                    print matched_index, tokenized_seq[matched_index]

                    sys.exit(1)
                visited |= set(xrange(i, end))
                matched_index = new_end
                new_alignment[index].append((new_start, new_end, wiki_label))
                #print " ".join(orig_toks[i:end]), "  :  ", " ".join(tokenized_seq[new_start:new_end])
    return new_alignment

