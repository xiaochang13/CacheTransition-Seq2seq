#!/usr/bin/python
import re, os
import sys
from collections import defaultdict
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
def searchSeq(s, seq, index, aligned_set=None):
    for i in xrange(index, len(seq)):
        if aligned_set and i in aligned_set:
            continue
        if s == "@-@":
            if "".join(seq[i:i+3]) == "@-@":
                # if aligned_set and len(set(range(i, i+3)) & aligned_set) == 0:
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
            if aligned_set and len(set(range(i, j)) & aligned_set) != 0:
                continue
            if equals(s, curr_seq):
                return (i, j)
    return (-1, -1)

def in_bracket(start, end, tok_seq):
    length = len(tok_seq)
    has_left = False
    for idx in range(start-3, start):
        if idx >= 0 and tok_seq[idx] == "(":
            has_left = True
    if not has_left:
        return False
    for idx in range(end, end+3):
        if idx < length and tok_seq[idx] == ")":
            return True
    return False

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

def saveMLE(counts, dists, path, used_set=None, verify=False):
    def check(toks, node_repr):
        for s in toks:
            if s not in node_repr:
                return False
        return True
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    with open(path, "w") as wf:
        for (item ,count) in sorted_items:
            if count == 1:
                if verify and not check(item.split(), dists[item].items()[0][0]):
                    continue

            sorted_repr = sorted(dists[item].items(), key=lambda x: -x[1])
            dist_repr = ";".join(["%s:%d" % (s, c) for (s, c) in sorted_repr])
            if used_set and item not in used_set:
                print "Filtered phrase:", item
                continue
            print >>wf, "%s\t%d\t%s" % (item, count, dist_repr)
        wf.close()

def loadCountTable(path, max_len=3):
    counts = {}
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                fields = line.strip().split("\t")
                if len(fields[0].split()) > max_len:
                    print "Pruned phrase:", fields[0]
                    continue
                counts[fields[0]] = int(fields[1])
    return counts

def saveCounter(counts, path):
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    with open(path, "w") as wf:
        for (item ,count) in sorted_items:
            print >>wf, "%s\t%d" % (item, count)
        wf.close()

def saveSetorList(entities, path):
    with open(path, "w") as wf:
        for item in entities:
            print >>wf, item
        wf.close()

def loadMLEFile(path):
    mle_map = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                splits = line.strip().split("####")
                word = splits[0]
                category = splits[1].split("##")[0]
                node_repr = splits[1].split("##")[1]
                mle_map[word] = (category, node_repr)
    return mle_map

def loadPhrases(path):
    mle_map = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                splits = line.strip().split("\t")
                phrase = splits[0]
                node_repr = splits[2].split("#")[0].split(":")[0]
                mle_map[phrase] = node_repr
    return mle_map

#Build the entity map for concept identification
#Choose either the most probable category or the most probable node repr
#In the current setting we only care about NE and DATE
def loadMap(map_file):
    span_to_cate = {}

    #First load all possible mappings each span has
    with open(map_file, 'r') as map_f:
        for line in map_f:
            if line.strip():
                spans = line.strip().split('##')
                for s in spans:
                    try:
                        fields = s.split('++')
                        toks = fields[1]
                        wiki_label = fields[2]
                        node_repr = fields[3]
                        category = fields[-1]
                    except:
                        print spans, line
                        print fields
                        sys.exit(1)
                    if toks not in span_to_cate:
                        span_to_cate[toks] = defaultdict(int)
                    span_to_cate[toks][(category, node_repr, wiki_label)] += 1

    mle_map = {}
    for toks in span_to_cate:
        sorted_types = sorted(span_to_cate[toks].items(), key=lambda x:-x[1])
        curr_type = sorted_types[0][0][0]
        if curr_type[:2] == 'NE' or curr_type[:4] == 'DATE':
            mle_map[toks] = sorted_types[0][0]
    return mle_map

def dumpMap(mle_map, result_file):
    with open(result_file, 'w') as wf:
        for toks in mle_map:
            print >>wf, ('%s####%s##%s' % (toks, mle_map[toks][0], mle_map[toks][1]))

def dateMap(dateFile):
    dates_in_lines = []
    for line in open(dateFile):
        date_spans = []
        if line.strip():
            spans = line.strip().split()
            for sp in spans:
                start = int(sp.split('-')[0])
                end = int(sp.split('-')[1])
                date_spans.append((start, end))
        dates_in_lines.append(date_spans)
    return dates_in_lines

def alignDates(tok_file, dateFile):
    tok_seqs = loadTokens(tok_file)
    date_spans = loadTokens(dateFile)
    assert len(tok_seqs) == len(date_spans)
    for (idx, dates) in enumerate(date_spans):
        toks = tok_seqs[idx]
        for sp in dates:
            start = int(sp.split("-")[0])
            end = int(sp.split("-")[1])
            print " ".join(toks[start:end])

def check_tokenizer(input_file, tokenized_file):
    with open(input_file, 'r') as input_f:
        with open(tokenized_file, 'r') as tokenized_f:
            for input_line in input_f:
                tokenized_line = tokenized_f.readline()
                if input_line.strip():
                    input_repr = "".join(input_line.strip().split()).replace("\"", "").replace("\'", "")
                    tokenized_repr = "".join(tokenized_line.strip().split()).replace("\'", "")
                    if input_repr != tokenized_repr:
                        print "original tokens:", input_line.strip()
                        print "tokenized:", tokenized_line.strip()

def loadFrequentSet(path):
    concept_path = os.path.join(path, "concept_counts.txt")
    frequent_concept = loadbyfreq(concept_path, 500)

    conceptoutgo_path = os.path.join(path, "concept_rels.txt")
    frequent_outgo = loadbyfreq(conceptoutgo_path, 100)

    conceptincome_path = os.path.join(path, "concept_incomes.txt")
    frequent_income = loadbyfreq(conceptincome_path, 100)

    return frequent_concept | (frequent_income & frequent_outgo)

def getCategories(token, frequent_set, verify=False):
    if verify:
        assert token != "NONE"

    if token in frequent_set:
        return token
    if token == "NONE":
        return token
    if "MULT" in token:
        assert "MULT_" in token, token
        return getCategories(token[5:], frequent_set)
    if "NEG" in token:
        assert "NEG_" in token, token
        return getCategories(token[4:], frequent_set)
    if token == "NE" or "NE_" in token:
        return "NE"
    if token == "NUMBER" or token == "DATE":
        return token
    if re.match(".*-[0-9]+", token) is not None:
        return "PRED"
    return "OTHER"

def special_categories(concept):
    if concept in set(["NUMBER", "DATE", "NE"]):
        return True
    return concept[:4] == "MULT" or concept[:3] == "NEG" or concept[:3] == "NE_"

def entity_category(concept):
    if concept in set(["NUMBER", "DATE", "NE"]):
        return True
    return concept[:3] == "NE_"

def loadbyfreq(file, threshold, delimiter="\t"):
    curr_f = open(file, 'r')
    concept_set = set()
    for line in curr_f:
        if line.strip():
            fields = line.strip().split(delimiter)
            concept = fields[0]
            curr_freq = int(fields[1])
            if curr_freq >= threshold:
                concept_set.add(concept)
    return concept_set

def saveSpanMap(span_lists, path, delimiter="####"):
    with open(path, "w") as wf:
        for span_seq in span_lists:
            new_span_seq = sorted([(int(s.split(delimiter)[0].split("-")[0]),
                                    int(s.split(delimiter)[0].split("-")[1]), s) for s in span_seq])
            print >> wf, "\t".join([s for (_, _, s) in new_span_seq])
        wf.close()

def loadSpanMap(path, delimiter="\t", type="NER"):
    all_maps = []
    with open(path, 'r') as f:
        for line in f:
            span_map = {}
            if not line.strip():
                all_maps.append(span_map)
                continue
            for map_str in line.strip().split(delimiter):
                span_str = map_str.split("####")[0]
                start = int(span_str.split("-")[0])
                end = int(span_str.split("-")[1])

                concept_str = map_str.split("####")[1].strip()
                if type == "NER":
                    ne_type = concept_str.split("##")[0].strip()
                    wiki_str = concept_str.split("##")[1].strip()
                    span_map[(start, end)] = (ne_type, wiki_str)
                else: #DATE
                    span_map[(start, end)] = concept_str
            all_maps.append(span_map)
        return all_maps

# def loadSpanMap(path, )

# alignDates(sys.argv[1], sys.argv[2])