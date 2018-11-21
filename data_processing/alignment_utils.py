#!/usr/bin/python
from amr_utils import *
import logger
from re_utils import *
from constants import *
from date_extraction import *
from utils import *

neg_prefix = ['dis','un','im','in', 'anti']
neg_suffix = ['less']
freq_vertices = ['person', 'thing']
role_rel_concepts = set(["have-org-role-91", "have-rel-role-91", "be-located-at-91"])
def similarity(toks, oth_toks):
    sim_score = 0
    for tok in toks:
        for oth_tok in oth_toks:
            if tok.lower() == oth_tok.lower():
                sim_score += 1
                break
    return sim_score

def removeAlignedSpans(spans, aligned_toks):
    new_spans = []
    for (start, end) in spans:
        curr_set = set(range(start, end))
        if len(curr_set & aligned_toks) == 0:
            new_spans.append((start, end))
    return new_spans

#Given the entity mention of the fragment
#Remove all mentions that are redundant
def removeRedundant(toks, entity_spans, op_toks):
    """
    Fix wrong alignments for entities.
    """
    all_spans = []
    for (start, end) in entity_spans:
        sim_score = similarity(toks[start:end], op_toks)
        all_spans.append((sim_score, start, end))

    all_spans = sorted(all_spans, key=lambda x: -x[0])

    max_score = all_spans[0][0]
    remained_spans = [(start, end) for (sim_score, start, end) in all_spans if sim_score == max_score]
    return remained_spans

def removeDateRedundant(date_spans):
    """
    Fix wrong alignments for dates.
    """
    max_span = max([end-start for (start, end) in date_spans])
    remained_spans = [(start, end) for (start, end) in date_spans if end-start == max_span]
    return remained_spans

def getDateAttr(frag):
    root_index = frag.root
    amr_graph = frag.graph
    root_node = amr_graph.nodes[root_index]

    index_to_attr = {}

    attr_indices = set()
    for edge_index in root_node.v_edges:
        curr_edge = amr_graph.edges[edge_index]
        if curr_edge.label in date_relations:
            attr_indices.add(curr_edge.tail)
            index_to_attr[curr_edge.tail] = curr_edge.label.upper()
    return (attr_indices, index_to_attr)

#Given an alignment and the fragment, output the covered span
def getSpanSide(alignments, frag):
    aligned_set = set()
    amr_graph = frag.graph

    covered_set = set()
    all_date_attrs, index_to_attr = getDateAttr(frag)

    index_to_toks = defaultdict(list)

    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        start = int(curr_tok)

        aligned_set.add(start)

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        if index_type == 'c':
            if frag.nodes[index] == 1: #Covered current
                covered_set.add(start)
                index_to_toks[index].append(start)
                if index in all_date_attrs:
                    all_date_attrs.remove(index)

        else: #An edge covered span
            if frag.edges[index] == 1:
                covered_set.add(start)

    covered_toks = sorted(list(covered_set))
    non_covered = [amr_graph.nodes[index].node_str() for index in all_date_attrs]
    return covered_toks, non_covered, index_to_toks

def initializeAlignment(amr):
    def resetBitarray(arr):
        if arr.count() != 0:
            arr ^= arr
    node_alignment = bitarray(len(amr.nodes))
    edge_alignment = bitarray(len(amr.edges))
    resetBitarray(node_alignment)
    resetBitarray(edge_alignment)

    assert node_alignment.count() == 0 and edge_alignment.count() == 0
    return node_alignment, edge_alignment

def extractNodeMapping(alignments, amr_graph):
    aligned_set = set()

    node_to_toks = defaultdict(list)
    edge_to_toks = defaultdict(list)

    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        tok_idx = int(curr_tok)
        aligned_set.add(tok_idx)

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        if index_type == 'c':
            node_to_toks[index].append(tok_idx)
            # curr_node = amr_graph.nodes[index]
            # concept_align_map[tok_idx].append(curr_node.node_str())

        else:
            edge_to_toks[index].append(tok_idx)
            # relation_align_map[tok_idx].append(amr_graph.edges[index].label)

    return (node_to_toks, aligned_set)

def searchEntityTokens(entity_mention_toks, tok_seq, unaligned):
    def fuzzy_equal(s1, s2, max_len=5):
        if s1[:-1] == s2:
            return True
        if len(s1) >= max_len and s1 in s2:
            return True
        if len(s2) >= max_len and s2 in s1:
            return True
        return s1[:2] == s2[:2] and s1[-2:] == s2[-2:]

    entity_mention = "".join(entity_mention_toks)
    for start in unaligned:
        if allSymbols(tok_seq[start]):
            continue
        for end in unaligned:
            if allSymbols(tok_seq[end]):
                continue
            if end >= start:
                span_set = set(range(start, end+1))
                if len(span_set & unaligned) == len(span_set):
                    curr_mention = "".join(tok_seq[start:end+1])
                    if equals(curr_mention, entity_mention):
                        return (start, end+1)

    # Return lemmatization.
    for start in unaligned:
        curr_tok = tok_seq[start]
        if fuzzy_equal(curr_tok, entity_mention):
            return (start, start+1)
    return None

def alignEntities(tok_seq, amr, align_seq, entity_toks, aligned_toks, all_alignments,
                  unaligned_set, node_alignment):

    def sent_opt_toks():
        opt_toks = []
        role_toks = []
        for curr_align in reversed(align_seq):
            curr_tok, curr_frag = curr_align.split('-')[0], curr_align.split('-')[1]

            start = int(curr_tok)

            (index_type, index) = amr.get_concept_relation(curr_frag)
            if index_type == 'c':
                curr_node = amr.nodes[index]
                if len(curr_node.p_edges) == 1:
                    par_edge = amr.edges[curr_node.p_edges[0]]
                    if 'op' == par_edge.label[:2]:
                        opt_toks.append((start, curr_node.c_edge))

                if curr_node.is_named_entity():
                    role_toks.append((start, curr_node.c_edge))
        return opt_toks, role_toks

    entity_not_align = False
    opt_toks, role_toks = sent_opt_toks()
    for (frag, wiki_label, category) in amr.extract_entities():
        if category == "NER":
            root_index = frag.root
            if len(opt_toks) == 0:
                logger.writeln("No alignment for the entity found")

            (aligned_indexes, entity_spans) = all_aligned_spans(frag, opt_toks, role_toks, unaligned_set)
            root_node = amr.nodes[frag.root]

            entity_mention_toks = root_node.namedEntityMention()
            entity_tok_repr = "_".join(entity_mention_toks)
            if entity_spans:
                entity_spans = removeAlignedSpans(entity_spans, aligned_toks)
            else:
                matched_span = searchEntityTokens(entity_mention_toks, tok_seq, unaligned_set)
                if matched_span:
                    entity_spans = [matched_span]

            if entity_spans:
                entity_spans = removeRedundant(tok_seq, entity_spans, entity_mention_toks)

                start, end = entity_spans[0]  # Currently we only align to the first mention.
                entity_toks |= set(xrange(start, end))
                aligned_toks |= set(xrange(start, end))
                logger.writeln("%d-%d: %s" % (start, end, ' '.join(tok_seq[start:end])))
                ner_repr = "%s--%s" % (wiki_label, entity_tok_repr)
                all_alignments[frag.root].append((start, end, ner_repr, "NER"))
                # print node_alignment, frag.nodes
                node_alignment |= frag.nodes
            else:
                entity_not_align = True
                logger.writeln("Unaligned entity: %s" % str(frag))
                logger.writeln("Entity mention tokens: %s" % " ".join(entity_mention_toks))
        elif category == "DATE":
            if frag is None:
                entity_not_align = True
                logger.writeln("Date with children relations:")
                logger.writeln(str(amr))
                continue
            covered_toks, non_covered, index_to_toks = getSpanSide(align_seq, frag)

            covered_set = set(covered_toks)
            root_index = frag.root

            all_spans = getContinuousSpans(covered_toks, unaligned_set, covered_set)
            all_spans = removeAlignedSpans(all_spans, aligned_toks)

            if all_spans:
                temp_spans = []
                for start, end in all_spans:
                    while start > 0 and (start-1) in unaligned_set:
                        if tok_seq[start-1] == "th" or (tok_seq[start-1]
                                                        in str(frag) and tok_seq[start-1][0] in '0123456789'):
                            start -= 1
                        else:
                            break

                    while end < len(tok_seq):
                        if tok_seq[end] in date_suffixes:  # Some date prefixes
                            end += 1
                        else:
                            break

                    temp_spans.append((start, end))
                all_spans = temp_spans
                all_spans = removeDateRedundant(all_spans)
                for start, end in all_spans:
                    curr_set = set(xrange(start, end))
                    if len(curr_set & aligned_toks) != 0:
                        continue
                    all_alignments[frag.root].append((start, end, "NONE", "DATE"))
                    aligned_toks |= set(xrange(start, end))
                    entity_toks |= set(xrange(start, end))
                    node_alignment |= frag.nodes
                    break
            else:
                date_repr = str(frag)
                for index in unaligned_set:
                    curr_tok = tok_seq[index]
                    found = False
                    for un_tok in non_covered:
                        if curr_tok[0] in '0123456789' and curr_tok in un_tok:
                            found = True
                            break
                    if curr_tok in decades and decades[curr_tok] in date_repr:
                        found = True
                    if found:
                        break
                if found:
                    all_alignments[frag.root].append((index, index+1, "NONE", "DATE"))
                    aligned_toks.add(index)
                    entity_toks.add(index)
                    node_alignment |= frag.nodes
                else:
                    logger.writeln("Unaligned entity: %s" % date_repr)
                    entity_not_align = True
    return entity_not_align

def outputEdgeAlignment(tok_seq, amr, edge_to_toks, tok2rels):
    for edge_index in edge_to_toks:
        edge_label = amr.edges[edge_index].label
        for tok_idx in edge_to_toks[edge_index]:
            logger.writeln("Relation align: align %s to %s" % (tok_seq[tok_idx], edge_label))
            tok2rels[tok_idx].add(edge_index)

def subgraph_str(subgraph):
    ret = ""
    for root_repr in subgraph:
        ret += root_repr
        for (rel, tail_repr) in subgraph[root_repr].items():
            ret += " :%s %s" % (rel, tail_repr)
    return ret

def alignVerbalization(tok_seq, lemma_seq, amr, verb_list, all_alignments, verb_map, aligned_toks, node_alignment,
                       multi_map):

    matched_tuples = set()
    for (idx, curr_tok) in enumerate(tok_seq):
        if idx in aligned_toks:
            continue
        if not curr_tok in verb_list:
            curr_tok = lemma_seq[idx]
        if curr_tok in verb_list:
            for subgraph in verb_list[curr_tok]:
                matched_frags = amr.matchSubgraph(subgraph)
                if matched_frags:
                    subgraph_repr = subgraph_str(subgraph)
                    if len(matched_frags) > 1:
                        logger.writeln("Verbalize %s to more than 1 occurrences!" % curr_tok)
                    for frag_tuples in matched_frags:
                        valid = True
                        for (head, rel, tail) in frag_tuples:
                            if (head, rel, tail) in matched_tuples:
                                valid = False
                                break
                            matched_tuples.add((head, rel, tail))
                        if valid:
                            logger.writeln("Verbalize %d-%d, %s to %s!" % (idx, idx+1, curr_tok, subgraph_repr))
                            aligned_toks.add(idx)
                            for (head, rel, tail) in frag_tuples:
                                verb_map[head].add((head, rel, tail))
                                node_alignment[head] = 1
                                node_alignment[tail] = 1
                            all_alignments[head].append((idx, idx+1, subgraph_repr, "MULT"))
                            head = frag_tuples[0][0]
                            # head_concept = amr.nodes[head].node_str()
                            multi_map[subgraph_repr] += 1
                            break

def alignOtherConcepts(tok_seq, lem_seq, amr, aligned_toks, aligned_nodes, node_to_toks, all_alignments,
                       multi_map=None, quantity_map=None, entity_map=None):
    def hasdigit(s):
        return any(c.isdigit() for c in s)

    def valid(node_str, length):
        if hasdigit(node_str):
            return True
        return length - len(node_str.split("-")) <= 1

    def removeAligned(mapped_toks, node_str):
        spans = []
        start_idx = -1
        end_idx = -1
        visited = set()
        for tok_idx in sorted(mapped_toks):
            if tok_idx in aligned_toks or tok_idx in visited:
                continue
            visited.add(tok_idx)
            if end_idx == -1:
                start_idx = tok_idx
            elif tok_idx != end_idx:
                if valid(node_str, end_idx-start_idx):
                    spans.append((start_idx, end_idx))
                start_idx = tok_idx
            end_idx = tok_idx + 1

        if end_idx != -1 and valid(node_str, end_idx-start_idx):
            spans.append((start_idx, end_idx))
        return spans

    def fuzzy_match(token, concept, max_len = 4):
        if len(token) < max_len:
            return False
        elif concept.startswith(token[:max_len]):
            return True
        else:
            return False

    def isNegation(word, concept):
        neg_concepts = [prefix + concept for prefix in neg_prefix]
        # neg_concepts.append(concept + "less")
        for neg in neg_concepts:
            if neg == word or fuzzy_match(word, neg, 7):
                return True
        return False

    def otherCategory(concept):
        if "-quantity" in concept:
            return "QTY"
        if "-entity" in concept:
            return "ENT"
        return "TOK"

    def searchParent(indices, word_repr):
        match_idx = -1
        for node_idx in indices:
            curr_node = amr.nodes[node_idx]
            if match_idx == -1 and curr_node.node_str() in word_repr:
                match_idx = node_idx
            for par_edge_idx in curr_node.p_edges:
                par_edge = amr.edges[par_edge_idx]
                par_node_idx = par_edge.head
                if par_node_idx in indices:
                    par_node = amr.nodes[par_node_idx]
                    ret = []
                    for edge_idx in par_node.v_edges:
                        tail_idx = amr.edges[edge_idx].tail
                        if tail_idx in indices:
                            ret.append((edge_idx, tail_idx))
                    assert ret, "Empty child list"
                    return par_node_idx, ret
        if match_idx != -1:
            return match_idx, []
        return indices[0], []

    def mergeNodes(aligned_toks, span_to_nodes):
        for (start, end) in span_to_nodes:
            indices = span_to_nodes[(start, end)]
            curr_words = "@".join(tok_seq[start:end])
            curr_set = set(range(start, end))
            if len(aligned_toks & curr_set) != 0:
                continue
            aligned_toks |= set(range(start, end))

            if len(indices) == 1:
                # First try to match negation.
                node_idx = indices[0]
                curr_node = amr.nodes[node_idx]
                node_repr = curr_node.node_str()
                parent_idx, parent_rel, parent_concept = None, None, None
                if curr_node.p_edges:
                    parent_edge = amr.edges[curr_node.p_edges[0]]
                    parent_idx = parent_edge.head
                    parent_rel = parent_edge.label
                    parent_concept = amr.nodes[parent_idx].node_str()

                if parent_idx is not None and curr_node.is_negative_polarity() and isNegation(curr_words,
                                                                                              parent_concept):
                    all_alignments[parent_idx].append((start, end, "NEG_%s" % parent_concept, "NEG"))
                    aligned_nodes.add(node_idx)
                    aligned_nodes.add(parent_idx)
                elif isNegation(curr_words, node_repr) and curr_node.search_polarity() != -1:
                    edge_idx = curr_node.search_polarity()
                    tail_idx = amr.edges[edge_idx].tail
                    all_alignments[node_idx].append((start, end, "NEG_%s" % node_repr, "NEG"))
                    aligned_nodes.add(node_idx)
                    aligned_nodes.add(tail_idx)
                else:
                    if parent_concept and parent_idx not in aligned_nodes and parent_idx not in node_to_toks:
                        if parent_rel[:3] == "ARG" and parent_rel[-3:] == "-of":
                            curr_repr = "%s %s:%s" % (parent_concept, parent_rel, node_repr)
                            all_alignments[parent_idx].append((start, end, curr_repr, "MULT"))
                            # multi_map[parent_concept] += 1
                            multi_map[curr_repr] += 1
                            aligned_nodes.add(node_idx)
                            aligned_nodes.add(parent_idx)
                            continue
                    aligned_nodes.add(node_idx)
                    category = "TOKEN"

                    if isNumber(node_repr):
                        template = []
                        for k in xrange(start, end):
                            if not isNumber(tok_seq[k]):
                                template.append(tok_seq[k])
                            else:
                                template.append("NUM")
                        category = "NUMBER"
                        logger.writeln("NUMBER template:#%s" % " ".join(template))
                    elif end - start > 1:
                        category = "PHRASE"

                    all_alignments[node_idx].append((start, end, node_repr, category))
            else: # More than one vertices, then try to merge.
                parent_idx, child_edges = searchParent(indices, curr_words)
                parent_concept = amr.nodes[parent_idx].node_str()
                curr_repr = parent_concept
                aligned_nodes.add(parent_idx)

                for (edge_idx, tail_idx) in child_edges:
                    aligned_nodes.add(tail_idx)
                    curr_repr += (" %s:%s" % (amr.edges[edge_idx].label, amr.nodes[tail_idx].node_str()))
                category = "TOKEN"

                if len(child_edges) > 0:
                    multi_map[curr_repr] += 1
                    category = "MULT"
                elif isNumber(parent_concept):
                    category = "NUMBER"
                elif end - start > 1:
                    category = "PHRASE"
                all_alignments[parent_idx].append((start, end, curr_repr, category))

                if len(child_edges) >= 2:
                    logger.writeln("Aligned to more than 2 concepts:")
                    logger.writeln("&&%d-%d: %s" % (start, end, curr_repr))

    def spanAlignment():
        span_to_nodes = defaultdict(list)
        for node_idx in node_to_toks:
            if node_idx in aligned_nodes:
                continue
            node_repr = amr.nodes[node_idx].node_str()
            spans = removeAligned(node_to_toks[node_idx], node_repr)
            if len(spans) > 1:
                # node_repr = amr.nodes[node_idx].node_str()
                logger.writeln("Multiple to one alignment for concept: %s" % node_repr)
                for start, end in spans:
                    logger.writeln("##%d-%d: %s" % (start, end, " ".join(tok_seq[start:end])))
            for start, end in spans:
                if end - start > 1:
                    logger.writeln("Multiple word mapped to concept: %s" % node_repr)
                    logger.writeln("---%d-%d: %s : %s" % (start, end, " ".join(tok_seq[start:end]), node_repr))
                span_to_nodes[(start, end)].append(node_idx)
        return span_to_nodes

    def similar(node_str, tok):
        if node_str == tok:
            return True
        if len(tok) > 3 and tok[:4] == node_str[:4]:
            return True
        if isNumber(tok) and tok in node_str:
            return True
        return isNumber(node_str) and node_str in tok

    def retrieveUnaligned():
        num_nodes = len(amr.nodes)
        for node_idx in range(num_nodes):
            if node_idx in aligned_nodes:
                continue
            freq = amr.getFreq(node_idx)
            curr_node = amr.nodes[node_idx]
            node_str = curr_node.node_str()
            if node_str in role_rel_concepts:
                continue
            if (freq and freq < 100) or amr.is_predicate(curr_node):
                for (idx, word) in enumerate(tok_seq):
                    if idx not in aligned_toks:
                        lem = lem_seq[idx]
                        if similar(node_str, word) or similar(node_str, lem):
                            logger.writeln("Retrieved concept map: %s, %s ; %s" % (word, lem, node_str))
                            category = "TOKEN"
                            if isNumber(node_str) or isNumber(word):
                                category = "NUMBER"
                            all_alignments[node_idx].append((idx, idx+1, node_str, category))
                            aligned_nodes.add(node_idx)
                            aligned_toks.add(idx)

    span_to_nodes = spanAlignment()
    mergeNodes(aligned_toks, span_to_nodes)
    retrieveUnaligned()
    return


def align_semeval_sentence(tok_seq, lemma_seq, alignment_seq, amr, verb_list, multi_map):
    node_alignment, _ = initializeAlignment(amr)
    entity_toks = set()
    aligned_toks = set()
    all_alignments = defaultdict(list)
    node_to_toks, temp_aligned = extractNodeMapping(alignment_seq, amr)
    unaligned_set = set(xrange(len(tok_seq))) - temp_aligned
    alignEntities(tok_seq, amr, alignment_seq, entity_toks,
                  aligned_toks, all_alignments, unaligned_set, node_alignment)

    #Verbalization list
    verb_map = defaultdict(set)
    alignVerbalization(tok_seq, lemma_seq, amr, verb_list, all_alignments, verb_map,
                       aligned_toks, node_alignment, multi_map)

    aligned_nodes = set([node_idx for (node_idx, aligned) in enumerate(node_alignment) if aligned])

    alignOtherConcepts(tok_seq, lemma_seq, amr, aligned_toks, aligned_nodes, node_to_toks,
                       all_alignments, multi_map)

    ##Based on the alignment from node index to spans in the string
    unaligned_set = set(xrange(len(tok_seq))) - aligned_toks
    unaligned_idxs = sorted(list(unaligned_set))
    logger.writeln("Unaligned tokens: %s" % (" ".join([tok_seq[i] for i in unaligned_idxs])))

    unaligned_nodes = amr.unaligned_nodes(aligned_nodes)
    logger.writeln("Unaligned vertices: %s" % " ".join([node.node_str() for node in unaligned_nodes]))

    return all_alignments

def build_alignment_maps(amr, all_alignments):
    start2end, category_map, node_map, wiki_map = {}, {}, {}, {}
    for node_idx, span_list in all_alignments.items():
        for span in span_list:
            if span[1] - span[0] > 6:
                continue
            start2end[span[0]] = span[1]
            wiki_map[span[0]] = span[2]
            category_map[span[0]] = span[3]
            node_map[span[0]] = amr.nodes[node_idx].node_str()
    return start2end, category_map, node_map, wiki_map

def align_jamr_sentence(tok_seq, alignment_seq, amr, phrases):
    print " ".join(tok_seq)
    # visited = set()
    length = len(tok_seq)
    visited_node_idxs = set()
    node_to_spans = defaultdict(list)
    all_alignments = defaultdict(list)
    idx_to_nodes = defaultdict(set)
    span_to_graph = {}
    global_aligned = set()
    aligned_set = set()
    aligned_node_idxs = set()
    for curr_align in alignment_seq:

        span_str = curr_align.split("|")[0]
        start = int(span_str.split("-")[0])
        end = int(span_str.split("-")[1])
        curr_set = set(xrange(start, end))
        if len(global_aligned & curr_set) != 0:
            continue
        global_aligned |= curr_set
        subgraph_repr = curr_align.split("|")[1]
        subgraph, node_idx_set = amr.initialize_subgraph(subgraph_repr)

        span_to_graph[(start, end)] = (subgraph, node_idx_set)

        for node_idx in node_idx_set:
            node_to_spans[node_idx].append((start, end))
            if len(node_to_spans[node_idx]) > 1:
                print "one to multiple spans:", alignment_seq, node_to_spans[node_idx]

        for idx in xrange(start, end):
            idx_to_nodes[idx] |= node_idx_set
        assert len(node_idx_set & visited_node_idxs) == 0
        visited_node_idxs |= node_idx_set

    unaligned_set = set(xrange(length)) - global_aligned
    alignJAMREntities(tok_seq, amr, aligned_set, all_alignments, unaligned_set, node_to_spans, aligned_node_idxs)
    unaligned_set -= aligned_set

    idx_to_collapsed = alignJAMROtherConcepts(tok_seq, amr, aligned_set, aligned_node_idxs, all_alignments,
                                              unaligned_set, node_to_spans, span_to_graph, phrases)
    return all_alignments, idx_to_collapsed

        # print "%d-%d, %s, %s" % (start, end, " ".join(tok_seq[start:end]), str(subgraph))

def alignJAMREntities(tok_seq, amr, aligned_set, all_alignments, unaligned_set, node_to_spans, aligned_node_idxs):

    for (frag, wiki_label, category) in amr.extract_entities():
        aligned_tok_list = []
        aligned_spans = []
        for node_idx in frag.node_list():
            if node_idx in node_to_spans:
                for start, end in node_to_spans[node_idx]:
                    aligned_spans.append((start, end))
                    for tok_idx in xrange(start, end):
                        aligned_tok_list.append(tok_idx)

        aligned_tok_list = sorted(aligned_tok_list)
        aligned_spans = getContinuousSpans(aligned_tok_list, unaligned_set, set(aligned_tok_list))
        aligned_spans = sorted(aligned_spans, key=lambda x: (x[0]-x[1], x))

        print str(frag)

        if category == "NER":
            root_node = amr.nodes[frag.root]
            entity_mention_toks = root_node.namedEntityMention()
            entity_tok_repr = "_".join(entity_mention_toks)
            if aligned_spans:
                start, end = aligned_spans[0]  # Currently we only align to the first mention.
                aligned_set |= set(xrange(start, end))
                print "NER: %d-%d, %s" % (start, end, ' '.join(tok_seq[start:end]))
                ner_repr = "%s--%s" % (wiki_label, entity_tok_repr)
                all_alignments[frag.root].append((start, end, ner_repr, "NER"))
                aligned_node_idxs |= set(frag.node_list())
                # node_alignment |= frag.nodes

        elif category == "DATE":
            all_date_attrs, index_to_attr = getDateAttr(frag)
            unmatched_attrs = [amr.nodes[index].node_str() for index in all_date_attrs if index not in node_to_spans]
            if aligned_spans:
                start, end = aligned_spans[0]
                # print tok_seq[start-1], unmatched_attrs
                while start > 0 and (start-1) in unaligned_set:
                    if tok_seq[start-1] == "th" or (tok_seq[start-1]
                                                    in str(frag) and tok_seq[start-1][0] in '0123456789'):
                        start -= 1
                    elif tok_seq[start-1] in unmatched_attrs:
                        start -= 1
                    else:
                        break

                    while end < len(tok_seq):
                        if tok_seq[end] in date_suffixes and end in unaligned_set:  # Some date prefixes
                            end += 1
                        else:
                            break

                print "DATE: %d-%d, %s" % (start, end, ' '.join(tok_seq[start:end]))
                if unmatched_attrs:
                    print "unmatched attributes: %s" % (" ".join(unmatched_attrs))
                all_alignments[frag.root].append((start, end, "NONE", "DATE"))
                aligned_node_idxs |= set(frag.node_list())
                aligned_set |= set(xrange(start, end))
        else:
            assert "Should be either NER or DATE entities"


def alignJAMROtherConcepts(tok_seq, amr, aligned_set, aligned_node_idxs, all_alignments, unaligned_set, node_to_spans,
                           span_to_graph, phrases, max_phrase_len=4, max_num_len=6):

    def value(toks): #Compute the value of the number representation
        number = 1

        for v in toks:
            if isNumber(v):
                if '.' in v:
                    if len(toks) == 1:
                        return v
                    number *= float(v)
                else:
                    v = v.replace(",", "")
                    number *= int(v)
            else:
                v = v.lower()
                if v in quantities:
                    # assert v in quantities, v
                    number *= quantities[v]
                    number = int(number)
        return str(number)

    def valid_phrase(start, end, span_set):
        for idx in span_set:
            if (idx >= start and idx < end) or idx in unaligned_set:
                continue
            return False
        return True

    def alignPhrases(node_idx):
        if node_idx not in node_to_spans:
            return None
        aligned_spans = node_to_spans[node_idx]
        assert len(aligned_spans) == 1, node_to_spans
        start, end = aligned_spans[0]

        for i in xrange(start - max_phrase_len, start+1):
            if i < 0:
                continue
            for j in xrange(end, end + max_phrase_len):
                if j > len(tok_seq):
                    break
                if " ".join(tok_seq[i:j]) in phrases:
                    span_set = set(xrange(i, j))
                    if valid_phrase(start, end, span_set):
                        return (i, j)
        return None

    def identifyNumber(seq):
        for tok in seq:
            if not (isNumber(tok) or tok in quantities):
                return False
        return True

    def alignNumber(node_idx):
        node_repr = amr.nodes[node_idx].node_str()
        if not isNumber(node_repr):
            return None

        matched_set = set()
        if node_idx in node_to_spans:
            for start, end in node_to_spans[node_idx]:
                matched_set |= set(xrange(start, end))

        # Greedily match numbers
        for tok_idx in xrange(len(tok_seq)):
            if tok_idx in aligned_set:
                continue
            for end_idx in xrange(tok_idx+max_num_len, tok_idx, -1):
                if end_idx > len(tok_seq):
                    continue
                if identifyNumber(tok_seq[tok_idx:end_idx]):
                    covered_set = set(xrange(tok_idx, end_idx))
                    if len(covered_set & aligned_set) == 0 and value(tok_seq[tok_idx:end_idx]) == node_repr:
                        return (tok_idx, end_idx)

        return None

    def alignMultiples(aligned_set, aligned_node_idxs):
        idx_to_collapsed = {}
        for (start, end) in span_to_graph:
            curr_set = set(xrange(start, end))
            if len(curr_set & aligned_set) != 0:
                continue
            aligned_subgraph, node_set = span_to_graph[(start, end)]
            if len(node_set & aligned_node_idxs) != 0:
                continue
            curr_repr = str(aligned_subgraph)
            if len(node_set) > 1:
                root_idx = amr.get_var_nodeidx(aligned_subgraph.roots[0])
            else:
                root_idx = list(node_set)[0]
            if len(node_set) > 1:
                all_alignments[root_idx].append((start, end, curr_repr, "MULT"))
                idx_to_collapsed[root_idx] = node_set
                print "MULTIPLE: %d-%d, %s, %s" % (start, end, ' '.join(tok_seq[start:end]), curr_repr)
                aligned_set |= curr_set
                aligned_node_idxs |= node_set
            elif end - start > 1:
                all_alignments[root_idx].append((start, end, curr_repr, "PHRASE"))
                print "PHRASE: %d-%d, %s, %s" % (start, end, ' '.join(tok_seq[start:end]), curr_repr)
                aligned_set |= curr_set
                aligned_node_idxs |= node_set
        return idx_to_collapsed

    n_nodes = len(amr.nodes)
    idx_to_collapsed = alignMultiples(aligned_set, aligned_node_idxs)
    # print "node to spans:", node_to_spans
    for node_idx in xrange(n_nodes):
        if node_idx in aligned_node_idxs:
            continue
        curr_repr = amr.nodes[node_idx].node_str()
        aligned_span = alignNumber(node_idx)
        if aligned_span:
            start, end = aligned_span
            curr_set = set(xrange(start, end))
            if len(aligned_set & curr_set) != 0:
                continue
            print "NUMBER: %d-%d, %s, %s" % (start, end, ' '.join(tok_seq[start:end]), curr_repr)
            all_alignments[node_idx].append((start, end, curr_repr, "NUMBER"))
            aligned_set |= curr_set
            aligned_node_idxs.add(node_idx)
            continue
        aligned_span = alignPhrases(node_idx)
        if aligned_span:
            start, end = aligned_span
            curr_set = set(xrange(start, end))
            if len(aligned_set & curr_set) != 0:
                continue
            all_alignments[node_idx].append((start, end, curr_repr, "PHRASE"))
            print "PHRASE: %d-%d, %s, %s" % (start, end, ' '.join(tok_seq[start:end]), curr_repr)
            aligned_set |= curr_set
            aligned_node_idxs.add(node_idx)
            continue

        # Otherwise, alignment one-to-one
        if node_idx in node_to_spans:

            aligned_spans = node_to_spans[node_idx]

            assert len(aligned_spans) == 1
            start, end = aligned_spans[0]
            curr_set = set(xrange(start, end))
            if len(aligned_set & curr_set) != 0:
                continue
            all_alignments[node_idx].append((start, end, curr_repr, "TOKEN"))
            aligned_set |= curr_set
            aligned_node_idxs.add(node_idx)

    return idx_to_collapsed
