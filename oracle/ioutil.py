import sys, os
from dependency import DependencyTree
from AMRGraph import *
NULL = "-NULL-"
UNKNOWN = "-UNK-"
from utils import loadDepTokens
class Dataset(object):
    def __init__(self, tok_thre=1, top_arc=50):
        self.tok_seqs = None
        self.lem_seqs = None
        self.pos_seqs = None
        self.dep_trees = None
        self.amr_graphs = None

        self.tok_threshold = tok_thre
        self.top_arc_num = top_arc

        self.known_toks = None
        self.known_lems = None
        self.known_poss = None
        self.known_deps = None
        self.known_concepts = None
        self.known_rels = None
        self.all_labels = None

        self.tokIDs = {}
        self.lemIDs = {}
        self.posIDs = {}
        self.depIDs = {}
        self.conceptIDs = {}
        self.relIDs = {}

        self.tok_offset = 0
        self.lem_offset = 0
        self.pos_offset = 0
        self.dep_offset = 0
        self.concept_offset = 0
        self.rel_offset = 0

        self.unaligned_set = set()

    def setTok(self, tok_seqs):
        self.tok_seqs = tok_seqs

    def setLemma(self, lem_seqs):
        self.lem_seqs = lem_seqs

    def setPOS(self, pos_seqs):
        self.pos_seqs = pos_seqs

    def setDepTrees(self, dep_trees):
        self.dep_trees = dep_trees

    def setAMRGraphs(self, amr_graphs):
        self.amr_graphs = amr_graphs

    def getInstance(self, idx):
        return (self.tok_seqs[idx], self.lem_seqs[idx],
                self.pos_seqs[idx], self.dep_trees[idx],
                self.amr_graphs[idx])

    def dataSize(self):
        return len(self.tok_seqs)

    def genDictionaries(self):
        """
        Given the tokens and labels of the dataset,
        generate the dictionaries for this dataset.
        :return:
        """
        def flattenedSeq(all_seqs):
            labels = []
            for seq in all_seqs:
                for tok in seq:
                    labels.append(tok)
            return labels

        def depLabels(all_trees):
            labels = []
            for tree in all_trees:
                for l in tree.label_list:
                    labels.append(l)
            return labels

        def amrLabels(all_graphs):
            concept_labels = []
            arc_labels = []
            for graph in all_graphs:
                for concept in graph.concepts:
                    concept_labels.append(concept.getValue())
                    for l in concept.rels:
                        arc_labels.append(l)
            return concept_labels, arc_labels

        tok_list = flattenedSeq(self.tok_seqs)
        lem_list = flattenedSeq(self.lem_seqs)
        pos_list = flattenedSeq(self.pos_seqs)
        dep_list = depLabels(self.dep_trees)
        concept_list, arc_list = amrLabels(self.amr_graphs)

        # Generate the label list.
        self.known_toks = generateDictionary(tok_list, self.tok_threshold)
        self.known_lems = generateDictionary(lem_list)
        self.known_poss = generateDictionary(pos_list)
        self.known_deps = generateDictionary(dep_list)
        self.known_concepts = generateDictionary(concept_list)
        self.known_rels = generateDictionary(arc_list, self.top_arc_num)

        self.known_toks.insert(0, UNKNOWN)
        self.known_toks.insert(1, NULL)

        self.known_lems.insert(0, UNKNOWN)
        self.known_lems.insert(1, NULL)

        self.known_poss.insert(0, UNKNOWN)
        self.known_poss.insert(1, NULL)

        self.known_deps.insert(0, UNKNOWN)
        self.known_deps.insert(1, NULL)

        self.known_concepts.insert(0, UNKNOWN)
        self.known_concepts.insert(1, NULL)

        self.known_rels.insert(0, UNKNOWN)
        self.known_rels.insert(1, NULL)

        self.all_labels = self.known_toks + self.known_lems + self.known_poss + self.known_deps + \
                          self.known_concepts + self.known_rels

        self.generateIDs()

        print>> sys.stderr, "#Tokens: %d" % len(self.known_toks)
        print>> sys.stderr, "#Lemmas: %d" % len(self.known_lems)
        print>> sys.stderr, "#POSs: %d" % len(self.known_poss)
        print>> sys.stderr, "#Dependency labels: %d" % len(self.known_deps)
        print>> sys.stderr, "#Concepts: %d" % len(self.known_concepts)
        print>> sys.stderr, "#Relations: %d" % len(self.known_rels)

    def generateIDs(self):
        for idx, tok in enumerate(self.known_toks):
            self.tokIDs[tok] = idx
        self.lem_offset = len(self.known_toks)
        for idx, lem in enumerate(self.known_lems):
            self.lemIDs[lem] = self.lem_offset + idx
        self.pos_offset = self.lem_offset + len(self.known_lems)
        for idx, pos in enumerate(self.known_poss):
            self.posIDs[pos] = self.pos_offset + idx
        self.dep_offset = self.pos_offset + len(self.known_poss)
        for idx, dep in enumerate(self.known_deps):
            self.depIDs[dep] = self.dep_offset + idx
        self.concept_offset = self.dep_offset + len(self.known_deps)
        for idx, concept in enumerate(self.known_concepts):
            self.conceptIDs[concept] = self.concept_offset + idx
        self.rel_offset = self.concept_offset + len(self.known_concepts)
        for idx, rel in enumerate(self.known_rels):
            self.relIDs[rel] = self.rel_offset + idx
            assert self.all_labels[self.rel_offset+idx] == rel

    def genConceptMap(self):
        """
        To be updated!
        :return:
        """
        return

    def saveConceptID(self, path):
        """
        To be updated!
        :param path:
        :return:
        """
        return

    def saveMLEConceptID(self, path):
        """
        To be updated!
        :param path:
        :return:
        """
        return


def generateDictionary(label_list, thre=1):
    counts = defaultdict(int)
    for l in label_list:
        counts[l] += 1
    new_label_list = []
    sorted_label_list = sorted(counts.items(), key=lambda x: -x[1])
    for (l, count) in sorted_label_list:
        if count < thre:
            break
        if l == UNKNOWN or l == NULL:
            continue
        new_label_list.append(l)
    return new_label_list

def topkDictionary(label_list, k):
    counts = defaultdict(int)
    for l in label_list:
        counts[l] += 1
    new_label_list = []
    sorted_label_list = sorted(counts.items(), key=lambda x: -x[1])
    for i in range(k):
        new_label_list.append(sorted_label_list[i][0])
    return new_label_list

def readToks(path):
    tok_seqs = []
    with open(path, "r") as tok_f:
        for line in tok_f:
            line = line.strip()
            if not line:
                break
            toks = line.split()
            tok_seqs.append(toks)
    return tok_seqs

def loadArcMaps(path, threshold=0.02):
    def reduce_op(s):
        if len(s) < 3:
            return s
        if s[:2] == "op" and s[2] in "0123456789":
            return "op"
        return s
    arc_choices = defaultdict(set)
    with open(path, "r") as arc_f:
        for line in arc_f:
            line = line.strip()
            if line:
                splits = line.split("\t")
                assert len(splits) == 3, "Incorrect arc map!"
                concept = splits[0].strip()
                total_count = int(splits[1])
                arc_tuples = [(item.split(":")[0], int(item.split(":")[1])) for item
                              in splits[2].strip().split(";")]
                # total_count = sum([count for (_, count) in arc_tuples])
                ceiling = total_count * (1-threshold)
                curr_total = 0.0
                for (l, count) in arc_tuples:
                    if curr_total > ceiling:
                        break
                    curr_total += count
                    arc_choices[concept].add(reduce_op(l))
    return arc_choices

def defaultArcChoices():
    frequent_rels = ["ARG0", "ARG1", "ARG2", "mod"]
    return set(["L-%s" % item for item in frequent_rels]) | set(["R-%s" % item for item in frequent_rels])

def loadCounter(path, delimiter="\t"):
    actions = set()
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                actions.add(line.strip().split(delimiter)[0])
    return actions

def loadDependency(path, tok_seqs, align=False):
    def dep_equal(first, second):
        first_repr = first.replace("`", "\'")
        second_repr = second.replace("`", "\'")
        return first_repr == second_repr

    def get_alignmaps():
        dep_tok_seqs = loadDepTokens(path)

        assert len(tok_seqs) == len(dep_tok_seqs)
        align_maps = []
        for (i, tok_seq) in enumerate(tok_seqs):
            dep_seq = dep_tok_seqs[i]
            align_map = {}
            tok_idx = 0
            curr_repr = ""
            for (k, dep_tok) in enumerate(dep_seq):
                curr_repr += dep_tok
                align_map[k] = tok_idx
                if dep_equal(curr_repr, tok_seq[tok_idx]):
                    tok_idx += 1
                    curr_repr = ""
            try:
                assert tok_idx == len(tok_seq)
            except:
                print tok_seq
                print dep_seq
                sys.exit(1)
            # print align_map
            align_maps.append(align_map)
        return align_maps

    dep_trees = []
    align_maps = None
    if align:
        align_maps = get_alignmaps()
    with open(path, "r") as dep_f:
        tree = DependencyTree()
        sent_idx = 0
        tok_idx = 0
        last_idx = -1
        toks = tok_seqs[sent_idx]
        # index_map = {}
        offset = 0
        prev_tok_idx = -1
        align_map = None
        if align:
            align_map = align_maps[sent_idx]
        for line in dep_f:
            splits = line.strip().split("\t")
            if len(splits) < 10:
                tree.buildDepDist(tree.dist_threshold)
                dep_trees.append(tree)
                # print toks

                assert len(toks) == len(tree.head_list), "%s %s %d %d" % (str(toks), str(tree.head_list), len(toks), len(tree.head_list))

                tree = DependencyTree()
                sent_idx += 1
                tok_idx = 0
                offset = 0
                prev_tok_idx = -1
                if sent_idx < len(tok_seqs):
                    toks = tok_seqs[sent_idx]
                    if align:
                        align_map = align_maps[sent_idx]
                    last_idx = -1
            else:
                dep_label = splits[7]

                tok_idx = int(splits[0]) - 1
                if tok_idx <= prev_tok_idx:
                    assert tok_idx == 0 # start a new sentence in dependency.
                    offset += (prev_tok_idx+1)
                prev_tok_idx = tok_idx

                if align:
                    tok_idx = align_map[tok_idx]
                    if tok_idx == last_idx:
                        continue
                    last_idx = tok_idx
                head_idx = int(splits[6]) - 1 # root becomes -1.
                if align and head_idx >= 0:
                    head_idx = align_map[head_idx]
                if head_idx != -1:
                    head_idx += offset

                tree.add(head_idx, dep_label)
        dep_f.close()
    return dep_trees

def loadAMRConll(path):
    amr_graphs = []
    with open(path, "r") as amr_f:
        graph = AMRGraph()
        visited = set()
        root_num = 0
        sent_idx = 0
        for line in amr_f:
            splits = line.strip().split("\t")
            if not line.strip():
                graph.buildEdgeMap()

                # TODO: change the concept to word mapping.
                graph.buildWordToConceptIDX()
                amr_graphs.append(graph)
                try:
                    assert root_num == 1, "Sentence %d" % sent_idx
                except:
                    print "cycle at sentence %d" % sent_idx

                graph = AMRGraph()
                root_num = 0
            elif splits[0].startswith("sentence "):  # sentence index
                visited = set()
                sent_idx += 1
                # print "Sentence %d" % sent_idx
            else:
                if len(splits) != 8:
                    print>> sys.stderr, "Length inconsistent in conll format %s" % len(splits)
                    print>> sys.stderr, " ".join(splits)
                    sys.exit(1)
                concept_idx = int(splits[0])
                is_var = bool(splits[1])
                concept_label = splits[2]
                # TODO: replace this with a span, and make further changes.
                word_idx = splits[3]
                outgoing_rels = splits[4]
                parent_rels = splits[5]
                category_label = splits[6]
                map_label = splits[7]
                c = ConceptLabel(concept_label)
                c.setVarType(is_var)
                c.map_info = map_label
                c.category = category_label
                if word_idx == "NONE":
                    c.aligned = False
                else:
                    c.aligned = True
                    # wids = word_idx.split("#")
                    word_span = word_idx.split("-")
                    start, end = int(word_span[0]), int(word_span[1])
                    curr_set = set(xrange(start, end))
                    # TODO: there can be repeated spans.
                    # assert len(visited & curr_set) == 0, "\nVisited span: %d-%d\n" % (start, end)
                    if len(visited & curr_set) != 0:
                        c.aligned = False
                    else:
                        assert end > start
                        visited |= curr_set

                        # Assume we only align concept to the last position of a span.
                        c.addAlignment(end-1)
                        c.aligned = True
                        c.setSpan((start, end))
                    # align = False
                    # TODO: change here to have alignments.
                    # for s in wids:
                    #     w_idx = int(s)
                    #     if w_idx not in visited:
                    #         c.addAlignment(w_idx)
                    #         align = True
                    #         visited.add(w_idx)
                    #         break
                    # if not align:
                    #     c.aligned = False

                # Processing outgoing relations.
                if outgoing_rels != "NONE":
                    out_rels = outgoing_rels.split("#")
                    for rel in out_rels:
                        fields = rel.split(":")
                        c.rels.append(fields[0])
                        c.tail_ids.append(int(fields[1]))

                # Processing incoming relations
                if parent_rels == "NONE":
                    graph.setRoot(concept_idx)
                    root_num += 1
                else:
                    in_rels = parent_rels.split("#")
                    for rel in in_rels:
                        fields = rel.split(":")
                        c.parent_rels.append(fields[0])
                        c.parent_ids.append(int(fields[1]))

                c.buildRelMap()
                graph.addConcept(c)
    return amr_graphs

def loadDataset(path):
    def ignore_instances(seqs, ignored_set):
        tok_seqs = [seq for (idx, seq) in enumerate(seqs) if idx not in ignored_set]
        return tok_seqs

    tok_file = os.path.join(path, "token")
    lemma_file = os.path.join(path, "lemma")
    pos_file = os.path.join(path, "pos")
    dep_file = os.path.join(path, "dep")
    amr_conll_file = os.path.join(path, "amr_conll")

    skipped_train_sentences = set([int(line.strip()) for line in open("./skipped_sentences")])

    tok_seqs = readToks(tok_file)
    lem_seqs = readToks(lemma_file)
    pos_seqs = readToks(pos_file)
    # dep_trees = loadDependency(dep_file, tok_seqs, True)
    dep_trees = loadDependency(dep_file, tok_seqs)
    amr_graphs = loadAMRConll(amr_conll_file)

    tok_seqs = ignore_instances(tok_seqs, skipped_train_sentences)
    lem_seqs = ignore_instances(lem_seqs, skipped_train_sentences)
    pos_seqs = ignore_instances(pos_seqs, skipped_train_sentences)
    dep_trees = ignore_instances(dep_trees, skipped_train_sentences)

    assert len(tok_seqs) == len(amr_graphs)

    dataset = Dataset(top_arc=50)
    dataset.setTok(tok_seqs)
    dataset.setLemma(lem_seqs)
    dataset.setPOS(pos_seqs)
    dataset.setDepTrees(dep_trees)
    dataset.setAMRGraphs(amr_graphs)

    return dataset

def saveCounter(counts, path):
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    with open(path, "w") as wf:
        for (item ,count) in sorted_items:
            print >>wf, "%s\t%d" % (item, count)
        wf.close()

