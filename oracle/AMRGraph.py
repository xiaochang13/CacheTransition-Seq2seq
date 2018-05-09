from collections import defaultdict
import utils
class ConceptLabel(object):
    def __init__(self, label=None):
        self.value = label
        self.alignments = []
        self.rels = []
        self.rel_map = {}
        self.par_rel_map = {}

        self.tail_ids = []
        self.parent_rels = []
        self.parent_ids = []
        self.aligned = False
        self.isVar = False
        self.category = None
        self.map_info = None
        self.span = None

    def setVarType(self, v):
        self.isVar = v

    def setSpan(self, sp):
        self.span = sp

    def getRelStr(self, idx):
        assert idx in self.rel_map
        return self.rel_map[idx]

    def buildRelMap(self):
        n_rels = len(self.rels)
        for i in range(n_rels):
            curr_idx = self.tail_ids[i]
            self.rel_map[curr_idx] = self.rels[i]
        n_rels = len(self.parent_rels)
        for i in range(n_rels):
            curr_idx = self.parent_ids[i]
            self.par_rel_map[curr_idx] = self.parent_rels[i]

    def concept_repr(self, graph):
        outgoing = []
        for (idx, l) in enumerate(self.rels):
            tail = self.tail_ids[idx]
            outgoing.append("%s:%s" % (l, graph.concepts[tail].value))
        return "%s %s" % (self.value, " ".join(sorted(outgoing)))


    def setValue(self, s):
        self.value = s

    def getValue(self):
        return self.value

    def rebuild_ops(self):
        """Rebuild the op relations."""
        op_tail_idxs = [tail_idx for (idx, tail_idx) in enumerate(self.tail_ids) if self.rels[idx] == "op"]
        if op_tail_idxs:
            sorted_ops = sorted(op_tail_idxs)
            idx_to_opstr = {}
            for (idx, tail_idx) in enumerate(sorted_ops, 1):
                idx_to_opstr[tail_idx] = "op%d" % idx
            for (idx, tail_idx) in enumerate(self.tail_ids):
                if tail_idx in idx_to_opstr:
                    self.rels[idx] = idx_to_opstr[tail_idx]

    def addAlignment(self, word_idx):
        self.alignments.append(word_idx)
        assert len(self.alignments) == 1

    def getArc(self, k):
        if k >= len(self.rels) or k < 0:
            return utils.NULL
        return self.rels[k]

    def getParentArc(self, k):
        if k >= len(self.parent_rels) or k < 0:
            return utils.NULL
        return self.parent_rels[k]

class AMRGraph(object):
    def __init__(self):
        self.n = 0
        self.concepts = []
        self.counter = 0
        self.widToConceptID = {}
        self.cidToSpan = {}
        self.right_edges = defaultdict(list)
        self.toks = None
        self.headToTail = defaultdict(set)

    def compare(self, other):
        # return False
        assert isinstance(other, self.__class__), "Comparing a graph to another class: %s" % other.__class__
        if self.n != other.n:
            return False
        for i in range(self.n):
            first_concept = self.concepts[i]
            second_concept = other.concepts[i]
            first_repr = first_concept.concept_repr(self)
            second_repr = second_concept.concept_repr(self)
            if first_repr != second_repr:
                # print "Inconsistent concept at %d: %s vs %s" % (i, first_repr, second_repr)
                return False
        return True

    def incomingArcs(self, v):
        if v < 0:
            return None
        concept = self.concepts[v]
        return concept.parent_rels

    def outgoingArcs(self, v):
        if v < 0:
            return []
        concept = self.concepts[v]
        return concept.rels

    def setRoot(self, v):
        self.root = v

    def count(self):
        self.counter += 1

    def isAligned(self, v):
        if v >= len(self.concepts):
            return False
        return self.concepts[v].aligned

    def initTokens(self, toks):
        self.toks = toks

    def initLemma(self, lems):
        self.lemmas = lems

    def nextConceptIDX(self):
        return self.counter

    def conceptLabel(self, idx):
        if idx < 0:
            return utils.NULL
        concept = self.concepts[idx]
        return concept.getValue()

    def getConcept(self, idx):
        return self.concepts[idx]

    def getConceptArc(self, concept_idx, rel_idx, outgoing=True):
        if concept_idx == -1:
            return utils.NULL
        concept = self.getConcept(concept_idx)
        if outgoing:
            return concept.getArc(rel_idx)
        return concept.getParentArc(rel_idx)

    def getConceptArcNum(self, concept_idx, outgoing=True):
        if concept_idx == -1:
            return 0
        concept = self.getConcept(concept_idx)
        if outgoing:
            return len(concept.rels)
        return len(concept.parent_rels)

    def getConceptSeq(self):
        return [concept.getValue() for concept in self.concepts]

    def getCategorySeq(self):
        return [concept.category for concept in self.concepts]

    def getMapInfoSeq(self):
        return [concept.map_info for concept in self.concepts]

    def buildWordToConceptIDX(self):
        """
        Assume multiple-to-one alignments
        :return:
        """
        for (i, concept) in enumerate(self.concepts):
            for word_idx in concept.alignments:
                self.widToConceptID[word_idx] = i
            span = concept.span
            if span is not None:
                self.cidToSpan[i] = span

    def __str__(self):
        ret = ""
        for (i, concept) in enumerate(self.concepts):
            ret += ("Current concept %d: %s\n" % (i, concept.getValue()))
            concept.buildRelMap()
            rel_repr = ""
            for tail_v in concept.tail_ids:
                rel_repr += concept.rel_map[tail_v]
                rel_repr += (":" + self.concepts[tail_v].getValue()+ " ")
            ret += ("Tail concepts: %s\n" % rel_repr)
        return ret

    def buildEdgeMap(self):
        right_edges_list = defaultdict(list)
        for (i, concept) in enumerate(self.concepts):
            for tail_v in concept.tail_ids:
                self.headToTail[i].add(tail_v)
                if tail_v > i:
                    right_edges_list[i].append(tail_v)
                elif tail_v < i:
                    right_edges_list[tail_v].append(i)
        for left_idx in right_edges_list:
            sorted_right_list = sorted(right_edges_list[left_idx])
            assert sorted_right_list[0] > left_idx
            self.right_edges[left_idx] = sorted_right_list

    def addConcept(self, c):
        self.concepts.append(c)
        self.n += 1
