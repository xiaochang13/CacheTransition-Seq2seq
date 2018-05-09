from collections import deque
from dependency import *
from AMRGraph import *
# from postprocessing import AMR_seq
import copy
import utils
# from utils import Tokentype
class CacheConfiguration(object):
    def __init__(self, size, length, config=None):
        """
        :param size: number of elems of the fixed-sized cache.
        :param length: number of words in the buffer.
        """
        self.cache_size = size
        if config is None:
            self.stack = []
            self.buffer = deque(range(length))

            # Each cache elem is a (word idx, concept idx) pair.
            self.cache = [(-1, -1) for _ in range(size)]

            self.candidate = None   # A register for the newly generated vertex.

            self.hypothesis = AMRGraph()  # The AMR graph being built.
            self.gold = None  # The reference AMR graph.

            self.wordSeq, self.lemSeq, self.posSeq = [], [], []
            self.conceptSeq, self.conceptAlign, self.actionSeq = [], [], []
            self.categorySeq = []

            self.start_word = True # Whether start processing a new word.
            self.tree = DependencyTree()

            self.cand_vertex = None

            self.last_action = None
            self.pop_buff = True
            self.phase = utils.FeatureType.SHIFTPOP
            self.widTocid = {}
        else:
            self.stack = copy.copy(config.stack)
            self.buffer = copy.copy(config.buffer)
            self.cache = copy.copy(config.cache)
            self.wordSeq, self.lemSeq, self.posSeq = config.wordSeq, config.lemSeq, config.posSeq
            self.conceptSeq, self.categorySeq = config.conceptSeq, config.categorySeq
            self.tree, self.conceptAlign = config.tree, config.conceptAlign
            self.phase = config.phase
            self.hypothesis = copy.deepcopy(config.hypothesis)
            self.start_word = config.start_word
            self.cand_vertex = copy.copy(config.cand_vertex)
            self.pop_buff = config.pop_buff
            self.last_action = config.last_action
            self.widTocid = config.widTocid

    def setGold(self, graph_):
        self.gold = graph_

    def buildWordToConcept(self):
        assert len(self.conceptAlign) > 0
        for (cidx, aligned_widx) in enumerate(self.conceptAlign):
            if aligned_widx != -1:
                self.widTocid[aligned_widx] = cidx

    def isUnalign(self, idx):
        return self.conceptAlign[idx] == -1

    def getConcept(self, idx):
        return self.conceptSeq[idx]

    def getTokenFeats(self, idx, type):
        if idx < 0:
            return type.name + ":" + utils.NULL
        if (type == utils.Tokentype.CONCEPT or type == utils.Tokentype.CATEGORY) and idx >= len(self.conceptSeq):
            return type.name + ":" + utils.NULL
        if (type != utils.Tokentype.CONCEPT and type != utils.Tokentype.CATEGORY) and idx >= len(self.wordSeq):
            return type.name + ":" + utils.NULL
        prefix = type.name + ":"
        if type == utils.Tokentype.WORD:
            return prefix + self.wordSeq[idx]
        if type == utils.Tokentype.LEM:
            return prefix + self.lemSeq[idx]
        if type == utils.Tokentype.POS:
            return prefix + self.posSeq[idx]
        if type == utils.Tokentype.CONCEPT:
            return prefix + self.conceptSeq[idx]
        if type == utils.Tokentype.CATEGORY:
            return prefix + self.categorySeq[idx]

    def getArcFeats(self, concept_idx, idx, prefix, outgoing=True):
        arc_label = self.hypothesis.getConceptArc(concept_idx, idx, outgoing)
        return prefix + arc_label

    def getNumArcFeats(self, concept_idx, prefix, outgoing=True):
        arc_num = self.hypothesis.getConceptArcNum(concept_idx, outgoing)
        return "%s%d" % (prefix, arc_num)

    def getDepDistFeats(self, idx1, idx2):
        prefix = "DepDist="
        if idx1 < 0 or idx2 < 0:
            return prefix + utils.NULL
        dep_dist = self.tree.getDepDist(idx1, idx2)
        return "%s%d" % (prefix, dep_dist)

    def getDepLabelFeats(self, idx, feats, prefix="dep", k=3):
        dep_arcs = self.tree.getAllChildren(idx)
        for i in range(k):
            curr_dep = prefix + ":" + utils.NULL
            if i < len(dep_arcs):
                curr_dep = prefix + ":" + dep_arcs[i]
            feats.append(curr_dep)

    def getDepParentFeat(self, idx, feats, prefix="pdep"):
        parent_arc = self.tree.getLabel(idx)
        curr_dep = prefix + ":" + utils.NULL
        if parent_arc:
            curr_dep = prefix + ":" + parent_arc
        feats.append(curr_dep)

    def getTokenDistFeats(self, idx1, idx2, upper, prefix):
        if idx1 < 0 or idx2 < 0:
            return prefix + utils.NULL
        assert idx1 < idx2, "Left token index not smaller than right"
        token_dist = idx2 - idx1
        if token_dist > upper:
            token_dist = upper
        return "%s%d" % (prefix, token_dist)

    def getTokenTypeFeatures(self, word_idx, concept_idx, feats, prefix=""):
        word_repr = prefix + self.getTokenFeats(word_idx, utils.Tokentype.WORD)
        lem_repr = prefix + self.getTokenFeats(word_idx, utils.Tokentype.LEM)
        pos_repr = prefix + self.getTokenFeats(word_idx, utils.Tokentype.POS)
        concept_repr = prefix + self.getTokenFeats(concept_idx, utils.Tokentype.CONCEPT)
        category_repr = prefix + self.getTokenFeats(concept_idx, utils.Tokentype.CATEGORY)
        feats.append(word_repr)
        feats.append(lem_repr)
        feats.append(pos_repr)
        feats.append(concept_repr)
        feats.append(category_repr)

    def getConceptRelationFeatures(self, concept_idx, feats):
        first_concept_arc = self.getArcFeats(concept_idx, 0, "ARC=")
        second_concept_arc = self.getArcFeats(concept_idx, 1, "ARC=")
        parent_concept_arc = self.getArcFeats(concept_idx, 0, "PARC=", False)
        concept_parrel_num = self.getNumArcFeats(concept_idx, "NPARC=", False)
        feats.append(first_concept_arc)
        feats.append(second_concept_arc)
        feats.append(parent_concept_arc)
        feats.append(concept_parrel_num)

    def getCacheFeat(self, word_idx=-1, concept_idx=-1, idx=-1, uniform_arc=True, arc_label=False):
        if idx == -1:
            if uniform_arc or arc_label:
                return ["NONE"] * (utils.cache_feat_num + 8)
            return ["NONE"] * utils.cache_feat_num

        feats = []
        cache_word_idx, cache_concept_idx = self.getCache(idx)

        # Candidate token features.
        self.getTokenTypeFeatures(word_idx, concept_idx, feats)

        # Cache token features.
        self.getTokenTypeFeatures(cache_word_idx, cache_concept_idx, feats)

        # Distance features
        word_dist_repr = self.getTokenDistFeats(cache_word_idx, word_idx, 4, "WordDist=")
        concept_dist_repr = self.getTokenDistFeats(cache_concept_idx, concept_idx, 4, "ConceptDist=")
        dep_dist_repr = self.getDepDistFeats(cache_word_idx, word_idx)

        # Dependency label
        dep_label_repr = "DepLabel=" + self.tree.getDepLabel(cache_word_idx, word_idx)

        feats.append(word_dist_repr)
        feats.append(concept_dist_repr)
        feats.append(dep_dist_repr)
        feats.append(dep_label_repr)

        # If the arc label, then extract all the dependency label features.
        if arc_label:
            self.getDepLabelFeats(word_idx, feats, "dep", 3)
            self.getDepParentFeat(word_idx, feats, "pdep")
            self.getDepLabelFeats(cache_word_idx, feats, "cdep", 3)
            self.getDepParentFeat(cache_word_idx, feats, "cpdep")
        elif uniform_arc:
            feats += ["NONE"] * 8

        # Get arc information for the current concept
        self.getConceptRelationFeatures(concept_idx, feats)

        self.getConceptRelationFeatures(cache_concept_idx, feats)

        if arc_label or uniform_arc:
            assert len(feats) == utils.cache_feat_num + 8
        else:
            assert len(feats) == utils.cache_feat_num
        return feats

    def pushIDXFeatures(self, word_idx=-1, concept_idx=-1):
        if concept_idx == -1:
            return ["NONE"] * utils.pushidx_feat_num
        feats = []

        # Candidate vertex features.
        self.getTokenTypeFeatures(word_idx, concept_idx, feats)

        # Cache vertex features.
        for cache_idx in range(self.cache_size):
            cache_word_idx, cache_concept_idx = self.cache[cache_idx]
            prefix = "cache%d_" % cache_idx
            self.getTokenTypeFeatures(cache_word_idx, cache_concept_idx, feats, prefix)

        assert len(feats) == utils.pushidx_feat_num
        return feats

    def shiftPopFeatures(self, word_idx=-1, concept_idx=-1, active=False):
        if not active:
            return ["NONE"] * utils.shiftpop_feat_num
        feats = []
        rst_word_idx, rst_concept_idx = self.cache[self.cache_size-1]
        # Right most cache token features
        self.getTokenTypeFeatures(rst_word_idx, rst_concept_idx, feats, "rst_")

        # Buffer token features
        self.getTokenTypeFeatures(word_idx, concept_idx, feats, "buf_")

        # Then get the dependency links to right words
        dep_list = self.bufferDepConnections(rst_word_idx)

        dep_num = len(dep_list)
        if dep_num > 4:
            dep_num = 4
        dep_num_repr = "depnum=%d" % dep_num
        feats.append(dep_num_repr)
        for i in range(3):
            if i >= dep_num:
                feats.append("dep=" + utils.NULL)
            else:
                feats.append("dep=" + dep_list[i])

        assert len(feats) == utils.shiftpop_feat_num
        return feats

    def extractFeatures(self, feature_type, word_idx=-1, concept_idx=-1, cache_idx=-1, uniform_arc=False):
        phase_feat = "PHASE="
        if feature_type == utils.FeatureType.SHIFTPOP:
            shiftpop_feats = self.shiftPopFeatures(word_idx, concept_idx, True)
            phase_feat += "SHTPOP"
        else:
            shiftpop_feats = self.shiftPopFeatures()

        if feature_type == utils.FeatureType.ARCBINARY or feature_type == utils.FeatureType.ARCCONNECT:
            # arc_label = False
            if feature_type == utils.FeatureType.ARCBINARY:
                phase_feat += "ARCBINARY"
                if uniform_arc: # cache_feats + 8
                    cache_feats = self.getCacheFeat(word_idx, concept_idx, cache_idx,
                                                    uniform_arc=True, arc_label=False)
                else:
                    binary_feats = self.getCacheFeat(word_idx, concept_idx, cache_idx,
                                                    uniform_arc=False, arc_label=False)
                    label_feats = self.getCacheFeat(uniform_arc=False, arc_label=True)
                    cache_feats = binary_feats + label_feats
            else:
                phase_feat += "ARCLABEL"
                # arc_label = True
                if uniform_arc: # cache_feats + 8
                    cache_feats = self.getCacheFeat(word_idx, concept_idx, cache_idx,
                                                    uniform_arc=True, arc_label=True)
                else:
                    binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
                    label_feats = self.getCacheFeat(word_idx, concept_idx, cache_idx,
                                                    uniform_arc=False, arc_label=True)
                    cache_feats = binary_feats + label_feats

        else:
            if uniform_arc:
                cache_feats = self.getCacheFeat(uniform_arc=True)
            else:
                binary_feats = self.getCacheFeat(uniform_arc=False, arc_label=False)
                label_feats = self.getCacheFeat(uniform_arc=False, arc_label=True)
                cache_feats = binary_feats + label_feats

        if feature_type == utils.FeatureType.PUSHIDX:
            phase_feat += "PUSHIDX"
            pushidx_feats = self.pushIDXFeatures(word_idx, concept_idx)
        else:
            pushidx_feats = self.pushIDXFeatures()
        assert phase_feat != "PHASE="
        return [phase_feat] + shiftpop_feats + cache_feats + pushidx_feats

    def clearBuffer(self):
        self.buffer.clear()

    def popBuffer(self):
        pop_elem = self.buffer.popleft()
        return pop_elem

    def nextBufferElem(self):
        if len(self.buffer) == 0:
            return -1
        # assert len(self.buffer) > 0, "Fetch word from empty buffer."
        return self.buffer[0]

    def bufferSize(self):
        return len(self.buffer)

    def getCache(self, idx):
        if idx < 0 or idx > self.cache_size:
            return None
        return self.cache[idx]

    def getCacheConcept(self, cache_idx, type=utils.Tokentype.CONCEPT):
        _, concept_idx = self.getCache(cache_idx)
        if type == utils.Tokentype.CONCEPT:
            return self.conceptSeq[concept_idx]
        return self.categorySeq[concept_idx]

    def getConnectedArcs(self, cache_idx, left=True):
        _, concept_idx = self.getCache(cache_idx)
        outgoing_rels = self.hypothesis.outgoingArcs(concept_idx)
        if left:
            return set(["R-%s" % l for l in outgoing_rels if (l != "op" and l != "mod")])
        return set(["L-%s" % l for l in outgoing_rels if (l != "op" and l != "mod")])

    def moveToCache(self, elem):
        self.cache.append(elem)

    def popStack(self):
        stack_size = len(self.stack)
        if stack_size < 1:
            return None
        top_elem = self.stack.pop()
        return top_elem

    def getStack(self, idx):
        stack_size = len(self.stack)
        if idx < 0 or idx >= stack_size:
            return None
        return self.stack[idx]

    def stackSize(self):
        return len(self.stack)

    # Whether the next operation should be processing a new word
    # or a vertex from the cache.
    def needsPop(self):
        """
        Whether the next operation should be processing a new word
        or a vertex from the cache.
        :return:
        """
        last_cache_word, last_cache_concept = self.cache[self.cache_size-1]
        right_edges = self.gold.right_edges
        # print "Current last cache word %d, cache concept %d" % (last_cache_word, last_cache_concept)
        # ($, $) at last cache position.
        if last_cache_concept == -1:
            return False

        next_buffer_concept_idx = self.hypothesis.nextConceptIDX()
        num_concepts = self.gold.n
        if next_buffer_concept_idx >= num_concepts:
            return True

        if last_cache_concept not in right_edges:
            return True

        assert next_buffer_concept_idx > last_cache_concept and num_concepts > last_cache_concept

        return right_edges[last_cache_concept][-1] < next_buffer_concept_idx

    def shiftBuffer(self):
        if len(self.buffer) == 0:
            return False
        self.popBuffer()
        return True

    def rightmostCache(self):
        return self.cache[self.cache_size-1]

    def pop(self):
        stack_size = len(self.stack)
        if stack_size < 1:
            return False
        cache_idx, vertex = self.stack.pop()

        # Insert a vertex to a certain cache position.
        # Then pop the last cache vertex.
        self.cache.insert(cache_idx, vertex)
        self.cache.pop()
        return True

    def connectArc(self, cand_vertex_idx, cache_vertex_idx, direction, arc_label):
        """
        Make a directed labeled arc between a cache vertex and the candidate vertex.
        :param cand_vertex: the newly generated concept from the buffer.
        :param cache_vertex: a certain vertex in the cache.
        :param direction: the direction of the connected arc.
        :param arc_label: the label of the arc.
        :return: None
        """
        cand_c = self.hypothesis.concepts[cand_vertex_idx]
        cache_c = self.hypothesis.concepts[cache_vertex_idx]
        if direction == 0: # an L-edge, from candidate to cache.
            cand_c.tail_ids.append(cache_vertex_idx)
            cand_c.rels.append(arc_label)
            cache_c.parent_ids.append(cand_vertex_idx)
            cache_c.parent_rels.append(arc_label)
        else:
            cand_c.parent_ids.append(cache_vertex_idx)
            cand_c.parent_rels.append(arc_label)
            cache_c.tail_ids.append(cand_vertex_idx)
            cache_c.rels.append(arc_label)

    def bufferDepConnections(self, word_idx, thre=20):
        ret_set = set()
        start = self.tree.n
        if len(self.buffer) > 0:
            start = self.buffer[0]
        end = self.tree.n
        if end - start > thre:
            end = start + thre
        for idx in range(start, end):
            if self.tree.getHead(idx) == word_idx:
                ret_set.add("R="+self.tree.getLabel(idx))
            elif self.tree.getHead(word_idx) == idx:
                ret_set.add("L="+self.tree.getLabel(word_idx))
        return list(ret_set)

    def bufferDepConnectionNum(self, word_idx, thre=20):
        ret_set = self.bufferDepConnections(word_idx, thre)
        return len(ret_set)

    def pushToStack(self, cache_idx):
        """
        Push a certain cache vertex onto the stack.
        :param cache_idx:
        :return:
        """
        cache_word_idx, cache_concept_idx = self.cache[cache_idx]
        del self.cache[cache_idx]
        self.stack.append((cache_idx, (cache_word_idx, cache_concept_idx)))

    def __str__(self):
        ret = "Buffer: %s" % str(self.buffer)
        ret += "  Cache: %s" % str(self.cache)
        ret += "  Stack: %s" % str(self.stack)
        return ret

    # dump the constructed hypothesis in current configuration.
    def toString(self, check=True):
        if check:
            assert len(self.conceptSeq) == self.hypothesis.counter
            assert len(self.conceptSeq) == len(self.hypothesis.concepts)

        concept_line_reprs = self.toConll()
        concept_line_reprs = ["%d\t%s\t%s\t%s" % (cidx, concept_l, rel_str, par_str) for
                              (cidx, (concept_l, rel_str, par_str)) in enumerate(concept_line_reprs)]
        return "\n".join(concept_line_reprs)

    def toConll(self):
        concept_line_reprs = []
        for (cidx, concept_l) in enumerate(self.conceptSeq):
            curr_concept = self.hypothesis.getConcept(cidx)
            curr_concept.rebuild_ops()
            assert curr_concept.getValue() == concept_l

            # Then relations
            rel_str = "#".join(["%s:%d" % (r_l, curr_concept.tail_ids[r_idx]) for (r_idx, r_l)
                                in enumerate(curr_concept.rels)])
            if not rel_str:
                rel_str = "NONE"

            # Then parent relations
            par_str = "#".join(["%s:%d" % (p_l, curr_concept.parent_ids[p_idx]) for (p_idx, p_l)
                                in enumerate(curr_concept.parent_rels)])
            if not par_str:
                par_str = "NONE"

            # concept_repr = "%d\t%s\t%s\t%s" % (cidx, concept_l, rel_str, par_str)
            concept_line_reprs.append((concept_l, rel_str, par_str))
        return concept_line_reprs

    # reformat to AMR graph
    # def toAMR(self, category_seq, align_map_seq):
    #     return

