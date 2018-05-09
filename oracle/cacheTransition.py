import sys, os
from cacheConfiguration import *
NULL = "-NULL-"
UNKNOWN = "-UNK-"
class CacheTransition(object):
    def __init__(self, size, oracle_type_, con_labs_=None, arc_labs_=None, unalign_labs_=None):
        self.con_labs = con_labs_
        self.arc_labs = arc_labs_
        self.unalign_labs = unalign_labs_
        self.cache_size = size
        self.push_actions = []
        self.conID_actions = []
        self.arc_actions = []
        self.push_actionIDs = dict()
        self.conID_actionIDs = dict()
        self.arc_actionIDs = dict()

        self.ne_choices = set()
        self.constInArc_choices = None
        self.constOutArc_choices = None

        self.outgo_arcChoices = None
        self.income_arcChoices = None
        self.default_arcChoices = None

        self.oracle_type = oracle_type_

        self.push_action_set = None
        self.arcbinary_action_set = None
        self.arclabel_action_set = None
        self.shiftpop_action_set = None

    def actionNum(self, action_type):
        if action_type == 0:
            return len(self.conID_actions)
        elif action_type == 1:
            return len(self.arc_actions)
        else:
            return len(self.push_actions)

    def constructActionSet(self, left_concept, right_concept):
        def is_const(s):
            const_set = set(['interrogative', 'imperative', 'expressive', '-'])
            return s in const_set or s == "NUMBER"
        label_candidates = set()
        if left_concept in self.outgo_arcChoices and right_concept in self.income_arcChoices:
            label_candidates |= set(["R-%s" % l for l in (
                self.outgo_arcChoices[left_concept] & self.income_arcChoices[right_concept])])
        if right_concept in self.outgo_arcChoices and left_concept in self.income_arcChoices:
            label_candidates |= set(["L-%s" % l for l in (
                self.outgo_arcChoices[right_concept] & self.income_arcChoices[left_concept])])
        if len(label_candidates) == 0:
            if is_const(left_concept):
                return set(["L-%s" % l for l in self.income_arcChoices[left_concept]])
            if is_const(right_concept):
                return set(["R-%s" % l for l in self.income_arcChoices[right_concept]])
            print>> sys.stderr, "Left concept: %s, Right concept: %s" % (left_concept, right_concept)
            return self.default_arcChoices
        return label_candidates

    def transitionActions(self, action_type):
        if action_type == 0:
            return self.conID_actions
        elif action_type == 1:
            return self.arc_actions
        return self.push_actions

    def actionStr(self, action_type, action_idx):
        if action_type == 0:
            return self.conID_actions[action_idx]
        elif action_type == 1:
            return self.arc_actions[action_idx]
        return self.push_actions[action_idx]

    def isTerminal(self, c):
        return c.stackSize() == 0 and c.bufferSize() == 0 and c.hypothesis.counter == len(c.conceptSeq)

    def makeTransitions(self):
        """
        Construct each type of transition actions.
        :param concept_labels:
        :param arc_labels:
        :return:
        """
        for l in self.con_labs:
            curr_action = "conID:" + l
            self.conID_actionIDs[curr_action] = len(self.conID_actions)
            self.conID_actions.append(curr_action)
            if l == "NE" or "NE_" in l:
                self.ne_choices.add(self.conID_actionIDs[curr_action])

        null_action = "conID:" + NULL
        self.conID_actionIDs[null_action] = len(self.conID_actions)
        self.conID_actions.append(null_action)

        unk_action = "conID" + UNKNOWN
        self.conID_actionIDs[unk_action] = len(self.conID_actions)
        self.conID_actions.append(unk_action)

        for l in self.arc_labs:
            if l == NULL or l == UNKNOWN:
                continue
            curr_action = "L-" + l
            self.arc_actionIDs[curr_action] = len(self.arc_actions)
            self.arc_actions.append(curr_action)

            curr_action = "R-" + l
            self.arc_actionIDs[curr_action] = len(self.arc_actions)
            self.arc_actions.append(curr_action)

        for i in range(self.cache_size):
            curr_action = "PUSH:%d" % i
            self.push_actionIDs[curr_action] = len(self.push_actions)
            self.push_actions.append(curr_action)

    def getActionIDX(self, action, type):
        if type == 0:
            return self.conID_actionIDs[action]
        elif type == 1:
            return self.arc_actionIDs[action]
        elif type == 2:
            return self.push_actionIDs[action]
        return -1

    # Currently only support CL oracle.
    def canApply(self, c, action, concept_idx=-1, use_refined=False, cache_idx=-1):
        if c.phase == utils.FeatureType.SHIFTPOP: # Can only shift or pop.
            if action not in self.shiftpop_action_set:
                return False
            if concept_idx >= len(c.conceptSeq):
                return action == "POP"
            if c.stackSize() == 0:
                assert concept_idx < len(c.conceptSeq)
                return action == "SHIFT"
            return True
        if c.phase == utils.FeatureType.PUSHIDX:
            return action in self.push_action_set
        if c.phase == utils.FeatureType.ARCBINARY:
            _, left_concept_idx = c.getCache(cache_idx)
            if left_concept_idx == -1:
                return action == "NOARC"

            return action in self.arcbinary_action_set
        if use_refined:
            left_concept = c.getCacheConcept(cache_idx, utils.Tokentype.CATEGORY)
            right_concept = c.getCacheConcept(self.cache_size-1, utils.Tokentype.CATEGORY)
            arc_choices = self.constructActionSet(left_concept, right_concept)
            connected = c.getConnectedArcs(cache_idx) | c.getConnectedArcs(self.cache_size-1, False)
            return action in arc_choices and action not in connected
        return action in self.arclabel_action_set

    def apply(self, c, action):
        if action == "POP": # POP action
            if not c.pop():
                assert False, "Pop from empty stack!"
            c.start_word = True
            c.phase = utils.FeatureType.SHIFTPOP
        elif action == "conID:-NULL-":
            c.popBuffer()
            c.start_word = True
            c.phase = utils.FeatureType.SHIFTPOP
        elif "conGen" in action or "conID" in action:
            l = action.split(":")[1]
            new_concept = ConceptLabel(l)
            new_concept_idx = c.hypothesis.nextConceptIDX()
            c.hypothesis.addConcept(new_concept)
            if self.oracle_type == utils.OracleType.AAAI:
                c.hypothesis.count()
                c.phase = utils.FeatureType.ARCBINARY
            else:
                c.phase = utils.FeatureType.PUSHIDX
            c.start_word = False # Have generated a candidate vertex.

            if "conID" in action:
                next_word = c.nextBufferElem()
                c.cand_vertex = (next_word, new_concept_idx)
                # c.popBuffer()
                c.pop_buff = True
            else:
                c.cand_vertex = (-1, new_concept_idx)
                c.pop_buff = False
            c.last_action = "conID"
        elif "ARC" in action:
            assert not c.start_word
            parts = action.split(":")
            cache_idx = int(parts[0][3:])
            arc_label = parts[1]
            c.last_action = "ARC"
            _, curr_cache_concept_idx = c.getCache(cache_idx)
            c.phase = utils.FeatureType.ARCBINARY
            if self.oracle_type == utils.OracleType.AAAI:
                if cache_idx == 0:
                    c.phase = utils.FeatureType.PUSHIDX
                # if cache_idx == self.cache_size - 1:
                #     c.phase = utils.FeatureType.PUSHIDX
            else:
                # if cache_idx == self.cache_size - 2:
                if cache_idx == 0:
                    c.phase = utils.FeatureType.SHIFTPOP
            if arc_label == "O":
                return
            elif arc_label[0] == "L":
                c.connectArc(c.cand_vertex[1], curr_cache_concept_idx, 0, arc_label[2:])
            else:
                c.connectArc(c.cand_vertex[1], curr_cache_concept_idx, 1, arc_label[2:])
            # c.start_word = False
        else: # PUSHx
            assert not c.start_word
            cache_idx = int(action.split(":")[1])
            c.pushToStack(cache_idx)
            c.moveToCache(c.cand_vertex)

            # For CL oracle, only move to next concept when the
            if self.oracle_type == utils.OracleType.CL:
                c.hypothesis.count()
                c.phase = utils.FeatureType.ARCBINARY
            else:
                c.phase = utils.FeatureType.SHIFTPOP
            if c.pop_buff:
                c.popBuffer()

    def chooseVertex(self, c, right_edges):
        max_dist = -1
        max_idx = -1
        next_buffer_concept_idx = c.hypothesis.nextConceptIDX()
        for cache_idx in range(c.cache_size):
            cache_word_idx, cache_concept_idx = c.getCache(cache_idx)
            curr_dist = 1000
            if cache_concept_idx == -1:
                return cache_idx

            # If no connection to any future vertices.
            if cache_concept_idx not in right_edges or right_edges[cache_concept_idx][-1] < next_buffer_concept_idx:
                return cache_idx

            for connect_idx in right_edges[cache_concept_idx]:
                if connect_idx >= next_buffer_concept_idx:
                    curr_dist = connect_idx
                    break

            assert curr_dist != 1000

            if curr_dist > max_dist:
                max_idx = cache_idx
                max_dist = curr_dist
        return max_idx

    def getOracleAction(self, c):
        gold_graph = c.gold
        headToTail = gold_graph.headToTail
        widToConceptID = gold_graph.widToConceptID
        # cidToWordSpan = gold_graph.cidToSpan
        right_edges = gold_graph.right_edges
        if c.start_word:
            word_idx = c.nextBufferElem()
            if word_idx != -1 and word_idx not in widToConceptID:
                c.last_action = "emp"
                return "conID:-NULL-"
            if c.needsPop():
                return "POP"
            # assert word_idx != -1

            hypo_graph = c.hypothesis
            next_concept_idx = hypo_graph.nextConceptIDX()
            concept_size = len(gold_graph.concepts) # The total number of concepts.
            c.last_action = "conID"
            if next_concept_idx < concept_size and (not gold_graph.isAligned(next_concept_idx)):
                unaligned_label = gold_graph.conceptLabel(next_concept_idx)
                return "conGen:" + unaligned_label
            assert word_idx in widToConceptID
            concept_idx = widToConceptID[word_idx]
            action = "conID:" + gold_graph.conceptLabel(concept_idx)
            if action not in self.conID_actionIDs:
                action = "conID:" + UNKNOWN
            return action

        if (self.oracle_type == utils.OracleType.AAAI and c.last_action == "conID") or (
                    self.oracle_type == utils.OracleType.CL and c.last_action == "PUSH"):
            arcs = []
            c.last_action = "ARC"
            num_connect = c.cache_size

            # If CL oracle, then connect from the rightmost.
            if self.oracle_type == utils.OracleType.CL:
                num_connect -= 1
                _, cand_concept_idx = c.getCache(num_connect)
            else:  # Otherwise AAAI oracle, from the generated concept.
                cand_concept_idx = c.cand_vertex[1]

            for cache_idx in range(num_connect):
                cache_word_idx, cache_concept_idx = c.getCache(cache_idx)
                if cache_concept_idx == -1:
                    arcs.append("O")
                    continue

                # Compute the directed arc label.
                if cache_concept_idx in headToTail[cand_concept_idx]:
                    cand_concept = gold_graph.getConcept(cand_concept_idx)
                    arc_label = "L-" + cand_concept.getRelStr(cache_concept_idx)
                    if arc_label not in self.arc_actionIDs:
                        # print >> sys.stderr, "Unseen:"+ arc_label
                        arc_label = "L-" + UNKNOWN
                    arcs.append(arc_label)
                elif cand_concept_idx in headToTail[cache_concept_idx]:
                    cache_concept = gold_graph.getConcept(cache_concept_idx)
                    arc_label = "R-" + cache_concept.getRelStr(cand_concept_idx)
                    if arc_label not in self.arc_actionIDs:
                        # print >> sys.stderr, "Unseen:" + arc_label
                        arc_label = "R-" + UNKNOWN
                    arcs.append(arc_label)
                else:
                    arcs.append("O")
            return "ARC:" + "#".join(arcs)
        if (self.oracle_type == utils.OracleType.AAAI and "ARC" in c.last_action) or (
                self.oracle_type == utils.OracleType.CL and c.last_action == "conID"):
            c.last_action = "PUSH"
            cache_idx = self.chooseVertex(c, right_edges)
            return "PUSH:%d" % cache_idx
        print>> sys.stderr, "Unable to proceed from last action:" + c.last_action
        sys.exit(1)
