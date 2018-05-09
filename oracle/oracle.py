from collections import defaultdict
from ioutil import *
from oracle_data import *
from cacheTransition import CacheTransition
from cacheConfiguration import CacheConfiguration
import time
import argparse
NULL = "-NULL-"
UNKNOWN = "-UNK-"
class CacheTransitionParser(object):
    def __init__(self, size):
        self.cache_size = size
        self.connectWordDistToCount = defaultdict(int)
        self.nonConnectDistToCount = defaultdict(int)
        self.depConnectDistToCount = defaultdict(int)
        self.depNonConnectDistToCount = defaultdict(int)

        self.wordConceptCounts = defaultdict(int)
        self.lemmaConceptCounts = defaultdict(int)
        self.mleConceptID = defaultdict(int)
        self.lemMLEConceptID = defaultdict(int)
        self.conceptIDDict = defaultdict(int)
        self.unalignedSet = defaultdict(int)

        self.wordIDs = {}
        self.lemmaIDs = {}
        self.conceptIDs = {}
        self.posIDs = {}
        self.depIDs = {}
        self.arcIDs = {}

    def OracleExtraction(self, data_dir, output_dir, oracle_type=utils.OracleType.AAAI, decode=False, uniform=False):
        print "Input data directory:", data_dir
        print "Output directory:", output_dir
        os.system("mkdir -p %s" % output_dir)

        oracle_output = os.path.join(output_dir, "oracle_output.txt")
        if decode:
            json_output = os.path.join(output_dir, "oracle_decode.json")
        else:
            json_output = os.path.join(output_dir, "oracle_examples.json")

        data_set = loadDataset(data_dir)
        oracle_set = OracleData()

        max_oracle_length = 0.0
        total_oracle_length = 0.0

        data_num = data_set.dataSize()
        data_set.genDictionaries()
        # data_set.genConceptMap()

        cache_transition = CacheTransition(
            self.cache_size, oracle_type, data_set.known_concepts, data_set.known_rels,
            data_set.unaligned_set)

        utils.pushidx_feat_num = (1+self.cache_size) * 5

        cache_transition.makeTransitions()

        push_actions, arc_binary_actions, arc_label_actions = defaultdict(int), defaultdict(int), defaultdict(int)
        num_pop_actions = 0
        num_shift_actions = 0
        print "Oracle type: %s" % oracle_type.name

        feat_dim = -1

        success_num = 0.0

        # skipped_train_sentences = set([int(line.strip() for line in open("./skipped_sentences"))])
        # amr_indices = 0

        # Run oracle on the training data.
        with open(oracle_output, 'w') as oracle_wf:
            for sent_idx in range(data_num):

                training_instance = data_set.getInstance(sent_idx)
                tok_seq, lem_seq, pos_seq = training_instance[0], training_instance[1], training_instance[2]
                dep_tree, amr_graph = training_instance[3], training_instance[4]
                concept_seq = amr_graph.getConceptSeq()
                category_seq = amr_graph.getCategorySeq()
                map_info_seq = amr_graph.getMapInfoSeq()

                amr_graph.initTokens(tok_seq)
                amr_graph.initLemma(lem_seq)

                # if decode and sent_idx >= 10: # Only use 10 examples for decoding.
                #     break

                # if sent_idx >= 10:
                #     break
                # if sent_idx != 4766:
                #     continue

                # TODO: this should be the length of the concept sequence.
                length = len(tok_seq)

                c = CacheConfiguration(self.cache_size, length)
                c.wordSeq, c.lemSeq, c.posSeq = tok_seq, lem_seq, pos_seq

                c.tree = dep_tree
                c.conceptSeq = concept_seq
                c.categorySeq = category_seq
                assert len(concept_seq) == len(category_seq)

                c.setGold(amr_graph)

                feat_seq = []

                word_align = []
                concept_align = []

                start_time = time.time()

                oracle_seq = []
                succeed = True

                concept_to_word = []

                concept_idx = 0

                # print sent_idx, amr_graph.widToConceptID
                # print tok_seq
                # print concept_seq
                # print amr_graph.cidToSpan

                while not cache_transition.isTerminal(c):
                    oracle_action = cache_transition.getOracleAction(c)
                    # print oracle_action

                    if time.time() - start_time > 4.0:
                        print >> sys.stderr, "Overtime sentence #%d" % sent_idx
                        print >> sys.stderr, "Sentence: %s" % " ".join(tok_seq)
                        succeed = False
                        break

                    word_idx = c.nextBufferElem()

                    if "ARC" in oracle_action:
                        parts = oracle_action.split(":")
                        arc_decisions = parts[1].split("#")

                        num_connect = c.cache_size
                        if oracle_type == utils.OracleType.CL:
                            num_connect -= 1
                            temp_word_idx = word_idx
                            if c.pop_buff:
                                if temp_word_idx == -1:
                                    temp_word_idx = len(tok_seq) - 1
                                else:
                                    temp_word_idx -= 1

                        word_idx, curr_concept_idx = c.getCache(num_connect)
                        # if curr_concept_idx in amr_graph.cidToSpan:
                        #     temp_word_idx = amr_graph.cidToSpan[curr_concept_idx][1] - 1

                        assert curr_concept_idx == concept_idx

                        # for cache_idx in range(num_connect):
                        for cache_idx in range(num_connect-1, -1, -1):
                            arc_label = arc_decisions[cache_idx]
                            curr_arc_action = "ARC%d:%s" % (cache_idx, arc_label)

                            if arc_label == "O":
                                arc_binary_actions["O"] += 1
                                feat_seq.append(c.extractFeatures(utils.FeatureType.ARCBINARY,
                                                                  word_idx, concept_idx, cache_idx, uniform_arc=uniform))
                                oracle_seq.append("NOARC")
                                word_align.append(temp_word_idx)
                                concept_align.append(concept_idx)
                            else:
                                arc_binary_actions["Y"] += 1
                                feat_seq.append(c.extractFeatures(utils.FeatureType.ARCBINARY,
                                                                  word_idx, concept_idx, cache_idx, uniform_arc=uniform))
                                oracle_seq.append("ARC")
                                word_align.append(temp_word_idx)
                                concept_align.append(concept_idx)

                                feat_seq.append(c.extractFeatures(utils.FeatureType.ARCCONNECT,
                                                                  word_idx, concept_idx, cache_idx, uniform_arc=uniform))
                                arc_label_actions[arc_label] += 1
                                oracle_seq.append(arc_label)
                                word_align.append(temp_word_idx)
                                concept_align.append(concept_idx)

                            cache_transition.apply(c, curr_arc_action)
                        if oracle_type == utils.OracleType.CL:
                            c.start_word = True
                            concept_idx += 1

                    else:
                        #Currently assume vertex generated separately.
                        if oracle_action == "conID:-NULL-":
                            assert c.phase == utils.FeatureType.SHIFTPOP
                            cache_transition.apply(c, oracle_action)
                            continue

                        if word_idx == -1: # The last few concepts can be unaligned.
                            word_idx = len(tok_seq)
                        assert concept_idx >= 0
                        # if concept_idx in amr_graph.cidToSpan:
                        #     word_idx = amr_graph.cidToSpan[concept_idx][1] - 1
                        word_align.append(word_idx)

                        if "conGen" not in oracle_action and "conID" not in oracle_action:
                            oracle_seq.append(oracle_action)
                            concept_align.append(concept_idx)
                            if oracle_action == "POP":
                                feat_seq.append(c.extractFeatures(utils.FeatureType.SHIFTPOP,
                                                                  word_idx, concept_idx, uniform_arc=uniform))
                                num_pop_actions += 1
                            else:
                                feat_seq.append(c.extractFeatures(utils.FeatureType.PUSHIDX,
                                                                  word_idx, concept_idx, uniform_arc=uniform))
                                push_actions[oracle_action] += 1
                                if oracle_type == utils.OracleType.AAAI:
                                    concept_idx += 1
                        elif "NULL" not in oracle_action:
                            oracle_seq.append("SHIFT")
                            num_shift_actions += 1
                            concept_align.append(concept_idx)
                            assert len(concept_to_word) == concept_idx, "%d : %d" % (concept_idx, len(concept_to_word))
                            if "conGen" in oracle_action:
                                concept_to_word.append(-1)
                            else:
                                aligned_word_idx = c.nextBufferElem()
                                assert aligned_word_idx != -1
                                concept_to_word.append(aligned_word_idx)

                            feat_seq.append(c.extractFeatures(utils.FeatureType.SHIFTPOP,
                                                              word_idx, concept_idx, uniform_arc=uniform))
                            if feat_dim == -1:
                                feat_dim = len(feat_seq[-1])
                        cache_transition.apply(c, oracle_action)
                        if "PUSH" in oracle_action and oracle_type == utils.OracleType.AAAI:
                            c.start_word = True

                if succeed:
                    assert len(feat_seq) == len(oracle_seq)
                    assert len(word_align) == len(concept_align)
                    try:
                        assert len(oracle_seq) > 0, concept_seq
                    except:
                        print concept_seq, c.wordSeq, c.conceptSeq
                        print sent_idx
                        sys.exit(1)
                    if not decode and (not oracle_seq):
                        continue
                    if concept_align:
                        assert concept_align[-1] >= 0 and concept_align[-1] <= len(concept_seq), concept_seq
                        assert word_align[-1] >= 0 and word_align[-1] <= len(tok_seq), tok_seq
                        assert len(concept_to_word) == len(concept_seq)

                    if decode:
                        feat_seq = [["" for _ in feats] for feats in feat_seq]
                        word_align = [-1 for _ in word_align]
                        concept_align = [-1 for _ in concept_align]

                    oracle_example = OracleExample(tok_seq, lem_seq, pos_seq, concept_seq, category_seq, map_info_seq,
                                                   feat_seq, oracle_seq, word_align, concept_align, concept_to_word)

                    # print "feature dimension: %d" % feat_dim
                    for feats in feat_seq:
                        assert len(feats) == feat_dim, "Feature dimensions not consistent: %s" % str(feats)
                    total_oracle_length += len(oracle_seq)
                    if len(oracle_seq) > max_oracle_length:
                        max_oracle_length = len(oracle_seq)
                    oracle_set.addExample(oracle_example)
                    print>> oracle_wf, "Sentence #%d: %s" % (sent_idx, " ".join(tok_seq))
                    print>> oracle_wf, "AMR graph:\n%s" %  str(amr_graph).strip()
                    print>> oracle_wf, "Constructed AMR graph:\n%s" % str(c.hypothesis).strip()

                    oracle_align = " ".join(["%s,%d,%d" % (w_a, c_a, o_s) for (w_a, c_a, o_s)
                                             in zip(oracle_seq, word_align, concept_align)])
                    print>> oracle_wf, "Oracle sequence: %s" % oracle_align
                    curr_feat_str = " ".join(["_#_".join(feats) for feats in feat_seq])
                    for feat_str in curr_feat_str.split():
                        assert len(feat_str.split("_#_")) == feat_dim, "%s\n%s\n" % (" ".join(tok_seq), feat_str)
                    # if len(feat_seq) < 40:
                    #     for (idx, feats) in enumerate(feat_seq):
                    #         curr_action = oracle_seq[idx]
                    #         print>> oracle_wf, "Action:%s  Features:%s" % (curr_action, "_#_".join(feats))

                    if c.gold.compare(c.hypothesis):
                        success_num += 1

                else:
                    print "Failed sentence %d" % sent_idx
                    print " ".join(tok_seq)
                    print str(amr_graph)
                    print "Oracle sequence so far: %s\n" % " ".join(oracle_seq)

            oracle_wf.close()

        if not decode:
            arc_binary_path = os.path.join(output_dir, "arc_binary_actions.txt")
            saveCounter(arc_binary_actions, arc_binary_path)
            arc_label_path = os.path.join(output_dir, "arc_label_actions.txt")
            saveCounter(arc_label_actions, arc_label_path)
            pushidx_path = os.path.join(output_dir, "pushidx_actions.txt")
            saveCounter(push_actions, pushidx_path)
        print "A total of %d shift actions" % num_shift_actions
        print "A total of %d pop actions" % num_pop_actions
        print "Maximum oracle sequence length is", max_oracle_length
        print "Average oracle sequence length is", total_oracle_length/data_num
        print "Oracle success ratio is", success_num/ data_num
        print "feature dimensions:", feat_dim
        oracle_set.toJSON(json_output)

    def isPredicate(self, s):
        length = len(s)
        if length < 3 or not (s[length-3] == '-'):
            return False
        last_char = s[-1]
        return last_char >= '0' and last_char <= '9'

    def conceptCategory(self, s, conceptArcChoices):
        """
        To be implemented!
        :param s:
        :param conceptArcChoices:
        :return:
        """
        if s in conceptArcChoices:
            return s
        if s == "NE" or "NE_" in s:
            return "NE"
        return "OTHER"

    def getWordID(self, s):
        if s in self.wordIDs:
            return self.wordIDs[s]
        return UNKNOWN

    def getLemmaID(self, s):
        if s in self.lemmaIDs:
            return self.lemmaIDs[s]
        return UNKNOWN

    def getConceptID(self, s):
        if s in self.conceptIDs:
            return self.conceptIDs[s]
        return UNKNOWN

    def getPOSID(self, s):
        if s in self.posIDs:
            return self.posIDs[s]
        return UNKNOWN

    def getDepID(self, s):
        if s in self.depIDs:
            return self.depIDs[s]
        return UNKNOWN

    def getArcID(self, s):
        if s in self.arcIDs:
            return self.arcIDs[s]
        return UNKNOWN

    def actionType(self, s):
        if s == "POP" or "conID" in s or "conGen" in s or "conEMP" in s:
            return 0
        if "ARC" in s:
            return 1
        else:
            return 2

    def generateTrainingExamples(self):
        return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_dir", type=str, help="The data directory for the input files.")
    argparser.add_argument("--output_dir", type=str, help="The directory for the output files.")
    argparser.add_argument("--oracle_type", type=str, help="The oracle type.")
    argparser.add_argument("--cache_size", type=int, default=6, help="Fixed cache size for the transition system.")
    argparser.add_argument("--decode", action="store_true", help="if to extract decoding examples.")
    argparser.add_argument("--uniform", action="store_true", help="if to use uniform arc features.")

    args = argparser.parse_args()
    parser = CacheTransitionParser(args.cache_size)
    if args.oracle_type == "aaai":
        oracle_type = utils.OracleType.AAAI
    else:
        oracle_type = utils.OracleType.CL
    parser.OracleExtraction(args.data_dir, args.output_dir, oracle_type, args.decode, args.uniform)
