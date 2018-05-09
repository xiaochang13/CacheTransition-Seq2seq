import json
import re, os, sys
import oracle.ioutil
import numpy as np
import random
import padding_utils
from sent_utils import AnonSentence

def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines


def read_all_GenerationDatasets(inpath, isLower=True, dep_path=None, token_path=None):
    with open(inpath) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')
    all_instances = []
    trees, tok_seqs = None, []
    sent_idx = 0
    for instance in dataset:
        text = instance['text']
        if dep_path:
            tok_seqs.append(text.split())
        lemma = instance['annotation']['lemmas']
        pos = instance['annotation']['POSs']
        concepts = instance['concepts'].strip().replace("soup domain:soup", "soup").replace("direct-02 polarity:-", "direct-02").split()
        categories = instance['annotation']['categories'].split()
        map_infos = instance['annotation']['mapinfo'].split("_#_")
        assert len(concepts) == len(categories) and len(concepts) == len(map_infos), "%d %d %d %s %s" % (len(concepts), len(categories), len(map_infos), str(concepts), str(categories))

        input_sent = AnonSentence(text, lemma, pos, concepts, categories, map_infos)

        input_sent.idx = sent_idx

        sent_idx += 1

        feats = [x.split('_#_') for x in instance['feats'].strip().split()]
        # print ("feature dimention: %d" % len(feats[0]))

        actions = instance['actionseq'].strip().split()
        action2cid = [int(x) for x in instance['alignment']['concept-align'].strip().split()]
        action2wid = [int(x) for x in instance['alignment']['word-align'].strip().split()]
        cid2wid = [int(x) for x in instance['alignment']['concept-to-word'].strip().split()]

        if dep_path is None and len(feats) == 0:
            continue

        assert len(feats) == len(actions)
        assert len(feats) == len(actions)
        assert len(feats) == len(action2cid)
        assert len(feats) == len(action2wid)

        # TODO: Check whether add cid2wid changes some parts of training!
        all_instances.append((input_sent, concepts, cid2wid, feats, actions, action2cid, action2wid))
    if dep_path:
        all_orig_tokens = oracle.ioutil.readToks(token_path)
        for (sent_idx, orig_seq) in enumerate(all_orig_tokens):
            assert len(orig_seq) == len(tok_seqs[sent_idx])
        # trees = oracle.ioutil.loadDependency(dep_path, tok_seqs, True)
        trees = oracle.ioutil.loadDependency(dep_path, all_orig_tokens)

        assert len(trees) == len(all_instances), ("inconsistent number of dependencies and instances: "
                                                  "%d vs %d" % (len(trees), len(all_instances)))
        for (idx, tree) in enumerate(trees):
            all_instances[idx][0].tree = tree

    return all_instances

def load_actions(indir):
    shiftpop_actions = set(["SHIFT", "POP"])
    pushidx_actions = oracle.ioutil.loadCounter(os.path.join(indir, "pushidx_actions.txt"))
    arcbinary_actions = set(["NOARC", "ARC"])
    # arcbinary_actions = oracle.ioutil.loadCounter(os.path.join(indir, "arc_binary_actions.txt"))
    arclabel_actions = oracle.ioutil.loadCounter(os.path.join(indir, "arc_label_actions.txt"))
    return shiftpop_actions, pushidx_actions, arcbinary_actions, arclabel_actions

def load_arc_choices(indir):
    outgo_arc_choices = oracle.ioutil.loadArcMaps(os.path.join(indir, "concept_rels.txt"))
    income_arc_choices = oracle.ioutil.loadArcMaps(os.path.join(indir, "concept_incomes.txt"))
    default_arc_choices = oracle.ioutil.defaultArcChoices()
    print ("ARC choices for %d incomes, %d outgos, %d defaults." % (len(income_arc_choices),
                                                                    len(outgo_arc_choices), len(default_arc_choices)))
    return income_arc_choices, outgo_arc_choices, default_arc_choices

def read_Testset(indir, isLower=True, decode=False):
    if decode:
        test_file = os.path.join(indir, "decode.json")
    else:
        test_file = os.path.join(indir, "oracle_examples.json") # TODO, change test file naming.
    dep_file = os.path.join(indir, "dep")
    token_file = os.path.join(indir, "token")
    return read_all_GenerationDatasets(test_file, isLower, dep_file, token_file)

def read_generation_datasets_from_fof(fofpath, isLower=True):
    all_paths = read_text_file(fofpath)
    all_instances = []
    for cur_path in all_paths:
        print(cur_path)
        cur_instances = read_all_GenerationDatasets(cur_path, isLower=isLower)
        all_instances.extend(cur_instances)
    return all_instances


def collect_vocabs(all_instances):
    all_words = set()
    all_POSs = set()
    all_feats = set()
    all_actions = set(['<s>', '-NULL-',])
    for (input_sent, concepts, cid2wid, feats, actions, action2cid, action2wid) in all_instances:
        all_words.update(input_sent.tok)
        all_words.update(input_sent.lemma)
        all_words.update(concepts)
        all_POSs.update(input_sent.pos)
        for feat in feats:
            all_feats.update(feat)
        all_actions.update(actions)
    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_POSs, all_feats, all_actions)


class DataStream(object):
    def __init__(self, all_questions, word_vocab=None, char_vocab=None, POS_vocab=None, feat_vocab=None, action_vocab=None,
            options=None, isShuffle=False, isLoop=False, isSort=True, batch_size=-1, decode=False):
        self.options = options
        if batch_size ==-1: batch_size=options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (input_sent, concepts, cid2wid, feats, actions, action2cid, action2wid) in all_questions:# sent1 is the long passage or article
            if options.max_passage_len!=-1 and not decode:
                if input_sent.get_length() > options.max_passage_len: continue # remove very long passages
            input_sent.convert2index(word_vocab, char_vocab, POS_vocab, max_char_per_word=options.max_char_per_word)
            concepts_idx = word_vocab.to_index_sequence_for_list(concepts)
            feats_idx = [feat_vocab.to_index_sequence_for_list(x) for x in feats]
            for x in feats_idx:
                assert len(x) == options.feat_num, len(x)
            actions_idx = action_vocab.to_index_sequence_for_list(actions)
            instances.append((input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid))

        all_questions = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_questions = sorted(all_questions, key=lambda question: (question[0].get_length(), len(question[4])))
        elif isShuffle:
            random.shuffle(all_questions)
            random.shuffle(all_questions)
        self.num_instances = len(all_questions)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_questions = []
            for i in xrange(batch_start, batch_end):
                cur_questions.append(all_questions[i])
            cur_batch = Batch(cur_questions, options, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab,
                    feat_vocab=feat_vocab, action_vocab=action_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

class Batch(object):
    def __init__(self, instances, options, word_vocab=None, char_vocab=None, POS_vocab=None, feat_vocab=None, action_vocab=None):
        self.options = options

        self.instances = instances
        self.batch_size = len(instances)
        self.action_vocab = action_vocab
        self.feat_vocab = feat_vocab  # Added the feature indexer as batch attributes.

        # create length
        self.input_length = [] # [batch_size]
        self.concept_length = [] # [batch_size]
        self.action_length = [] # [batch_size]
        for (input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid) in instances:
            self.input_length.append(input_sent.get_length()+1)
            self.concept_length.append(len(concepts_idx)+1)
            self.action_length.append(min(options.max_answer_len,len(actions_idx)))
        self.input_length = np.array(self.input_length, dtype=np.int32)
        self.concept_length = np.array(self.concept_length, dtype=np.int32)
        self.action_length = np.array(self.action_length, dtype=np.int32)

        start_id = action_vocab.getIndex('<s>')
        self.action_inp = []
        self.action_ref = []
        self.feats = []
        self.action2cid = []
        self.action2wid = []
        for (input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid) in instances:
            self.action_inp.append([start_id,]+actions_idx[:-1])
            self.action_ref.append(actions_idx)
            self.feats.append(feats_idx)
            self.action2cid.append(action2cid)
            self.action2wid.append(action2wid)
        self.action_inp = padding_utils.pad_2d_vals(self.action_inp, len(self.action_inp), options.max_answer_len)
        self.action_ref = padding_utils.pad_2d_vals(self.action_ref, len(self.action_ref), options.max_answer_len)
        self.feats = padding_utils.pad_3d_vals(self.feats, len(self.feats), options.max_answer_len, len(self.feats[0][0]))
        self.action2cid = padding_utils.pad_2d_vals(self.action2cid, len(self.action2cid), options.max_answer_len)
        self.action2wid = padding_utils.pad_2d_vals(self.action2wid, len(self.action2wid), options.max_answer_len)

        append_id = word_vocab.getIndex('-NULL-')
        self.input_word = [] # [batch_size, sent_len]
        self.concept_word = [] # [batch_size, sent_len]
        for (input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid) in instances:
            self.input_word.append(input_sent.word_idx_seq+[append_id,])
            self.concept_word.append(concepts_idx+[append_id,])
        self.input_word = padding_utils.pad_2d_vals_no_size(self.input_word)
        self.concept_word = padding_utils.pad_2d_vals_no_size(self.concept_word)

        if options.with_lemma:
            self.input_lemma = []
            for (input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid) in instances:
                self.input_lemma.append(input_sent.lemma_idx_seq+[append_id,])
            self.input_lemma = padding_utils.pad_2d_vals_no_size(self.input_lemma)

        if options.with_char:
            assert False
            self.input_char = [] # [batch_size, sent_len, char_size]
            self.input_char_len = [] # [batch_size, sent_len]
            for (input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid) in instances:
                self.input_char.append(input_sent.char_idx_matrix)
                self.input_char_len.append([len(x) for x in input_sent.tok])
            self.input_char = padding_utils.pad_3d_vals_no_size(self.input_char)
            self.input_char_len = padding_utils.pad_2d_vals_no_size(self.input_char_len)

        if options.with_POS:
            append_pos_id = POS_vocab.getIndex('-NULL-')
            self.input_POS = [] # [batch_size, sent1_len]
            for (input_sent, concepts_idx, cid2wid, feats_idx, actions_idx, action2cid, action2wid) in instances:
                self.input_POS.append(input_sent.POS_idx_seq+[append_pos_id,])
            self.input_POS = padding_utils.pad_2d_vals_no_size(self.input_POS)

    def get_feature_idxs(self, feats):
        return self.feat_vocab.to_index_sequence_for_list(feats)

    def map_idx_to_text(self, samples):
        '''
        sample: [batch_size, length] of idx
        '''
        all_words = []
        all_word_idx = []
        for i in xrange(len(samples)):
            cur_sample = samples[i]
            cur_words = []
            cur_word_idx = []
            for idx in cur_sample:
                cur_word = self.action_vocab.getWord(idx)
                cur_words.append(cur_word)
                cur_word_idx.append(idx)
            all_words.append(cur_words)
            all_word_idx.append(cur_word_idx)
        return (all_words, all_word_idx) # [batch_size, length]


