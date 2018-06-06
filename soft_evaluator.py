# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import numpy as np
from oracle.cacheConfiguration import CacheConfiguration
from oracle.cacheTransition import CacheTransition
from postprocessing.amr_format import get_amr
import oracle.utils
import oracle.ioutil

from vocab_utils import Vocab
import namespace_utils
import soft_NP2P_data_stream
from soft_NP2P_model_graph import ModelGraph

import re

import tensorflow as tf
import soft_NP2P_trainer
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL

def search(sess, model, vocab, batch, options, decode_mode='greedy'):
    assert False, 'not in use'
    '''
    for greedy search, multinomial search
    '''
    # Run the encoder to get the encoder hidden states and decoder initial state
    (phrase_representations, initial_state, encoder_features,phrase_idx, phrase_mask) = model.run_encoder(sess, batch, options)
    # phrase_representations: [batch_size, passage_len, encode_dim]
    # initial_state: a tupel of [batch_size, gen_dim]
    # encoder_features: [batch_size, passage_len, attention_vec_size]
    # phrase_idx: [batch_size, passage_len]
    # phrase_mask: [batch_size, passage_len]

    word_t = batch.gen_input_words[:,0]
    state_t = initial_state
    context_t = np.zeros([batch.batch_size, model.encode_dim])
    coverage_t = np.zeros((batch.batch_size, phrase_representations.shape[1]))
    generator_output_idx = [] # store phrase index prediction
    text_results = []
    generator_input_idx = [word_t] # store word index
    for i in xrange(options.max_answer_len):
        if decode_mode == "pointwise": word_t = batch.gen_input_words[:,i]
        feed_dict = {}
        feed_dict[model.init_decoder_state] = state_t
        feed_dict[model.context_t_1] = context_t
        feed_dict[model.coverage_t_1] = coverage_t
        feed_dict[model.word_t] = word_t

        feed_dict[model.phrase_representations] = phrase_representations
        feed_dict[model.encoder_features] = encoder_features
        feed_dict[model.phrase_idx] = phrase_idx
        feed_dict[model.phrase_mask] = phrase_mask
        if options.with_phrase_projection:
            feed_dict[model.max_phrase_size] = batch.max_phrase_size
            if options.add_first_word_prob_for_phrase:
                feed_dict[model.in_passage_words] = batch.sent1_word
                feed_dict[model.phrase_starts] = batch.phrase_starts



        if decode_mode in ["greedy","pointwise"]:
            prediction = model.greedy_prediction
        elif decode_mode == "multinomial":
            prediction = model.multinomial_prediction

        (state_t, context_t, coverage_t, prediction) = sess.run([model.state_t, model.context_t,
                                                                 model.coverage_t, prediction], feed_dict)
        # convert prediction to word ids
        generator_output_idx.append(prediction)
        prediction = np.reshape(prediction, [prediction.size, 1])
        [cur_words, cur_word_idx] = batch.map_phrase_idx_to_text(prediction) # [batch_size, 1]
        cur_word_idx = np.array(cur_word_idx)
        cur_word_idx = np.reshape(cur_word_idx, [cur_word_idx.size])
        word_t = cur_word_idx
        cur_words = flatten_words(cur_words)
        text_results.append(cur_words)
        generator_input_idx.append(cur_word_idx)

    generator_input_idx = generator_input_idx[:-1] # remove the last word to shift one position to the right
    generator_output_idx = np.stack(generator_output_idx, axis=1) # [batch_size, max_len]
    generator_input_idx = np.stack(generator_input_idx, axis=1) # [batch_size, max_len]

    prediction_lengths = [] # [batch_size]
    sentences = [] # [batch_size]
    for i in xrange(batch.batch_size):
        words = []
        for j in xrange(options.max_answer_len):
            cur_phrase = text_results[j][i]
#             cur_phrase = cur_batch_text[j]
            words.append(cur_phrase)
            if cur_phrase == "</s>": break# filter out based on end symbol
        prediction_lengths.append(len(words))
        cur_sent = " ".join(words)
        sentences.append(cur_sent)

    return (sentences, prediction_lengths, generator_input_idx, generator_output_idx)

def flatten_words(cur_words):
    all_words = []
    for i in xrange(len(cur_words)):
        all_words.append(cur_words[i][0])
    return all_words

class Hypothesis(object):
    def __init__(self, actions, log_ps, state, context_input, context_concept, cache_config=None):

        self.actions = actions # store all actions
        self.log_probs = log_ps # store log_probs for each time-step

        self.state = state
        self.context_input = context_input
        self.context_concept = context_concept

        # TODO xiaochang
        self.trans_state = cache_config
        self.word_focus = 0
        self.concept_focus = 0

        self.cache_idx = 0

    def addAction(self, action):
        self.actions.append(action)

    def actionSeqStr(self, action_vocab):
        return "#".join([action_vocab.getWord(action_id) for action_id in self.actions])

    def extend(self, system, action_id, log_prob, state, context_input, context_concept, action_vocab, gold=False):
        action = action_vocab.getWord(action_id)
        if (not gold) and not system.canApply(self.trans_state, action, self.concept_focus, True, self.cache_idx):
            return None

        cache_size = self.trans_state.cache_size
        new_config = CacheConfiguration(cache_size, -1,
                                        self.trans_state) # Initialize from another config
        next_cache_idx = self.cache_idx
        new_focus = self.concept_focus
        if new_config.phase == oracle.utils.FeatureType.SHIFTPOP:
            if action == "SHIFT": # Process the next concept.
                assert self.concept_focus == self.trans_state.hypothesis.nextConceptIDX()
                curr_concept = new_config.getConcept(self.concept_focus)
                oracle_action = "conID:" + curr_concept
                if new_config.isUnalign(self.concept_focus):
                    oracle_action = "conGen:" + curr_concept
                next_cache_idx = cache_size - 2
            else:
                assert action == "POP"
                oracle_action = action
            system.apply(new_config, oracle_action)
        elif new_config.phase == oracle.utils.FeatureType.PUSHIDX:
            assert action_vocab.getWord(self.actions[-1]) == "SHIFT"
            assert next_cache_idx == cache_size - 2
            system.apply(new_config, action)
        elif new_config.phase == oracle.utils.FeatureType.ARCBINARY:
            if action == "NOARC": # No arc made to current cache index
                # next_cache_idx += 1
                if next_cache_idx == 0: # Already the last cache index
                    next_cache_idx = 0.5
                    new_config.phase = oracle.utils.FeatureType.SHIFTPOP
                    new_focus += 1
                else:
                    next_cache_idx -= 1
                # if next_cache_idx == cache_size - 1: # Have processed all vertices.
                #     next_cache_idx = 0
                #     new_config.phase = oracle.utils.FeatureType.SHIFTPOP
                #     new_focus += 1
            else: # Then process the label
                assert action == "ARC"
                new_config.phase = oracle.utils.FeatureType.ARCCONNECT
        else:
            assert new_config.phase == oracle.utils.FeatureType.ARCCONNECT
            oracle_action = "ARC%d:%s" % (next_cache_idx, action)
            system.apply(new_config, oracle_action)
            # next_cache_idx += 1
            # if next_cache_idx == cache_size - 1:
            if next_cache_idx == 0:
                next_cache_idx = 0.5
                assert new_config.phase == oracle.utils.FeatureType.SHIFTPOP
                new_focus += 1
            else:
                next_cache_idx -= 1

        new_actions = self.actions + [action_id]
        new_probs = self.log_probs + [log_prob]
        new_hyp = Hypothesis(new_actions, new_probs, state, context_input, context_concept, new_config)
        if new_config.phase == oracle.utils.FeatureType.SHIFTPOP:
            new_hyp.word_focus = new_config.nextBufferElem()
            if new_hyp.word_focus == -1: # Either POP or after PUSHIDX.
                new_hyp.word_focus = len(self.trans_state.wordSeq)
        else: # ARC or PUSHIDX
            new_hyp.word_focus = self.word_focus # The word focus does not change during arc or pushidx.
        new_hyp.cache_idx = next_cache_idx
        new_hyp.concept_focus = new_focus
        return new_hyp

    def readOffUnalignWords(self):
        concept_align = self.trans_state.conceptAlign

        # If all concepts are read, should also move word pointer to the last.
        if self.concept_focus >= len(concept_align):
            self.trans_state.clearBuffer()
            self.word_focus = len(self.trans_state.wordSeq)
            return

        length = len(self.trans_state.wordSeq)
        while (self.word_focus not in self.trans_state.widTocid) and self.word_focus < length: # Some words are unaligned.
            popped = self.trans_state.popBuffer()
            assert popped == self.word_focus
            self.word_focus += 1

    def extractFeatures(self):
        # At first step, decide whether to shift or pop.
        word_idx, concept_idx = self.word_focus, self.concept_focus
        if (self.trans_state.phase == oracle.utils.FeatureType.ARCBINARY or self.trans_state.phase
            == oracle.utils.FeatureType.ARCCONNECT):
            assert self.actions, "Empty action sequence start without shift or pop"
            assert self.cache_idx != -1, "Cache related operation without cache index."
            word_idx, concept_idx = self.trans_state.rightmostCache()
        return self.trans_state.extractFeatures(self.trans_state.phase, word_idx, concept_idx, self.cache_idx)

    def isFinal(self):
        return

    def latest_action(self):
        return self.actions[-1]

    def avg_log_prob(self):
        return np.sum(self.log_probs[1:])/ (len(self.actions)-1)

    def probs2string(self):
        out_string = ""
        for prob in self.log_probs:
            out_string += " %.4f" % prob
        return out_string.strip()

    def amr_string(self):
        pass
        # TODO xiaochang, make a string that represents an AMR from self.trans_state


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)


def run_beam_search(sess, trans_system, model, feat_vocab, action_vocab, batch, cache_size, options, category_res):
    # Run encoder
    # TODO: this is inconsistent with the paramters returned by run_encoder?
    (initial_state, input_hiddens, input_features, input_mask,
            concept_hiddens, concept_features, concept_mask) = model.run_encoder(sess, batch, options)

    sent_stop_id = action_vocab.getIndex('</s>')
    # Initialize this first hypothesis
    context_input_init = np.zeros([model.input_hidden_dim])
    context_concept_init = np.zeros([model.concept_hidden_dim])

    # Initialize decode
    sent_anno = batch.instances[0][0]
    concept_seq = sent_anno.concepts
    concept_align = batch.instances[0][2]
    sent_length = sent_anno.length
    concept_num = len(concept_seq)
    concept_categories = sent_anno.categories
    assert len(concept_align) == concept_num, "%s %s" % (str(concept_seq), str(concept_align))
    initial_config = CacheConfiguration(cache_size, sent_length)

    # all these attributes should be shared by all hypothesis
    initial_config.wordSeq, initial_config.lemSeq, initial_config.posSeq = sent_anno.tok, sent_anno.lemma, sent_anno.pos
    initial_config.conceptSeq, initial_config.conceptAlign = concept_seq, concept_align
    initial_config.categorySeq = concept_categories
    initial_config.tree = sent_anno.tree
    initial_config.buildWordToConcept()
    print (sent_anno.tok)
    print (concept_seq)
    print (concept_categories)

    assert sent_anno.tree is not None

    start_action_id = batch.action_inp[0][0]
    initial_actionseq = [start_action_id]
    initial_hypo = Hypothesis(initial_actionseq, [0.0], initial_state, context_input_init,
                              context_concept_init, initial_config)
    hyps = [initial_hypo]

    # beam search decoding
    steps = 0
    while steps < len(batch.instances[0][-3]) and steps < options.max_answer_len:
        cur_size = len(hyps) # current number of hypothesis in the beam

        cur_input_hiddens = np.tile(input_hiddens, (cur_size, 1, 1)) # [batch_size, passage_len, enc_hidden_dim]
        cur_input_features = np.tile(input_features, (cur_size, 1, 1)) # [batch_size, passage_len, options.attention_vec_size]
        cur_input_mask = np.tile(input_mask, (cur_size, 1)) # [batch_size, passage_len]

        cur_concept_hiddens = np.tile(concept_hiddens, (cur_size, 1, 1))
        cur_concept_features = np.tile(concept_features, (cur_size, 1, 1)) # [batch_size, passage_len, options.attention_vec_size]
        cur_concept_mask = np.tile(concept_mask, (cur_size, 1)) # [batch_size, passage_len]

        cur_state_t_1 = [] # [2, gen_steps]

        cur_context_input_t_1 = [] # [batch_size, input_hidden_dim]
        cur_context_concept_t_1 = [] # [batch_size, concept_hidden_len]
        cur_action_t = [] # [batch_size]

        cur_action_feats = batch.feats[:,steps,:].tolist() # [batch, feat_num]
        feat_reprs = []
        for h in hyps:
            cur_state_t_1.append(h.state)
            cur_context_input_t_1.append(h.context_input)
            cur_context_concept_t_1.append(h.context_concept)
            cur_action_t.append(h.latest_action())

        cur_context_input_t_1 = np.stack(cur_context_input_t_1, axis=0)
        cur_context_concept_t_1 = np.stack(cur_context_concept_t_1, axis=0)
        cur_action_t = np.array(cur_action_t, dtype='int32')
        cur_action_feats = np.array(cur_action_feats, dtype='int32')

        cells = [state.c for state in cur_state_t_1]
        hidds = [state.h for state in cur_state_t_1]
        new_c = np.concatenate(cells, axis=0)
        new_h = np.concatenate(hidds, axis=0)
        new_dec_init_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        feed_dict = {}
        feed_dict[model.init_decoder_state] = new_dec_init_state
        feed_dict[model.context_input_t_1] = cur_context_input_t_1
        feed_dict[model.context_concept_t_1] = cur_context_concept_t_1
        feed_dict[model.actionidx_t] = cur_action_t

        # TODO: extract configuration features and map them to feature indices.
        feed_dict[model.featidx_t] = cur_action_feats

        feed_dict[model.input_hiddens] = cur_input_hiddens
        feed_dict[model.input_features] = cur_input_features
        feed_dict[model.input_mask] = cur_input_mask

        feed_dict[model.concept_hiddens] = cur_concept_hiddens
        feed_dict[model.concept_features] = cur_concept_features
        feed_dict[model.concept_mask] = cur_concept_mask

        (state_t, context_input_t, context_concept_t, topk_log_probs, topk_ids) = sess.run([model.state_t,
                model.context_input_t, model.context_concept_t, model.topk_log_probs, model.topk_ids], feed_dict)

        new_states = [tf.nn.rnn_cell.LSTMStateTuple(state_t.c[i:i+1, :], state_t.h[i:i+1, :]) for i in xrange(cur_size)]


        # Extend each hypothesis and collect them all in all_hyps
        feat_id = batch.instances[0][-4][steps][0]
        action_id = batch.instances[0][-3][steps]
        all_hyps = []
        assert cur_size == 1
        for i in xrange(cur_size):
            h = hyps[i]
            cur_state = new_states[i]
            cur_context_input = context_input_t[i]
            cur_context_concept = context_concept_t[i]
            # get accuracy
            for j in xrange(options.topk_size):
                cur_action_id = topk_ids[i, j]
                # cur_action = action_vocab.getWord(cur_action_id)
                cur_action_log_prob = topk_log_probs[i, j]

                new_hyp = h.extend(trans_system, cur_action_id, cur_action_log_prob, cur_state,
                                   cur_context_input, cur_context_concept, action_vocab)
                if new_hyp:
                    category_res[feat_id][0] += 1.0
                    if cur_action_id == action_id:
                        category_res[feat_id][1] += 1.0
                    break
            # fill the gold action
            new_hyp = h.extend(trans_system, action_id, 1.0, cur_state,
                                cur_context_input, cur_context_concept, action_vocab, gold=True)
            assert new_hyp is not None
            # if new_hyp == None:
            #     break
            all_hyps.append(new_hyp)

        if len(all_hyps) == 0:
            print ("No hypothesis found at step %d" % steps)
            break

        # Filter and collect any hypotheses that have produced the end action.
        # hyps will contain hypotheses for the next step
        hyps = []
        for h in sort_hyps(all_hyps):
            # If this hypothesis is sufficiently long, put in results. Otherwise discard.
            if h.latest_action() == sent_stop_id or trans_system.isTerminal(h.trans_state):
                if steps >= options.min_answer_len:
                    print('!!!break here')
                    break
            # hasn't reached stop action, so continue to extend this hypothesis
            else:
                hyps.append(h)

        if len(hyps) == 0:
            break

        steps += 1


def generateAMR(hypo, sent_anno):
    concept_line_reprs = hypo.trans_state.toConll()
    category_map = sent_anno.map_info
    return get_amr(concept_line_reprs, category_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--cache_size', type=int, help='Cache size for the cache transition system.')
    parser.add_argument("--decode", action="store_true", help="if to decode new sentences.")

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    cache_size = args.cache_size
    use_dep = args.decode

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = soft_NP2P_trainer.enrich_options(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = char_vocab = POS_vocab = NER_vocab = None
    word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    if FLAGS.with_POS:
        POS_vocab = Vocab(model_prefix + ".POS_vocab", fileformat='txt2')
        print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
    action_vocab = Vocab(model_prefix + ".action_vocab", fileformat='txt2')
    print('action_vocab: {}'.format(action_vocab.word_vecs.shape))
    feat_vocab = Vocab(model_prefix + ".feat_vocab", fileformat='txt2')
    print('feat_vocab: {}'.format(feat_vocab.word_vecs.shape))

    print('Loading test set.')
    if use_dep:
        testset = soft_NP2P_data_stream.read_Testset(in_path)
    elif FLAGS.infile_format == 'fof':
        testset = soft_NP2P_data_stream.read_generation_datasets_from_fof(in_path, isLower=FLAGS.isLower)
    else:
        testset = soft_NP2P_data_stream.read_all_GenerationDatasets(in_path, isLower=FLAGS.isLower)
    print('Number of samples: {}'.format(len(testset)))

    print('Build DataStream ... ')
    batch_size=1
    assert batch_size == 1

    devDataStream = soft_NP2P_data_stream.DataStream(testset,
            word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, feat_vocab=feat_vocab, action_vocab=action_vocab,
            options=FLAGS, isShuffle=False, isLoop=False, isSort=True, batch_size=batch_size, decode=True)
    print('Number of instances in testDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(devDataStream.get_num_batch()))

    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab,
                        feat_vocab=feat_vocab, action_vocab=action_vocab, options=FLAGS, mode="decode")

        ## remove word _embedding
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        system = CacheTransition(cache_size, oracle.utils.OracleType.CL)
        if use_dep:
            shiftpop, pushidx, arcbinary, arclabel = soft_NP2P_data_stream.load_actions(in_path)
            system.shiftpop_action_set, system.push_action_set = shiftpop, pushidx
            system.arcbinary_action_set, system.arclabel_action_set = arcbinary, arclabel
            income_arc_choices, outgo_arc_choices, default_arc_choices = soft_NP2P_data_stream.load_arc_choices(in_path)
            system.income_arcChoices, system.outgo_arcChoices = income_arc_choices, outgo_arc_choices
            system.default_arcChoices = default_arc_choices

        category_res = {feat_vocab.getIndex(x):[0.0,0.0,] for x in ('PHASE=PUSHIDX', 'PHASE=SHTPOP', 'PHASE=ARCBINARY', 'PHASE=ARCLABEL',)}
        devDataStream.reset()
        for i in range(devDataStream.get_num_batch()):
            cur_batch = devDataStream.get_batch(i)
            print('Instance {}'.format(i))
            run_beam_search(sess, system, valid_graph, feat_vocab, action_vocab, cur_batch, cache_size, FLAGS, category_res)
        for k,v in category_res.iteritems():
            k = feat_vocab.getWord(k)
            print('%s : %.4f %d/%d' %(k, v[1]/v[0], v[1], v[0]))


