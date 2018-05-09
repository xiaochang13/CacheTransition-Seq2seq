import tensorflow as tf
from tensorflow.python.ops import variable_scope

import encoder_utils
import hard_generator_utils
import numpy as np
import padding_utils
import random
import copy


class ModelGraph(object):
    def __init__(self, word_vocab=None, char_vocab=None, POS_vocab=None, feat_vocab=None, action_vocab=None,
            options=None, mode='ce_train'):

        # here 'mode', whose value can be:
        #  'ce_train',
        #  'rl_train',
        #  'evaluate',
        #  'evaluate_bleu',
        #  'decode'.
        # it is different from 'mode_gen' in generator_utils.py
        # value of 'mode_gen' can be 'ce_train', 'loss', 'greedy' or 'sample'
        self.mode = mode

        # is_training controls whether to use dropout
        is_training = True if mode in ('ce_train', ) else False

        self.options = options
        self.word_vocab = word_vocab

        with tf.variable_scope('input_encoder'):
            self.input_encoder = encoder_utils.SeqEncoder(options,
                    word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab)
            self.input_hidden_dim, self.input_hiddens, self.input_decinit = \
                    self.input_encoder.encode(is_training=is_training)
            self.input_mask = self.input_encoder.passage_mask

        with tf.variable_scope('concept_encoder'):
            options_copy = copy.copy(options)
            options_copy.with_char = False
            options_copy.with_POS = False
            options_copy.with_lemma = False
            self.concept_encoder = encoder_utils.SeqEncoder(options_copy,
                    word_vocab=word_vocab, char_vocab=None, POS_vocab=None)
            self.concept_hidden_dim, self.concept_hiddens, self.concept_decinit = \
                    self.concept_encoder.encode(is_training=is_training)
            self.concept_mask = self.concept_encoder.passage_mask

        cat_c = tf.concat([self.input_decinit.c, self.concept_decinit.c], axis=1)
        cat_h = tf.concat([self.input_decinit.h, self.concept_decinit.h], axis=1)
        compress_w = tf.get_variable('compress_w', [self.input_hidden_dim+self.concept_hidden_dim, options.gen_hidden_size], dtype=tf.float32)
        compress_b = tf.get_variable('compress_b', [options.gen_hidden_size], dtype=tf.float32)
        cat_c = tf.matmul(cat_c, compress_w) + compress_b
        cat_h = tf.matmul(cat_h, compress_w) + compress_b
        self.init_decoder_state = tf.contrib.rnn.LSTMStateTuple(cat_c, cat_h)

        self.create_placeholders(options)

        gen_loss_mask = tf.sequence_mask(self.action_len, options.max_answer_len, dtype=tf.float32) # [batch_size, gen_steps]

        with variable_scope.variable_scope("generator"):
            # create generator
            self.generator = hard_generator_utils.HardAttnGen(self, options, action_vocab, feat_vocab)

            if mode == 'decode':
                # [batch_size, encode_dim]
                self.context_input_t_1 = tf.placeholder(tf.float32,
                        [None, self.input_hidden_dim], name='context_input_t_1')
                # [batch_size, encode_dim]
                self.context_concept_t_1 = tf.placeholder(tf.float32,
                        [None, self.concept_hidden_dim], name='context_concept_t_1')
                # [batch_size, feat_num]
                self.featidx_t = tf.placeholder(tf.int32, [None, None], name='featidx_t')
                # [batch_size]
                self.actionidx_t = tf.placeholder(tf.int32, [None], name='actionidx_t')
                # [batch_size]
                self.wid_t = tf.placeholder(tf.int32, [None], name='wid_t')
                # [batch_size]
                self.cid_t = tf.placeholder(tf.int32, [None], name='cid_t')

                (self.state_t, self.context_input_t, self.context_concept_t, self.ouput_t,
                    self.topk_log_probs, self.topk_ids, self.greedy_prediction, self.sample_prediction) = self.generator.decode_mode(
                        self.init_decoder_state, self.context_input_t_1, self.context_concept_t_1, self.actionidx_t, self.featidx_t, self.wid_t, self.cid_t,
                        self.input_hiddens, self.input_mask, self.concept_hiddens, self.concept_mask)
                # not buiding training op for this mode
                return
            elif mode == 'evaluate_bleu':
                assert False, 'not in use'
                _, _, self.greedy_words = self.generator.train_mode(mode_gen='greedy')
                # not buiding training op for this mode
                return
            elif mode in ('ce_train', 'evaluate', ):
                self.accu, self.loss, self.sampled_words = self.generator.train_mode(
                        self.input_hidden_dim, self.input_hiddens, self.input_mask, self.concept_hidden_dim, self.concept_hiddens, self.concept_mask,
                        self.init_decoder_state, self.action_inp, self.action_ref, self.feats, self.action_wids, self.action_cids,
                        gen_loss_mask, mode_gen='ce_train')
                if mode == 'evaluate': return # not buiding training op for evaluation
            elif mode == 'rl_train':
                assert False, 'not in use'
                _, self.loss, _ = self.generator.train_mode(mode_gen='loss')

                tf.get_variable_scope().reuse_variables()

                _, _, self.sampled_words = self.generator.train_mode(mode_gen='sample')

                _, _, self.greedy_words = self.generator.train_mode(mode_gen='greedy')
            else:
                assert False, 'unknow mode'


        if options.optimize_type == 'adadelta':
            clipper = 50
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=options.learning_rate)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        elif options.optimize_type == 'adam':
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)


    def create_placeholders(self, options):
        # build placeholder for answer
        self.feats = tf.placeholder(tf.int32, [None, options.max_answer_len, None], name="feats") # [batch_size, gen_steps, feat_num]

        self.action_wids = tf.placeholder(tf.int32, [None, options.max_answer_len], name="action_wids") # [batch_size, gen_steps]
        self.action_cids = tf.placeholder(tf.int32, [None, options.max_answer_len], name="action_cids") # [batch_size, gen_steps]

        self.action_inp = tf.placeholder(tf.int32, [None, options.max_answer_len], name="action_inp") # [batch_size, gen_steps]
        self.action_ref = tf.placeholder(tf.int32, [None, options.max_answer_len], name="action_ref") # [batch_size, gen_steps]
        self.action_len = tf.placeholder(tf.int32, [None], name="action_len") # [batch_size]

        # build placeholder for reinforcement learning
        self.reward = tf.placeholder(tf.float32, [None], name="reward")


    def run_greedy(self, sess, batch, options):
        assert False, 'under construction'
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True) # reuse this function to construct feed_dict
        feed_dict[self.gen_input_words] = batch.gen_input_words
        return sess.run(self.greedy_words, feed_dict)


    def run_ce_training(self, sess, batch, options, only_eval=False):
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True) # reuse this function to construct feed_dict

        feed_dict[self.feats] = batch.feats

        feed_dict[self.action_wids] = batch.action2wid
        feed_dict[self.action_cids] = batch.action2cid

        feed_dict[self.action_inp] = batch.action_inp
        feed_dict[self.action_ref] = batch.action_ref
        feed_dict[self.action_len] = batch.action_length

        if only_eval:
            return sess.run([self.accu, self.loss, self.sampled_words], feed_dict)
        else:
            return sess.run([self.train_op, self.loss], feed_dict)[1]


    def run_rl_training(self, sess, batch, options):
        assert False, 'not supported yet'
        flipp = options.flipp if options.__dict__.has_key('flipp') else 0.1

        # make feed_dict
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.action_inp] = batch.action_inp

        # get greedy and gold outputs
        greedy_output = sess.run(self.greedy_words, feed_dict)
        greedy_output = greedy_output.tolist()
        gold_output = batch.in_answer_words.tolist()

        # generate sample_output by flipping coins
        sample_output = np.copy(batch.action_out)
        for i in range(batch.in_answer_words.shape[0]):
            seq_len = min(options.max_answer_len, batch.action_length[i]-1) # don't change stop token '</s>'
            for j in range(seq_len):
                if greedy_output[i][j] != 0 and random.random() < flipp:
                    sample_output[i,j] = greedy_output[i][j]
        sample_output = sample_output.tolist()

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        reward = []
        for i, (sout,gout) in enumerate(zip(sample_output,greedy_output)):
            sout, slex = self.word_vocab.getLexical(sout)
            gout, glex = self.word_vocab.getLexical(gout)
            rl_inputs.append([int(batch.gen_input_words[i,0])]+sout[:-1])
            rl_outputs.append(sout)
            rl_input_lengths.append(len(sout))
            _, ref_lex = self.word_vocab.getLexical(gold_output[i])
            slst = slex.split()
            glst = glex.split()
            rlst = ref_lex.split()
            reward.append(r-b)

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.options.max_answer_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.options.max_answer_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.reward] = reward
        feed_dict[self.gen_input_words] = rl_inputs
        feed_dict[self.in_answer_words] = rl_outputs
        feed_dict[self.answer_lengths] = rl_input_lengths

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


    def run_encoder(self, sess, batch, options, only_feed_dict=False):
        feed_dict = {}
        feed_dict[self.input_encoder.passage_lengths] = batch.input_length
        feed_dict[self.input_encoder.in_passage_words] = batch.input_word
        feed_dict[self.concept_encoder.passage_lengths] = batch.concept_length
        feed_dict[self.concept_encoder.in_passage_words] = batch.concept_word
        if options.with_lemma:
            feed_dict[self.input_encoder.in_passage_lemma] = batch.input_lemma
        if options.with_POS:
            feed_dict[self.input_encoder.in_passage_POSs] = batch.input_POS
        if options.with_char:
            feed_dict[self.input_encoder.passage_char_lengths] = batch.input_char_len
            feed_dict[self.input_encoder.in_passage_chars] = batch.input_char

        if only_feed_dict:
            return feed_dict

        return sess.run([self.init_decoder_state,
            self.input_hiddens, self.input_mask,
            self.concept_hiddens, self.concept_mask], feed_dict)


