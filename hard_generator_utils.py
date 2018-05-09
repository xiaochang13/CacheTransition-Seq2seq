from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops

def collect_step(memory, step): # [batch, node_len, dim], [batch]
    step = tf.maximum(step, tf.zeros_like(step, dtype=tf.int32)) # [batch,]
    rowid = tf.range(0, limit=tf.shape(step)[0]) # [batch,]
    indices = tf.stack((rowid,step), axis=1) # [batch_size, 2]
    return tf.gather_nd(memory, indices)

class HardAttnGen:
    def __init__(self, placeholders, options, action_vocab, feat_vocab):
        self.options = options
        self.action_vocab = action_vocab
        self.feat_vocab = feat_vocab
        self.cell = tf.contrib.rnn.LSTMCell(
                    options.gen_hidden_size,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=True)
        self.placeholders = placeholders

        with tf.variable_scope("embedding"):
            self.action_embedding = tf.get_variable('action_embedding',
                                    initializer=tf.constant(self.action_vocab.word_vecs), dtype=tf.float32)
            self.feat_embedding = tf.get_variable('feat_embedding',
                                    initializer=tf.constant(self.feat_vocab.word_vecs), dtype=tf.float32)


    def one_step_decoder(self, state_t_1, context_input_t_1, context_concept_t_1, word_t, featidx_t, wid_t, cid_t,
            input_states, input_mask, concept_states, concept_mask):
        '''
        state_t_1: Tuple of [batch_size, gen_hidden_size]
        context_t_1: [batch_size, encoder_dim]
        word_t: [batch_size, word_dim]
        featidx_t: [batch_size]
        wid_t: [batch_size]
        cid_t: [batch_size]
        encoder_states: [batch_size, passage_len, encoder_dim]
        encoder_mask: [batch_size, passage_len]
        '''

        options = self.options
        batch_size = tf.shape(featidx_t)[0]

        # [batch, feat_num] -> [batch, feat_num, feat_dim]
        feat_t = _embedding_lookup(featidx_t, self.feat_embedding)
        # [batch, feat_num*feat_dim]
        feat_t = tf.reshape(feat_t, [batch_size, options.feat_num*options.feat_dim])
        w_feat = tf.get_variable("w_feat", [options.feat_num*options.feat_dim, options.feat_comp_dim])
        b_feat = tf.get_variable("b_feat", [options.feat_comp_dim])
        # [batch, feat_comp_dim]
        feat_t = tf.matmul(feat_t, w_feat) + b_feat

        x = _linear([word_t, feat_t, context_input_t_1, context_concept_t_1], options.attention_vec_size, True)

        # Run the decoder RNN cell. cell_output = decoder state
        cell_output, state_t = self.cell(x, state_t_1)

        context_input_t = collect_step(input_states, wid_t)

        context_concept_t = collect_step(concept_states, cid_t)

        # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
        # This is V[s_t, h*_t] + b in the paper
        with variable_scope.variable_scope("output_projection_1"):
            output_t = _linear([cell_output] + [context_input_t] + [context_concept_t], options.gen_hidden_size, True)

        with tf.variable_scope('output_projection_2'):
            w = tf.get_variable('w', [options.gen_hidden_size, self.action_vocab.vocab_size+1], dtype=tf.float32)
            b = tf.get_variable('b', [self.action_vocab.vocab_size+1], dtype=tf.float32)
            # vocab_scores is the vocabulary distribution before applying softmax.
            # Each entry on the list corresponds to one decoder step
            vocab_score_t = tf.nn.xw_plus_b(output_t, w, b) # apply the linear layer
            vocab_score_t = tf.nn.softmax(vocab_score_t)

            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            vocab_score_t = _clip_and_normalize(vocab_score_t, 1e-6)

        return (state_t, context_input_t, context_concept_t, vocab_score_t)


    def train_mode(self, input_dim, input_states, input_mask, concept_dim, concept_states, concept_mask,
            init_state, decoder_inputs, decoder_refs, decoder_feats, decoder_wids, decoder_cids, loss_weights, mode_gen='ce_train'):
        '''
        encoder_dim: int-valued
        encoder_states: [batch_size, passage_len, encoder_dim].
        encoder_mask: [batch_size, passage_len] int32
        init_state: Tuple of [batch_size, gen_hidden_size]
        decoder_inputs: [batch_size, max_dec_steps].
        decoder_refs: [batch_size, max_dec_steps]
        decoder_feats: [batch_size, max_dec_steps]
        decoder_wids: [batch_size, max_dec_steps]
        decoder_cids: [batch_size, max_dec_steps]
        '''
        options = self.options

        batch_size = tf.shape(input_states)[0]
        input_len = tf.shape(input_states)[1]
        concept_len = tf.shape(concept_states)[1]


        decoder_inputs = tf.unstack(decoder_inputs, axis=1) # max_dec_steps * [batch_size]
        decoder_refs_unstack = tf.unstack(decoder_refs, axis=1) # max_dec_steps * [batch_size]
        decoder_feats = tf.unstack(decoder_feats, axis=1) # max_dec_steps * [batch_size, feat_num]
        decoder_wids = tf.unstack(decoder_wids, axis=1) # max_dec_steps * [batch_size]
        decoder_cids = tf.unstack(decoder_cids, axis=1) # max_dec_steps * [batch_size]

        # initialize all the variables
        state_t_1 = init_state
        context_input_t_1 = tf.zeros([batch_size, input_dim])
        context_concept_t_1 = tf.zeros([batch_size, concept_dim])

        # store variables from each time-step
        vocab_scores = []
        sampled_words = []
        with variable_scope.variable_scope("attention_decoder"):
            wordidx_t = decoder_inputs[0] # [batch_size] int32
            featidx_t = decoder_feats[0] # [batch_size, feat_num] int32
            wid_t = decoder_wids[0]
            cid_t = decoder_cids[0]
            for i in range(options.max_answer_len):
                if mode_gen in ('ce_train', 'loss',):
                    wordidx_t = decoder_inputs[i]
                    featidx_t = decoder_feats[i]
                    wid_t = decoder_wids[i]
                    cid_t = decoder_cids[i]
                word_t = _embedding_lookup(wordidx_t, self.action_embedding)
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                (state_t, context_input_t, context_concept_t, output_t) = \
                        self.one_step_decoder(state_t_1, context_input_t_1, context_concept_t_1, word_t, featidx_t, wid_t, cid_t,
                                input_states, input_mask, concept_states, concept_mask)

                vocab_scores.append(output_t)

                state_t_1 = state_t
                context_input_t_1 = context_input_t
                context_concept_t_1 = context_concept_t

                if mode_gen == 'greedy':
                    # TODO update featidx_t
                    wordidx_t = tf.argmax(output_t, 1) # [batch_size]
                    wordidx_t = tf.reshape(wordidx_t, [-1]) # [batch_size]
                elif mode_gen == 'sample':
                    # TODO update featidx_t
                    log_score_t = tf.log(output_t) # [batch_size, vsize]
                    wordidx_t = tf.multinomial(log_score_t, 1) # [batch_size, 1]
                    wordidx_t = tf.reshape(wordidx_t, [-1]) # [batch_size]
                elif mode_gen in ('ce_train', 'loss',):
                    wordidx_t = tf.argmax(output_t, axis=1) # [batch]
                else:
                    assert False, 'unknown generating mode %s' % mode_gen
                sampled_words.append(wordidx_t)

        if len(sampled_words)!=0:
            sampled_words = tf.stack(sampled_words, axis=1) # [batch_size, max_dec_steps]

        vocab_scores = tf.stack(vocab_scores, axis=1) # [batch_size, max_dec_steps, vocab]

        # calculating loss
        self.loss = None
        if mode_gen in ('ce_train', 'loss', ):
            xent = _CE_loss(vocab_scores, decoder_refs, loss_weights) # [batch_size]
            if mode_gen == 'loss': xent *= self.placeholders.reward # multiply with rewards
            self.loss = tf.reduce_mean(xent)

        # accuracy is calculated only under 'ce_train', where true answer is given
        if mode_gen == 'ce_train':
            accuracy = _mask_and_accuracy(vocab_scores, decoder_refs, loss_weights)
            return accuracy, self.loss, sampled_words
        else:
            return None, self.loss, sampled_words


    def decode_mode(self, state_t_1, context_input_t_1, context_concept_t_1, wordidx_t, featidx_t, wid_t, cid_t,
                input_states, input_mask, concept_states, concept_mask):

        options = self.options

        with variable_scope.variable_scope("attention_decoder"):
            word_t = _embedding_lookup(wordidx_t, self.action_embedding)

            (state_t, context_input_t, context_concept_t, output_t) = \
                    self.one_step_decoder(state_t_1, context_input_t_1, context_concept_t_1, word_t, featidx_t, wid_t, cid_t,
                            input_states, input_mask, concept_states, concept_mask)

            vocab_scores = tf.log(output_t)
            greedy_prediction = tf.reshape(tf.argmax(output_t, 1),[-1]) # calcualte greedy
            sample_prediction = tf.reshape(tf.multinomial(vocab_scores, 1),[-1]) # calculate multinomial
            topk_log_probs, topk_ids = tf.nn.top_k(vocab_scores, options.topk_size) # calculate topK

        return (state_t, context_input_t, context_concept_t, output_t,
                topk_log_probs, topk_ids, greedy_prediction, sample_prediction)


def _linear(args, output_size, bias=True, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(values=args, axis=1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return res + bias_term


def _clip_and_normalize(word_probs, epsilon):
    '''
    word_probs: 1D tensor of [vsize]
    '''
    word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True) # scale preds so that the class probas of each sample sum to 1


def _CE_loss(word_probs, answers, loss_weights):
    '''
    word_probs: [batch_size, max_dec_steps, vocab]
    answers: [batch_size, max_dec_steps]
    loss_weigts: [batch_size, max_dec_steps]
    '''
    #word_probs = tf.nn.softmax(word_probs, dim=-1)
    input_shape = tf.shape(word_probs)
    vsize = input_shape[2]

    epsilon = 1.0e-6
    word_probs = _clip_and_normalize(word_probs, epsilon)

    one_hot_spare_rep = tf.one_hot(answers, vsize)

    xent = -tf.reduce_sum(one_hot_spare_rep * tf.log(word_probs), axis=-1) # [batch_size, max_dec_steps]
    if loss_weights != None:
        xent = xent * loss_weights
    xent = tf.reduce_sum(xent, axis=-1)
    return xent


def _mask_and_avg(values, loss_weights):
    """Applies mask to values then returns overall average (a scalar)

      Args:
        values: a list length max_dec_steps containing arrays shape (batch_size).
        loss_weights: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

      Returns:
        a scalar
    """
    if loss_weights == None:
        return tf.reduce_mean(tf.stack(values, axis=0))

    dec_lens = tf.reduce_sum(loss_weights, axis=1) # shape batch_size. float32
    values_per_step = [v * loss_weights[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average


# values: [batch_size, step_size, vocab_size]
# answers: [batch_size, step_size]
def _mask_and_accuracy(values, answers, loss_weights):
    values = tf.argmax(values,axis=2)
    x = tf.cast(values, dtype=tf.int32)
    y = tf.cast(answers, dtype=tf.int32)
    res = tf.equal(x, y)
    res = tf.cast(res, dtype=tf.float32)
    res = tf.multiply(res, loss_weights)
    return tf.reduce_sum(res)


def _embedding_lookup(inputs, embedding):
    '''
    inputs: list of [batch_size], int32
    '''
    if type(inputs) is list:
        return [tf.nn.embedding_lookup(embedding, x) for x in inputs]
    else:
        return  tf.nn.embedding_lookup(embedding, inputs)


