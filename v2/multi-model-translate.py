# coding:utf-8

'''
Translates a source file using a translation model.
'''

import argparse
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
# import ipdb
import numpy
import copy
import pprint
import math
import os
import warnings
import sys
import time
import numpy as np
from collections import OrderedDict

from data_iterator import dataIterator, dataIterator_valid

profile = False

import random
import re

theano.config.floatX = 'float32'


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


'''
返回一个维度转换矩阵
'''


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def init_params(options, params_dict):
    params = params_dict
    # embedding
    params['Wemb_dec'] = norm_weight(options['dim_target'], options['dim_word'])

    # encoder: bidirectional RNN
    # get_layer(options['encoder'])[0] = param_init_gru
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder0',
                                              nin=options['dim_feature'],
                                              dim=options['dim_enc'][0])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r0',
                                              nin=options['dim_feature'],
                                              dim=options['dim_enc'][0])

    hiddenSizes = options['dim_enc']
    for i in range(1, len(hiddenSizes)):
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder' + str(i),
                                                  nin=hiddenSizes[i - 1] * 2,
                                                  dim=hiddenSizes[i])

        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder_r' + str(i),
                                                  nin=hiddenSizes[i - 1] * 2,
                                                  dim=hiddenSizes[i])
    ctxdim = 2 * hiddenSizes[-1]

    # init_state, init_cell
    # get_layer(options['ff'])[0] = param_init_fflayer
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim_dec'])
    # decoder
    # get_layer(options['decoder'])[0] = param_init_gru_cond
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim_dec'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim_dec'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'] / 2,
                                nout=options['dim_target'])

    return params


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]
    return params


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def gen_sample(function_list, x, k=1, maxlen=30,
               stochastic=True, argmax=False):
    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    # 多模型hyp_scores不应该变
    hyp_scores = numpy.zeros(live_k).astype('float32')
    next_state_list = []
    ctx0_list = []
    h_list = []
    next_alpha_past_list = []
    options = function_list[0][2]

    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    SeqL = x.shape[0]
    hidden_sizes = options['dim_enc']
    for i in range(len(hidden_sizes)):
        if options['down_sample'][i] == 1:
            SeqL = math.ceil(SeqL / 2.)
    # get initial state of decoder rnn and encoder context
    for item in function_list:
        temp_ret = item[0](x)
        next_state_list.append(temp_ret[0])
        ctx0_list.append(temp_ret[1])
        h_list.append(temp_ret[2])
        next_alpha_past_list.append(0.0 * numpy.ones((1, int(SeqL))).astype('float32'))  # start position

    for ii in xrange(maxlen):
        # 每个模型进行预测
        assemble_list = []
        total_next_p = 0
        for index, item in enumerate(function_list):
            ctx0 = ctx0_list[index]
            # next_state:候选数量 × 256
            next_state = next_state_list[index]
            # next_alpha_past：候选数量 × 8
            next_alpha_past = next_alpha_past_list[index]
            ctx = numpy.tile(ctx0, [live_k, 1])
            inps = [next_w, ctx, next_state, next_alpha_past]
            ret = function_list[index][1](*inps)
            # 候选数量 × num_target
            next_p, new_next_w, next_state, next_alpha_past = ret[0], ret[1], ret[2], ret[3]
            total_next_p += next_p
            assemble_list.append((next_p, new_next_w, next_state, next_alpha_past))

        # 更新了多模型
        if stochastic:
            pass
            # if argmax:
            #     nw = next_p[0].argmax()
            # else:
            #     nw = next_w[0]
            # sample.append(nw)
            # sample_score += next_p[0, nw]
            # if nw == 0:
            #     break
        else:
            next_p = total_next_p / len(assemble_list)
            # cand_scores:live_k × num_target = live_k × 1 - live_k × num_target
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            # cand_flat:live_k * num_target
            cand_flat = cand_scores.flatten()
            # 返回按分数排序的最大下标，有可能排名前面候选前一个词相同
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = next_p.shape[1]
            # 理论上不会有超过词表大小的下标
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            # 对应的排序前几的分数，越小越牛逼
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_alpha_past = []

            # ti与wi都是一个数字
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                # 这里需要变成多模型
                temp_list_0 = []
                temp_list_1 = []
                for index, item in enumerate(assemble_list):
                    temp_list_0.append(copy.copy(item[2][ti]))
                    temp_list_1.append(copy.copy(item[3][ti]))
                new_hyp_states.append(temp_list_0)
                new_hyp_alpha_past.append(temp_list_1)

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_alpha_past = []

            for idx in xrange(len(new_hyp_samples)):
                # 对每个候选进行判断
                if new_hyp_samples[idx][-1] == 0:  # <eol>
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_alpha_past.append(new_hyp_alpha_past[idx])

            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])

            # 每个候选
            # for idx, item in enumerate(hyp_states):
            #     # 候选里的每个模型
            #     for jdx, _ in enumerate(hyp_states[idx]):
            #         hyp_states[idx][jdx] = numpy.array(hyp_states[idx][jdx])
            #         hyp_alpha_past[idx][jdx] = numpy.array(hyp_alpha_past[idx][jdx])
            next_state_list = np.transpose(numpy.array(hyp_states), [1, 0, 2])
            next_alpha_past_list = np.transpose(numpy.array(hyp_alpha_past), [1, 0, 2])
    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def conv_norm_weight(kernel_size, nin, nout=None, scale=0.01):
    W = scale * numpy.random.rand(nout, nin, kernel_size, 1)
    return W.astype('float32')


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim * 2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    # coverage conv
    params[_p(prefix, 'conv_Q')] = conv_norm_weight(options['dim_coverage'], 1, dim_nonlin)
    params[_p(prefix, 'conv_Uf')] = norm_weight(dim_nonlin, dimctx)
    params[_p(prefix, 'conv_b')] = numpy.zeros((dimctx,)).astype('float32')

    # when to attention
    params[_p(prefix, 'Wyg')] = norm_weight(nin_nonlin, 2 * options['dim_enc'][-1])
    params[_p(prefix, 'byg')] = numpy.zeros((2 * options['dim_enc'][-1],)).astype('float32')
    params[_p(prefix, 'Whg')] = norm_weight(dim_nonlin, 2 * options['dim_enc'][-1])
    params[_p(prefix, 'bhg')] = numpy.zeros((2 * options['dim_enc'][-1],)).astype('float32')
    params[_p(prefix, 'Umg')] = norm_weight(dim_nonlin, 2 * options['dim_enc'][-1])
    params[_p(prefix, 'W_m_att')] = norm_weight(2 * options['dim_enc'][-1], dimctx)
    params[_p(prefix, 'U_when_att')] = norm_weight(dimctx, 1)
    params[_p(prefix, 'c_when_att')] = numpy.zeros((1,)).astype('float32')
    return params


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    # state_below = batch_size(8)×500
    # tparams[_p(prefix, 'W') = 500 × 256
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.tensor3('x', dtype='float32')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    h = x
    hr = xr
    hidden_sizes = options['dim_enc']

    for i in range(len(hidden_sizes)):
        proj = get_layer(options['encoder'])[1](tparams, h, options,
                                                prefix='encoder' + str(i))
        # word embedding for backward rnn (source)
        projr = get_layer(options['encoder'])[1](tparams, hr, options,
                                                 prefix='encoder_r' + str(i))

        h = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)
        if options['down_sample'][i] == 1:
            h = h[0::2]
        hr = h[::-1]

    ctx = h
    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    # 1× 256
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx, h]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    alpha_past = tensor.matrix('alpha_past', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    # (1, 256)
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state, alpha_past=alpha_past)
    # batch_size(1)× 256
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]
    next_alpha_past = proj[3]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')

    logit = logit_lstm + logit_prev + logit_ctx

    # maxout layer
    shape = logit.shape
    shape1 = tensor.cast(shape[1] / 2, 'int64')
    shape2 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0], shape1, shape2])  # batch*256 -> batch*128*2
    logit = logit.max(2)  # batch*500
    # (1, 111)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # (1, 111)
    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)
    # 把最大位置1，其余全部置0
    what = trng.multinomial(pvals=next_probs)
    # sample from softmax distribution to get the sample

    next_sample = what.argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state, alpha_past]
    outs = [next_probs, next_sample, next_state, next_alpha_past]
    f_next = theano.function(inps, outs, name='f_next', profile=profile, on_unused_input='ignore')
    print 'Done'

    return f_init, f_next


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    # W = nin × 2 * dim
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    # U = dim × 2*dim
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    # dim = 500
    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    # tparams[_p(prefix, 'W')] = 9 × 500
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    # tparams[_p(prefix, 'Wx')] = 9 × 250
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    # x_ = 8 × 500
    # xx_ = 8 × 250
    # h_ = 8 × 250
    # U = 250 × 500
    # Ux = 250× 250
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


def load_single_model(model_path):
    with open('%s.pkl' % model_path, 'rb') as f:
        options = pkl.load(f)
    params_dict = {}
    params = init_params(options, params_dict)
    params = load_params(model_path, params)
    tparams = init_tparams(params)
    trng = RandomStreams(1234)
    # build_sample出错
    f_init, f_next = build_sampler(tparams, options, trng)
    return f_init, f_next, options, trng


# state_below = seq_y × batch_size × 256
# mask = seq_y * batch_size(8)
# context = seq_x × batch_size(8) × 500 实际上也是隐层得到的状态
# context_mask = seq_x × batch_size(8)
# init_state = batch_size × 256
def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None, alpha_past=None,
                   context_mask=None,
                   **kwargs):
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # tparams[_p(prefix, 'Wcx')]  = 500 × 256
    dim = tparams[_p(prefix, 'Wcx')].shape[1]
    dimctx = tparams[_p(prefix, 'Wcx')].shape[0]
    # tparams[_p(prefix, 'conv_Q')] = 256 × 1 × 121 × 1
    pad = (tparams[_p(prefix, 'conv_Q')].shape[2] - 1) / 2

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    # alpha_past = batch_size × seq_x
    if alpha_past is None:
        alpha_past = tensor.alloc(0., n_samples, context.shape[0])

    # tparams[_p(prefix, 'Wc_att')] = 500 × 500
    # pctx_ = seq_x × batch_size(8) × 500
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
            tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # projected x

    # state_belowx = seq_y × batch_size × 256 = seq_y × batch_size × 256 dot 256 × 256
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
                   tparams[_p(prefix, 'bx')]
    # state_below_ = seq_y × batch_size × 512 = seq_y × batch_size × 256 dot 256 × 512
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                   tparams[_p(prefix, 'b')]
    # state_belowyg = seq_y × batch_size × 500 = seq_y × batch_size × 256 dot 256 × 500
    state_belowyg = tensor.dot(state_below, tparams[_p(prefix, 'Wyg')]) + \
                    tparams[_p(prefix, 'byg')]

    # m_ = batch_size
    # x_ = batch_size × 512
    # xx_ = batch_size × 256
    # yg = batch_size × 500
    # h_ = batch_size × 256
    # ctx_ = batch_size × 500
    # alpha = batch_size × seq_x
    # alpha_past_ = batch_size × seq_x
    # beta = batch_size
    # pctx_ = seq_x × batch_size(8) × 500
    # cc_ = seq_x × batch_size(8) × 500
    # U = 256 × 512
    # Wc = 500 × 512
    # W_comb_att = 256 × 500
    # U_att = 500 × 1
    # c_tt = 1
    # Ux = 256 × 256
    # Wcx = 500 × 256
    # U_nl = 256 × 512
    # Ux_nl = 256 × 256
    # b_nl = 512
    # bx_nl = 256
    # conv_Q = 256 × 1 × 121 × 1
    # conv_Uf = 256 × 500
    # conv_b = 500
    # Whg = 256 × 500
    # bhg = 500
    # Umg = 256 × 500
    # W_m_att = 500×500
    # U_when_att = 500 × 1
    # c_when_att = 1


    def _step_slice(m_, x_, xx_, yg, h_, ctx_, alpha_, alpha_past_, beta, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl, conv_Q, conv_Uf, conv_b,
                    Whg, bhg, Umg, W_m_att, U_when_att, c_when_att):
        # preact1 = batch_size × 512
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)  # reset gate
        u1 = _slice(preact1, 1, dim)  # update gate

        # preact1 = batch_size × 256
        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        # h1 = batch_size × 256

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # gm = batch_size × 500 = batch_size × 256 dot 256 × 500
        g_m = tensor.dot(h_, Whg) + bhg
        g_m += yg
        g_m = tensor.nnet.sigmoid(g_m)
        # mt = batch_size × 500 = batch_size × 256 dot 256 × 500
        mt = tensor.dot(h1, Umg)
        mt = tensor.tanh(mt)
        mt *= g_m
        # attention
        # pstate_ = batch_size × 500
        pstate_ = tensor.dot(h1, W_comb_att)

        # converage vector
        # batch_size × in_chancel × width × height 过卷积 out_channel × in_channel × width × height
        # batch_size × 1 × seq_x × 1 过卷积 256 × 1 × 121 × 1
        # cover_F =  batch_size × 256 × seq_x × 1
        cover_F = theano.tensor.nnet.conv2d(alpha_past_[:, None, :, None], conv_Q,
                                            border_mode='half')  # batch x dim x SeqL x 1

        cover_F = cover_F.dimshuffle(1, 2, 0, 3)  # dim(256) x seq_x x batch_size x 1
        # cover_F = # dim(256) x seq_x x batch_size
        cover_F = cover_F.reshape([cover_F.shape[0], cover_F.shape[1], cover_F.shape[2]])
        assert cover_F.ndim == 3, \
            'Output of conv must be 3-d: #dim x SeqL x batch'
        # cover_F = cover_F[:,pad:-pad,:]
        # cover_F = # seq_x × batch_size × dim(256)
        cover_F = cover_F.dimshuffle(1, 2, 0)
        # cover_F must be SeqL x batch x dimctx
        # cover_vector = Seqx x batch x 500
        cover_vector = tensor.dot(cover_F, conv_Uf) + conv_b
        # cover_vector = cover_vector * context_mask[:,:,None]

        # seq_x × batch_size(8) × 500 + 1 × batch_size(8) × 500 + Seqx x batch x 500
        pctx__ = pctx_ + pstate_[None, :, :] + cover_vector
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        # alpha = seq_x × batch_size × 1
        alpha = tensor.dot(pctx__, U_att) + c_tt
        # compute alpha_when
        # pctx_when = batch_size × 500
        pctx_when = tensor.dot(mt, W_m_att)
        # pstate_ = batch_size × 500
        pctx_when += pstate_
        pctx_when = tensor.tanh(pctx_when)
        # alpha_when = batch_size × 1
        alpha_when = tensor.dot(pctx_when, U_when_att) + c_when_att  # batch * 1

        # alpha = Seq_x × batch
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])  # Seq_x * batch
        # alpha = Seq_x × batch
        alpha = tensor.exp(alpha)
        alpha_when = tensor.exp(alpha_when)
        if context_mask:
            alpha = alpha * context_mask
        if context_mask:
            alpha_mean = alpha.sum(0, keepdims=True) / context_mask.sum(0, keepdims=True)
        else:
            # alpha_mean = 1 × batch_size
            alpha_mean = alpha.mean(0, keepdims=True)
        # alpha_when = (1+1)×batch
        alpha_when = concatenate([alpha_mean, alpha_when.T], axis=0)  # (SeqL+1)*batch
        # alpha = Seq_x × batch
        alpha = alpha / alpha.sum(0, keepdims=True)
        # 2 × batch_size
        alpha_when = alpha_when / alpha_when.sum(0, keepdims=True)
        # beta = batch_size
        beta = alpha_when[-1, :]
        # alpha_past = batch × Seql
        alpha_past = alpha_past_ + alpha.T
        # ctx_ = batch_size(8) × 500 = (seq_x × batch_size(8) × 500 * seq_x ×batch_size × 1).sum(0)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context
        # batch_size × 1 * batch_size × 500 + ...
        # ctx_ = batch_size × 500
        ctx_ = beta[:, None] * mt + (1. - beta)[:, None] * ctx_

        # preact2 = batch_size × 512 = batch_size × 256 * 256 × 512
        preact2 = tensor.dot(h1, U_nl) + b_nl
        # preact2 = batch_size × 512 = batch_size × 500 * 500 × 512
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)
        # preactx2 = batch_size × 256 = batch_size × 256 * 256 × 256
        preactx2 = tensor.dot(h1, Ux_nl) + bx_nl
        preactx2 *= r2
        # preactx2 += batch_size × 256 = batch_size × 500 * 500 × 256
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T, alpha_past, beta  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx, state_belowyg]
    # seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')],
                   tparams[_p(prefix, 'conv_Q')],
                   tparams[_p(prefix, 'conv_Uf')],
                   tparams[_p(prefix, 'conv_b')],
                   tparams[_p(prefix, 'Whg')],
                   tparams[_p(prefix, 'bhg')],
                   tparams[_p(prefix, 'Umg')],
                   tparams[_p(prefix, 'W_m_att')],
                   tparams[_p(prefix, 'U_when_att')],
                   tparams[_p(prefix, 'c_when_att')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, alpha_past, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples, )],
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])

    print 'total words/phones', len(lexicon)
    return lexicon


def main(model_list, dictionary_target, source_fea, source_latex, saveto, wer_file, k=5):
    # load source dictionary and invert
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    valid, valid_uid_list = dataIterator_valid(source_fea, source_latex,
                                               worddicts, batch_size=1, maxlen=2000)

    model_function_list = []
    # 将list中的模型加载
    for x in model_list:
        temp_f_init, temp_f_next, temp_options, temp_trng = load_single_model(x)
        model_function_list.append((temp_f_init, temp_f_next, temp_options, temp_trng))

    fpp_sample = open(saveto, 'w')
    valid_count_idx = 0

    print 'Decoding...'
    ud_epoch = 0
    ud_epoch_start = time.time()
    # x:batch_size(1) × seq_x ×9
    for x, y in valid:
        for xx in x:
            print '%d : %s' % (valid_count_idx + 1, valid_uid_list[valid_count_idx])
            xx_pad = numpy.zeros((xx.shape[0] + 1, xx.shape[1]), dtype='float32')
            # 最后一维为0，其余与xx一样
            xx_pad[:xx.shape[0], :] = xx
            stochastic = False
            sample, score = gen_sample(model_function_list,
                                       xx_pad[:, None, :], k=k,
                                       maxlen=1000,
                                       stochastic=stochastic,
                                       argmax=False)

            if stochastic:
                ss = sample
            else:
                score = score / numpy.array([len(s) for s in sample])
                ss = sample[score.argmin()]

            fpp_sample.write(valid_uid_list[valid_count_idx])
            valid_count_idx = valid_count_idx + 1
            for vv in ss:
                if vv == 0:  # <eol>
                    break
                fpp_sample.write(' ' + worddicts_r[vv])
            fpp_sample.write('\n')
    fpp_sample.close()
    ud_epoch = (time.time() - ud_epoch_start) / 60.
    print 'test set decode done, cost time ...', ud_epoch
    os.system('python compute-wer.py ' + saveto + ' ' + source_latex + ' ' + wer_file)
    fpp = open(wer_file)
    stuff = fpp.readlines()
    fpp.close()
    m = re.search('WER (.*)\n', stuff[0])
    valid_per = 100. * float(m.group(1))
    m = re.search('ExpRate (.*)\n', stuff[1])
    valid_sacc = 100. * float(m.group(1))

    print 'Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_per, valid_sacc)


if __name__ == "__main__":
    dictionary_target = "../data/dictionary.txt"
    source_fea = "../data/online-test.pkl"
    source_latex = "../data/test_caption.txt"
    saveto = "./result/test_decode_result.txt"
    wer_file = "./result/test.wer"
    model_list = ["./models/attention_maxlen[400]_dimWord256_dim256_assemble0.npz",
                  "./models/attention_maxlen[400]_dimWord256_dim256_assemble1.npz",
                  "./models/attention_maxlen[400]_dimWord256_dim256_assemble2.npz"]
    main(model_list, dictionary_target, source_fea, source_latex, saveto, wer_file, k=10)
