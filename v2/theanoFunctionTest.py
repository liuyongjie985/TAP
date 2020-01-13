# coding:utf-8
import theano
import theano.tensor as T
import numpy as np

# ==============================================================================
print "function Test"
# ==============================================================================




my_r = []
with open('./random.txt', 'r') as f:
    while 1:
        lines = f.readlines(100000)
        if not lines:
            break
        for line in lines:
            temp_list = [np.float(p) for p in line.split()]
            my_r.append(temp_list)
my_r = np.array(my_r, dtype="float32")

U = theano.shared(np.concatenate([my_r, my_r], axis=1), name="U")
Ux = theano.shared(my_r, name="Ux")

mask = T.matrix('x_mask_original', dtype='float32')
state_below_ = T.tensor3('state_below_', dtype='float32')
state_belowx = T.tensor3('state_belowx', dtype='float32')

nsteps = 24
n_samples = 8
# dim = 500
dim = Ux.shape[1]


# utility function to slice a tensor
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


# step function to be used by scan
# arguments    | sequences |outputs-info| non-seqs
# x_ = 8 × 500
# xx_ = 8 × 250
# h_ = 8 × 250
# U = 250 × 500
# Ux = 250× 250
def _step_slice(m_, x_, xx_, h_, U, Ux):
    preact = T.dot(h_, U)
    preact += x_

    # reset and update gates
    r = T.nnet.sigmoid(_slice(preact, 0, dim))
    u = T.nnet.sigmoid(_slice(preact, 1, dim))

    # compute the hidden state proposal
    preactx = T.dot(h_, Ux)
    preactx = preactx * r
    preactx = preactx + xx_

    # hidden state proposal
    h = T.tanh(preactx)

    # leaky integrate and obtain next hidden state
    h = u * h_ + (1. - u) * h
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h


# prepare scan arguments
seqs = [mask, state_below_, state_belowx]
init_states = [T.alloc(0., n_samples, dim)]
_step = _step_slice
shared_vars = [U,
               Ux]

rval, updates = theano.scan(_step,
                            sequences=seqs,
                            outputs_info=init_states,
                            non_sequences=shared_vars,
                            name="encoder0_layers",
                            n_steps=nsteps,
                            profile="False",
                            strict=True)

Test_fn = theano.function([mask, state_below_, state_belowx], rval)

o = np.array([[1 for y in range(8)] for x in range(12)] + [[0 for y in range(8)] for x in range(12)],
             dtype="float32")
p = np.array([0.0000003 for x in range(24 * 8 * 500)], dtype="float32").reshape([24, 8, 500])
q = np.array([0.0000003 for x in range(24 * 8 * 250)], dtype="float32")[::-1].reshape([24, 8, 250])

print "result is"
result = Test_fn(o, p, q)
print result.shape
print result
