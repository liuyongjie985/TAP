from theano import tensor as T
from theano.ifelse import ifelse
import theano, time
import numpy as np

a = T.matrix('a')
b = T.matrix('b')
# x, y = T.matrices('x', 'y')
c = a + b

ctx = c

d = T.matrix('d')


def _start(d):
    return d + ctx


f = _start(d)
test_a = theano.function([a, b], c)
test_b = theano.function([d, ctx], [f, c], profile=False, on_unused_input='ignore')

out_c = test_a(np.array(range(10, 20)).reshape([2, -1]), np.array(range(30, 40)).reshape([2, -1]))
print(out_c)
# for i in range(10):
#     out_f, out_c = test_b(np.array(range(30, 40)).reshape([2, -1]) * i)
#     print(out_c)
