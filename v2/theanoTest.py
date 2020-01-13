# coding:utf-8
import theano
import theano.tensor as T
import numpy as np
from nmt import ortho_weight, norm_weight
import pickle

# Calculate A**k

print("Calculate A**k")
k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result

# compiled function that returns A**k
power = theano.function(inputs=[A, k], outputs=final_result, updates=updates)

print power(range(10), 2)
print power(range(10), 4)

# Computing tanh(x(t).dot(W) + b)



print("Computing tanh(x(t).dot(W) + b)")

X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.dot(v, W) + b_sym, sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=[results])

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2
print compute_elementwise(x, w, b)
# comparison with numpy
print x.dot(w) + b

# ==============================================================================
# polynomial -- c0*x^0 + c1*x^1 + c2*x^2 + c3*x^3...
# ==============================================================================
# print("Computing polynomial -- c0*x^0 + c1*x^1 + c2*x^2 + c3*x^3...")
# coefficients = T.vector('coeff')
# x = T.iscalar('x')
# sum_poly_init = T.fscalar('sum_poly')
# result, update = theano.scan(lambda coefficients, power, sum_poly, x: T.cast(sum_poly +
#                                                                              coefficients * (x ** power),
#                                                                              dtype='float32'),
#                              sequences=[coefficients, T.arange(coefficients.size)],
#                              outputs_info=[sum_poly_init],
#                              non_sequences=[x])
#
# poly_fn = theano.function([coefficients, sum_poly_init, x], result, updates=update)
#
# coeff_value = np.asarray([1., 3., 6., 5.], dtype='float32')
# x_value = 3
# poly_init_value = 0.
# print poly_fn(coeff_value, poly_init_value, x_value)

# ==============================================================================
# theano.scan_module.until
# ==============================================================================

print 'theano.scan_module.until:'


def prod_2(pre_value, max_value):
    return pre_value * 2, theano.scan_module.until(pre_value * 2 > max_value)


max_value = T.iscalar('max_value')
result, update = theano.scan(prod_2, outputs_info=T.constant(1.),
                             non_sequences=[max_value], n_steps=100)

prod_fn = theano.function([max_value], result, updates=update)
print prod_fn(400)

# ==============================================================================
# taps scalar -- Fibonacci sequence
# 参数fn是一个你需要计算的函数，一般用lambda来定义，参数是有顺序要求的，先是sequances的参数(y,p)，然后是output_info的参数(x_tm2,x_tm1)，然后是no_sequences的参数(A)。
# sequences就是需要迭代的序列，序列的第一个维度(leading dimension)就是需要迭代的次数。所以，Y和P[::-1]的第一维大小应该相同，如果不同的话，就会取最小的。
# outputs_info描述了需要用到前几次迭代输出的结果，dict(initial=X, taps=[-2, -1])表示使用前一次和前两次输出的结果。如果当前迭代输出为x(t)，则计算中使用了(x(t-1)和x(t-2)。
# non_sequences描述了非序列的输入，即A是一个固定的输入，每次迭代加的A都是相同的。如果Y是一个向量，A就是一个常数，总之，A比Y少一个维度。
# 在有sequence与n_steps的情况下，n_steps <= sequence长度，两个同时起效
# ==============================================================================

Fibo_arr = T.vector('Fibonacci')
k = T.iscalar('n_steps')
result, update = theano.scan(lambda tm2, tm1: tm2 + tm1,
                             outputs_info=[dict(initial=Fibo_arr, taps=[-2, -1])],
                             n_steps=k)
Fibo_fn = theano.function([Fibo_arr, k], result, updates=update)
Fibo_init = np.asarray([1, 1], dtype=theano.config.floatX)
k_value = 12
print Fibo_fn(Fibo_init, k_value)

# ==============================================================================
# my theano Test
# ==============================================================================

print("Calculate A**k")
k = T.iscalar("k")
A = T.vector("A")
S = T.vector("S")
# Symbolic description of the result
result, updates = theano.scan(fn=lambda s, prior_result, A: s * prior_result * A,
                              sequences=S,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result

# compiled function that returns A**k
power = theano.function(inputs=[S, A, k], outputs=final_result, updates=updates)
print range(1, 5)
print power(range(1, 5), range(10), 3)

# ==============================================================================
# mask Test
# ==============================================================================
a = [[[1, 2, 3, 4, 5]], [[4, 5, 6, 7, 8]]]
b = np.array(xrange(20)).reshape([4, -1])
print "a is"
print(a)
print "b is"
print(b)

print a * b

# ==============================================================================
print "sum Test"
# ==============================================================================
b = T.tensor3('a_original', dtype='float32')

c = T.sum(b, axis=0, keepdims=True)
Test_fn = theano.function([b], c)
test_a = np.array(xrange(24), dtype='float32').reshape([2, 3, -1])
print "a is "
print test_a
print "c is"
print Test_fn(test_a)

# ==============================================================================
print "concatenate Test"
# ==============================================================================
b = T.tensor3('a_original', dtype='float32')
result = theano.tensor.concatenate([b, b[::-1]], axis=b.ndim - 1)
Test_fn = theano.function([b], result)
test_a = np.array(xrange(24), dtype='float32').reshape([2, 3, -1])
print "test_a is "
print test_a
print "test_a[::-1] is "
print test_a[::-1]

print "conTest is"
print Test_fn(test_a)

# ==============================================================================
print "None Test"
# ==============================================================================

o = np.array(xrange(24), dtype='float32').reshape([4, -1, ])
print "o[:,:,None] is "
print o[:, :, None].shape
# 4×6 => 4×6×1



# ==============================================================================
print "sum(0) Test"
# ==============================================================================
b = T.tensor3('a_original', dtype='float32')
result = b.sum(0)
Test_fn = theano.function([b], result)
test_a = np.array(xrange(24), dtype='float32').reshape([2, 3, -1])
print "result is "
print Test_fn(test_a)

# ==============================================================================
print "flatten(0) Test"
# ==============================================================================
b = T.matrix('a_original', dtype='float32')
result = b.flatten()
Test_fn = theano.function([b], result)
test_a = np.array(xrange(24), dtype='float32').reshape([2, -1])
print "result is "
print Test_fn(test_a)

# ==============================================================================
print "【：-1】 Test"
# ==============================================================================
test_a = np.array(xrange(24), dtype='float32')
print "test_a[:-1] is "
print test_a[:-1]

# ==============================================================================
print "conv2d Test"
# ==============================================================================

alpha_past_ = theano.tensor.tensor4(name=None, dtype='float32')
conv_Q = theano.tensor.tensor4(name=None, dtype='float32')
cover_F = theano.tensor.nnet.conv2d(alpha_past_, conv_Q, border_mode='half')
cover_F = cover_F.dimshuffle(1, 2, 0, 3)

Test_Fn = theano.function([alpha_past_, conv_Q], cover_F)
a = np.array(xrange(128 * 8), dtype='float32').reshape([8, 1, 128, 1])
b = np.array(xrange(121 * 256), dtype='float32').reshape([256, 1, 121, 1])

result = Test_Fn(a, b)
print "result.dimshuffle(1,2,0,3)"
print result.shape

# ==============================================================================
print "mean(0) Test"
# ==============================================================================
alpha = T.matrix(None, dtype='float32')
# Seql × batch
alpha_mean = alpha.mean(0)

Test_fn = theano.function([alpha], alpha_mean)
test_a = np.array(xrange(24), dtype='float32').reshape([2, -1])
print "result is "
print Test_fn(test_a)

# ==============================================================================
print "【-1.：】 Test"
# ==============================================================================

test_a = np.array(xrange(24), dtype='float32').reshape([2, -1])
print "result is "
print test_a[-1, :]

# ==============================================================================
print "tensor.nnet.softmax Test"
# ==============================================================================
a = T.matrix(None, dtype='float32')
# Seql × batch
a_result = T.nnet.softmax(a)  # (seqL*batch, dim_target)
Test_fn = theano.function([a], a_result)
test_a = np.array(xrange(24), dtype='float32').reshape([2, -1])
print "result is"
print Test_fn(test_a)

# ==============================================================================
print "tensor.nnet.sigmoid Test"
# ==============================================================================
a = T.matrix(None, dtype='float32')
# Seql × batch
# input:(2,12)
a_result = T.nnet.sigmoid(a)  # (seqL*batch, dim_target)
Test_fn = theano.function([a], a_result)
test_a = np.array(xrange(24), dtype='float32').reshape([2, -1])
print "result is"
print Test_fn(test_a)

# ==============================================================================
print "theano.tensor.concatenate() Test"
# ==============================================================================
a = T.matrix(None, dtype='float32')
# Seql × batch
# input:(2,12)
a_result = T.tanh(a)  # (seqL*batch, dim_target)
r = theano.tensor.concatenate([a_result, a_result[::-1]], axis=a_result.ndim - 1)
Test_fn = theano.function([a], r)
test_a = np.array(xrange(24), dtype='float32').reshape([2, -1])
print "result is"
o_r = Test_fn(test_a)
print o_r.shape
print o_r

# ==============================================================================
print "theano.tensor.concatenate() Test"
# ==============================================================================
a = T.tensor3(None, dtype='int32')
# Seql × batch
# input:(2,12)



d = theano.shared(name="d", value=norm_weight(112, 128))
result = d[a.flatten()]

Test_fn = theano.function([a], result)
test_a = np.array(xrange(24), dtype='int32').reshape([2, 3, -1])

print "result is"
o_r = Test_fn(test_a)
print o_r.shape
print o_r

# ==============================================================================
print "tensor.nnet.softmax + tensor.nnet.categorical_crossentropy Test"
# ==============================================================================
logit = T.matrix(None, dtype='float32')
# Seql × batch
# input:(2,12)
label = T.vector(None, dtype="int32")

probs = T.nnet.softmax(logit)  # (seqL * batch, dim_target)

# cost
cost = T.nnet.categorical_crossentropy(probs, label.flatten())  # x is a vector,each value is a 1-of-N position

Test_fn = theano.function([logit, label], cost)

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

print "result is"
c = Test_fn(my_r, np.array(xrange(250), dtype='int32'))
print c.shape
print c

# ==============================================================================
print "theano.tensor.nnet.conv2d Test"
# ==============================================================================
a = T.matrix(None, dtype='float32')

conv_Q = T.tensor4(None, dtype="float32")

# batch_size × in_chancel × width × height 过卷积 out_channel × in_channel × width × height
# batch_size × 1 × seq_x × 1 过卷积 256 × 1 × 121 × 1
# cover_F =  batch_size × 256 × seq_x × 1
cover_F = theano.tensor.nnet.conv2d(a[:, None, :, None], conv_Q,
                                    border_mode='valid')  # batch x dim x SeqL x 1

Test_fn = theano.function([a, conv_Q], cover_F)

print "result is"
# batch_size width in height
# out width in height
c = Test_fn(np.array(xrange(8 * 128), dtype='float32').reshape([8, 128]),
            np.array(xrange(256 * 121), dtype='float32').reshape([256, 1, 121, 1]))
print c.shape
print c

# ==============================================================================
print "theano.tensor.mean Test"
# ==============================================================================
a = T.matrix(None, dtype='float32')

mask = T.alloc(1., 8, 2)

Test_fn = theano.function([a], [a.mean(), mask[:, None]])

print "result is"
# batch_size width in height
# out width in height
c, d = Test_fn(np.array(xrange(8 * 128), dtype='float32').reshape([8, 128]))
print d.shape
print d
