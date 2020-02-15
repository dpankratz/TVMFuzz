# 2//2020
# Bug descrption: Bitshifting by a float is the identity function
# PR: https://github.com/apache/incubator-tvm/pull/4892


import tvm
import numpy as np
import sys

hadError = False

try: 

	shape = (1,1)
	a = tvm.const(dtype='int32',value=10)
	c = tvm.compute(shape,lambda i,j: a << 2.0) #this should either be impossible or materialize as a * (2 ** 1.5)
	s = tvm.create_schedule([c.op])
	f = tvm.build(s,[c])
	c_tvm= tvm.nd.array(np.zeros(shape,dtype='float32'))
	f(c_tvm)
	print(c_tvm)
	assert False
except tvm.TVMError:
	pass
	

