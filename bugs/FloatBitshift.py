# 2/9/2020
# Bug descrption: Bitshifting by a float is the identity function
# PR: TODO


import tvm
import numpy as np
import sys

hadError = False

try: 

	shape = (5,5)

	a = tvm.const(dtype='int32',value=10)

	c = tvm.compute(shape,lambda i,j: a << 1.5) #this should either be impossible or materialize as a * (2 ** 1.5)

	s = tvm.create_schedule([c.op])

	f = tvm.build(s,[c])

	c_tvm= tvm.nd.array(np.zeros(shape,dtype='float32'))
	f(c_tvm)
	print(c_tvm)

except:
	hadError = True

assert hadError

