# 2/15/2020
# Bug descrption: And'ing by a float causes codegen crash
# PR: https://github.com/apache/incubator-tvm/pull/4892


import tvm
from tvm import tir,te
import numpy as np
import sys

try:

	shape = (5,5)

	a = tir.const(dtype='float32',value=10)

	c = te.compute(shape,lambda i,j: a & 1.5 ^ a | ~a) 
	#Also affects | , &, ^, ~

	s = tvm.create_schedule([c.op])

	f = tvm.build(s,[c])

	c_tvm= tvm.nd.array(np.zeros(shape,dtype='float32'))
	f(c_tvm)
	print(c_tvm)
	assert false
except tvm.TVMError:
	pass