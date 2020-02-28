# 2/28/2020
# Bug descrption: Bitshifting by a negative int
# PR: 


import tvm
import numpy as np
import sys

def test_negative_bitshift():
	shape = (1,)

	a = tvm.var(name='a',dtype='int32')
	b = tvm.var(name='b',dtype='int32')
	c = tvm.compute(shape,lambda i: a << b)
	s = tvm.create_schedule([c.op])
	left_shift = tvm.build(s,[a,b,c])

	c = tvm.compute(shape,lambda i: a >> b)
	s = tvm.create_schedule([c.op])
	right_shift = tvm.build(s,[a,b,c])

	def test_right_shift(lhs,rhs):
		c_tvm= tvm.nd.array(np.zeros(shape,dtype='int32'))
		right_shift(lhs,rhs,c_tvm)
		frontend_value = (tvm.const(lhs) >> tvm.const(rhs)).value
		runtime_value = c_tvm.asnumpy()[0]
		print(frontend_value,runtime_value)
		#assert runtime_value == frontend_value

	def test_left_shift(lhs,rhs):
		c_tvm= tvm.nd.array(np.zeros(shape,dtype='int32'))
		left_shift(lhs,rhs,c_tvm)
		frontend_value = (tvm.const(lhs) << tvm.const(rhs)).value
		runtime_value = c_tvm.asnumpy()[0]
		print(frontend_value,runtime_value)
		#assert runtime_value == frontend_value
	

	for test in [(10,-3),(-10,-10),(-3,-1),(-1,4)]:
		test_left_shift(*test)
		test_right_shift(*test)

if (__name__ == "__main__"):
	test_negative_bitshift()
