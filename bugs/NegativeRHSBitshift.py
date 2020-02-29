# 2/28/2020
# Bug descrption: Bitshifting left by a negative int. This is caused by ir.IntImm having 64 bit value 
# PR: Decided not to submit since this is a larger issue regarding dealing with overflows 
import tvm
from tvm import tir,te
import numpy as np

def test_negative_bitwise():
	shape = (1,)

	ops = [ (lambda a,b : a ^ b),
		   (lambda a,b : a >> b),
		   (lambda a,b : a << b),
		   (lambda a,b : a | b),
		   (lambda a,b : a & b)
	]

	for j in range(len(ops)):
		op = ops[j]
		a = te.var(name='a',dtype='int32')
		b = te.var(name='b',dtype='int32')
		c = te.compute(shape,lambda i: op(a,b))
		s = te.create_schedule([c.op])
		f = tvm.build(s,[a,b,c])


		for test in [(1,-3),(10,-2),(100,-3),(1000,-1000),(0,-1),(-1,-1)]:
			lhs = test[0]
			rhs = test[1]
			c_tvm= tvm.nd.array(np.zeros(shape,dtype='int32'))
			f(lhs,rhs,c_tvm)
			frontend_value = op(tir.const(lhs,dtype='int32'),tir.const(rhs,dtype='int32'))
			runtime_value = c_tvm.asnumpy()[0]
			
			if (isinstance(frontend_value,tir.expr.IntImm)):
				frontend_value = frontend_value.value
			if (frontend_value != runtime_value):
				print("{0}!={1} for op at index {2}".format(runtime_value,frontend_value,j))

if (__name__ == "__main__"):
	test_negative_bitwise()
