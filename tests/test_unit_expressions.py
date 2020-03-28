import __init__
from expression import *
from test_bed import *
import numpy as np


def _test_ops(a,b,binds):
	for bin_op in [Add,Sub,Mul,Mod,Min,Max,FloorDiv,FloorMod,TruncDiv,TruncMod,
					BitwiseAnd,BitwiseOr,BitwiseXor,ShiftRight,ShiftLeft,
					GT,GE,LT,LE,EQ,NE,Pow]:


		
		tvm_res = evaluate_tvm_expr(bin_op.apply(a,b),[a,b],binds)
		np_res = bin_op.apply_np(lambda : binds['a'], lambda : binds['b'])()

		if (not compare_results(tvm_res,np_res)):
			print("Error in {0} for args {1},{2}. {3} != {4}".format(bin_op.__name__,binds['a'],binds['b'],tvm_res,np_res))

	for unary_op in [Neg,BitwiseNeg,Abs]:
		tvm_res = evaluate_tvm_expr(unary_op.apply(a),[a],binds)
		np_res = unary_op.apply_np(lambda : binds['a'])()

		if (not compare_results(tvm_res,np_res)):
			print("Error in {0} for arg {1}. {2} != {3}".format(unary_op.__name__,binds['a'],tvm_res,np_res))

def test_ops_on_ints():
	a,b = te.var("a"),te.var("b")
	binds = {"a":1,"b":-5}
	_test_ops(a,b,binds)

def test_ops_on_floats():
	a,b = te.var(name='a',dtype='float32'),te.var(name='b',dtype='float32')
	binds = {"a":2.5,"b":-3.5}
	_test_ops(a,b,binds)
	binds = {"a":10.5,"b":1.0}
	_test_ops(a,b,binds)
	binds = {"a":-10.0,"b":2.4}
	_test_ops(a,b,binds)
	binds = {"a":-10.0,"b":1.5}
	_test_ops(a,b,binds)

def test_ops_on_bools():
	a,b = te.var(name='a',dtype='bool'),te.var(name='b',dtype='bool')
	binds = {"a":False,"b":True}
	_test_ops(a,b,binds)
	a,b = te.var(name='a',dtype='bool'),te.var(name='b',dtype='bool')
	binds = {"a":True,"b":False}
	_test_ops(a,b,binds)

if __name__ == "__main__":
	test_ops_on_ints()
	test_ops_on_floats()
	test_ops_on_bools()