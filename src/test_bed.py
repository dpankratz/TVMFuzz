import tvm
from tvm import te,tir
import numpy as np

def evaluate_tvm_expr(expr,tvm_vars,np_var_binds):
	
	dtype = None
	if(isinstance(expr,tir.expr.ExprOp)):
		dtype = expr.dtype
	else:
		return expr

	shape = (1,)
	c = te.compute(shape,lambda i: expr)
	s = te.create_schedule([c.op])
	f = tvm.build(s,tvm_vars + [c])
	c_tvm= tvm.nd.array(np.zeros(shape,dtype))

	tvm_binds = []
	for var in tvm_vars:
		value = np_var_binds[var.name]
		if(var.dtype == 'uint32' or var.dtype == 'int32'):
			tvm_binds.append(int(value))
		elif(var.dtype == "float"):
			tvm_binds.append(float(value))
		else:
			tvm_binds.append(value)

	f(*(tvm_binds + [c_tvm]))
	return  c_tvm.asnumpy()[0]

def compare_results(result_one,result_two):
	if(isinstance(result_one,float) or isinstance(result_two,float)):
		return np.isclose(result_one, result_two)
	return result_one == result_two

if (__name__ == "__main__"):
	a = te.var(name="a",dtype='int32')
	assert evaluate_tvm_expr(tir.const(5) + a,[a],{"a":5}) == 10

