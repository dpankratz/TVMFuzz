import tvm
from tvm import te,tir
import numpy as np
import traceback 
import signal
from symboltable import SymbolTable
from expression import Neg, Add, Any
from tvmfuzz_config import TVMFuzzConfig
from termcolor import colored
from datetime import datetime

def evaluate_tvm_expr(expr,suppress_errors = False):
	""" Evaluate given tvm expr

	This uses the binds in SymbolTable to provide the 
	variables with their values during evaluation

	Parameters
	----------
	expr : tvm.expr.PrimExpr
		expr to evaluate
	suppress_errors : bool
		Set to True to not print error traceback
		False to print error tracebac

	Returns
	-------
	result : int or float or bool or str
		result of evaluation
		or "Runtime Error" if there was an 
		error during evaluation
	"""
	
	if len(SymbolTable.binds) == 0:
		SymbolTable.populate()
	
	tvm_vars = SymbolTable.variables
	var_binds = SymbolTable.binds

	dtype = None
	if(isinstance(expr,tir.expr.ExprOp)):
		dtype = expr.dtype
	else:
		return expr

	try: 
		shape = (1,)
		c = te.compute(shape,lambda i: expr)
		s = te.create_schedule([c.op])
		f = tvm.build(s,tvm_vars + [c])
		c_tvm= tvm.nd.array(np.zeros(shape,dtype))

		tvm_binds = []
		for var in tvm_vars:
			value = var_binds[var.name]
			if(var.dtype == 'uint32' or var.dtype == 'int32'):
				tvm_binds.append(int(value))
			elif(var.dtype == "float"):
				tvm_binds.append(float(value))
			else:
				tvm_binds.append(value)


		f(*(tvm_binds + [c_tvm]))
		return  c_tvm.asnumpy()[0]
	except Exception:
		if (not suppress_errors):
			traceback.print_exc()
		return "Runtime Exception"

def evaluate_np_expr(expr, suppress_errors = False):
	""" Evaluate numpy expression

	Parameters
	----------
	expr : function
		Root of numpy expression

	Returns
	-------
	result : int or float or bool or str
		result of evaluation
		or "Runtime Error" if there was an 
		error during evaluation
	"""
	try:
		return expr()
	except Exception:
		traceback.print_exc()
		return "Runtime Exception"

def compare_results(result_one,result_two):
	if(isinstance(result_one,(float,np.float32)) or isinstance(result_two,(float,np.float32))):
		if np.isnan(result_one) and np.isnan(result_two):
			return True
		elif np.isinf(result_one) and np.isinf(result_two):
			if result_one > 0 and result_two > 0:
				return True
			elif result_one < 0 and result_two < 0:
				return True
			return False
		return np.isclose(result_one, result_two)
	return result_one == result_two

if (__name__ == "__main__"):
	a = te.var(name="a",dtype='int32')
	assert evaluate_tvm_expr(tir.const(5) + a,[a],{"a":5}) == 10

