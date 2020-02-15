import tvm
import random
import traceback
import sys 
from ProbabilisticSelection import ProbabilisticSelection
from Expression import *

_BINARY_OP_WEIGHT = 1
_DIV_WEIGHT = 1
_UNARY_WEIGHT = 1
_LIT_WEIGHT = 4
_VAR_WEIGHT = 4
_PLACEHOLDER_WEIGHT = 4

# Specify the generation proabilities of each respective option

_expr_selection = ProbabilisticSelection([
	(_BINARY_OP_WEIGHT , Add),
	(_BINARY_OP_WEIGHT , Sub),
	(_BINARY_OP_WEIGHT , Mul),
	(_BINARY_OP_WEIGHT , BitwiseAnd),
	(_BINARY_OP_WEIGHT , BitwiseOr),
	(_BINARY_OP_WEIGHT , BitwiseXor),
	#(_BINARY_OP_WEIGHT , ShiftRight), #Currently bugged. Can re-add after PR
	#(_BINARY_OP_WEIGHT , ShiftLeft),
	(_BINARY_OP_WEIGHT , Min),
	(_BINARY_OP_WEIGHT , Max),
	(_BINARY_OP_WEIGHT , GT),
	(_BINARY_OP_WEIGHT , GE),
	(_BINARY_OP_WEIGHT , LT),
	(_BINARY_OP_WEIGHT , LE),
	(_BINARY_OP_WEIGHT , EQ),
	(_BINARY_OP_WEIGHT , NE),
	(_BINARY_OP_WEIGHT , Pow),


	(_UNARY_WEIGHT, Neg),
	(_UNARY_WEIGHT, BitwiseNeg),
	(_UNARY_WEIGHT, Abs),

	(_DIV_WEIGHT, Div), # is this even supported by tvm?
	#(_DIV_WEIGHT, Mod), #Currently bugged
	(_DIV_WEIGHT, FloorDiv),
	(_DIV_WEIGHT, FloorMod),
	(_DIV_WEIGHT, IndexDiv),
	(_DIV_WEIGHT, IndexMod),
	(_DIV_WEIGHT, TruncDiv),
	(_DIV_WEIGHT, TruncMod),
	(_LIT_WEIGHT, IntLit),
	(_LIT_WEIGHT, BoolLit),
	(_VAR_WEIGHT, ExistingVar),

	
	(1 , NewVar),

	(4, None) #Terminate recursion

])

_nonrecursive_selection = ProbabilisticSelection([
	(_LIT_WEIGHT, IntLit),
	(_LIT_WEIGHT, BoolLit),
	(_VAR_WEIGHT, ExistingVar)
])



def get_expr_type_as_str(e):
	"""
	Tvm types are string literals but as the AST is being built there are also python literals that exist.
	Thus this function unifies the behaviour by returning the tvm string type is there is one or converting the python type to a string.
	"""
	if(isinstance(e,tvm.expr.ExprOp)):
		return e.dtype
	return type(e).__name__

def generate_tvm_expr():
		
	def _generate_expr(depth):
		global a_tvm,b_tvm
		expr = _expr_selection.select()
		if(depth > 100 or expr == None):
			return _nonrecursive_selection.select().apply()
		
		if(expr.nargs == 0):
			return expr.apply()
		elif(expr.nargs == 1):
			a_tvm = _generate_expr(depth + 1)
			b_tvm = None
			return expr.apply(a_tvm)
		elif(expr.nargs == 2):
			a_tvm = _generate_expr(depth + 1)
			b_tvm = _generate_expr(depth + 1)
			if(not isinstance(a_tvm,tvm.expr.ExprOp) and not isinstance(b_tvm,tvm.expr.ExprOp)):
				#if both are python types then it's failing to test tvm so replace with tvm expr
				if(random.random() >= 0.5):
					a_tvm = ExistingVar.apply()
				else:
					b_tvm = ExistingVar.apply()
			return expr.apply(a_tvm,b_tvm)

	return _generate_expr(0)

def generate_tvm_and_np_expr():
	
	def _generate_expr(depth):
		
		expr = _expr_selection.select()
		if(depth > 100 or expr == None):
			term = _nonrecursive_selection.select()
			return (term.apply(), term.apply_np())

		if(expr.nargs == 0):
			return (expr.apply(),expr.apply_np())
		elif(expr.nargs == 1):
			a_tvm,a_np = _generate_expr(depth + 1)
			return (expr.apply(a_tvm),expr.apply_np(a_np))
		elif(expr.nargs == 2):
			a_tvm,a_np = _generate_expr(depth + 1)
			b_tvm,b_np = _generate_expr(depth + 1)
			if(not isinstance(a_tvm,tvm.expr.ExprOp) and not isinstance(b_tvm,tvm.expr.ExprOp)):
				#if both are python types then it's failing to test tvm so replace with tvm expr
				if(random.random() >= 0.5):
					a_tvm = ExistingVar.apply()
					a_np = ExistingVar.apply_np()
				else:
					b_tvm = ExistingVar.apply()
					b_np = ExistingVar.apply_np()

			# need to add type promotion to match TVM
			if("float" in get_expr_type_as_str(a_tvm) and "int" in get_expr_type_as_str(b_tvm)):
				b = b_np
				b_np = lambda : float(b())
			elif("int" in get_expr_type_as_str(a_tvm) and "float" in get_expr_type_as_str(b_tvm)):
				a = a_np
				a_np = lambda : float(a())

			# TODO: bool promotion

			return (expr.apply(a_tvm,b_tvm),
						expr.apply_np(a_np,b_np)) 

	return _generate_expr(0)


if __name__ == "__main__":
	try:
		while(True):
			print(generate_tvm_expr())
	except Exception:
		traceback.print_exc()
		print("a={0}\nb={1}\n".format(a_tvm,b_tvm))