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


	(_UNARY_WEIGHT, Neg),
	(_UNARY_WEIGHT, BitwiseNeg),

	(_DIV_WEIGHT, Div),
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





def generate_tvm_expr():
	
	def _generate_expr(depth):
		
		expr = _expr_selection.select()
		if(depth > 100 or expr == None):
			return _nonrecursive_selection.select().apply()
		
		if(expr.nargs == 0):
			return expr.apply()
		elif(expr.nargs == 1):
			a_tvm = _generate_expr(depth + 1)
			b_tvm = None
			return expr.apply(a)
		elif(expr.nargs == 2):
			a_tvm = _generate_expr(depth + 1)
			b_tvm = _generate_expr(depth + 1)
			if(not isinstance(a,tvm.expr.ExprOp) and not isinstance(b,tvm.expr.ExprOp)):
				#if both are python types then it's failing to test tvm so replace with tvm expr
				if(random.random() >= 0.5):
					a_tvm = ExistingVar.apply(depth + 1)
				else:
					b_tvm = ExistingVar.apply(depth + 1)
			return expr.apply(a,b)

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
			return (expr.apply(a_tvm,b_tvm),expr.apply_np(a_np,b_np))

	return _generate_expr(0)


if __name__ == "__main__":
	try:
		print(generate_tvm_expr())
	except Exception:
		traceback.print_exc()
