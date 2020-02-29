import tvm
from tvm import te,tir
import random
import traceback
import sys 
from ProbabilisticSelection import ProbabilisticSelection
from Expression import *
from generation_node import GenerationNode
from Util import dtype_is_float,dtype_is_int

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
	(_BINARY_OP_WEIGHT , ShiftRight), 
	(_BINARY_OP_WEIGHT , ShiftLeft),
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

	#(_DIV_WEIGHT, Mod), #Currently bugged
	(_DIV_WEIGHT, FloorDiv),
	(_DIV_WEIGHT, FloorMod),
	(_DIV_WEIGHT, TruncDiv),
	(_DIV_WEIGHT, TruncMod),
	(_LIT_WEIGHT, IntLit),
	(_LIT_WEIGHT, BoolLit),
	(_VAR_WEIGHT, ExistingVar),

	
	(1 , NewVar),

	(8, None) #Terminate recursion

])

_nonrecursive_selection = ProbabilisticSelection([
	(_LIT_WEIGHT, IntLit),
	(_LIT_WEIGHT, BoolLit),
	(_VAR_WEIGHT, ExistingVar)
])


def _handle_underflow(a_tvm,b_tvm,a_np,b_np):
	if(dtype_is_uint(a_tvm) and dtype_is_uint(b_tvm)):
		if(get_literal_value(b_tvm) > get_literal_value(a_tvm)):
			return (b_tvm,a_tvm,b_np,a_np) #handle underflow by switching
	return (a_tvm,b_tvm,a_np,b_np)

def _handle_promotion(a_tvm,b_tvm,a_np,b_np):
	# need to add type promotion to match TVM
	if(dtype_is_float(a_tvm) and not dtype_is_float(b_tvm)):
		b = b_np
		b_np = lambda : float(b())
	elif(not dtype_is_float(a_tvm) and dtype_is_float(b_tvm)):
		a = a_np
		a_np = lambda : float(a())

	return (a_tvm,b_tvm,a_np,b_np)

def _handle_no_tvm(a_tvm,b_tvm,a_np,b_np):
	if(not isinstance(a_tvm,tir.expr.ExprOp) and not isinstance(b_tvm,tir.expr.ExprOp)):
		#if both are python types then it's failing to test tvm so replace with tvm expr
		if(random.random() >= 0.5):
			a_tvm = ExistingVar.apply()
			a_np = ExistingVar.apply_np()
		else:
			b_tvm = ExistingVar.apply()
			b_np = ExistingVar.apply_np()

	return (a_tvm,b_tvm,a_np,b_np)


def handle_special_cases(a_tvm,b_tvm,a_np,b_np,expr):
	a_tvm,b_tvm,a_np,b_np = _handle_no_tvm(a_tvm,b_tvm,a_np,b_np)
	a_tvm,b_tvm,a_np,b_np = _handle_promotion(a_tvm,b_tvm,a_np,b_np)
	if(isinstance(expr,Sub)):
		a_tvm,b_tvm,a_np,b_np = _handle_underflow(a_tvm,b_tvm,a_np,b_np)
	return (a_tvm,b_tvm,a_np,b_np)



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
			if(not isinstance(a_tvm,tir.expr.ExprOp) and not isinstance(b_tvm,tir.expr.ExprOp)):
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


			a_tvm,b_tvm,a_np,b_np = handle_special_cases(a_tvm,b_tvm,a_np,b_np,expr)

			return (expr.apply(a_tvm,b_tvm),
						expr.apply_np(a_np,b_np)) 

	return _generate_expr(0)


def generate_tvm_and_np_tree():

	def _generate_expr(depth):
		
		expr = _expr_selection.select()
		if(depth > 100 or expr == None):
			term = _nonrecursive_selection.select()
			return GenerationNode(term)
		if(expr.nargs == 0):
			return GenerationNode(expr)
		elif(expr.nargs == 1):
			op = _generate_expr(depth + 1)
			return GenerationNode(expr,[op])
		elif(expr.nargs == 2):
			ops = [_generate_expr(depth + 1) , _generate_expr(depth + 1)]
			return GenerationNode(expr,ops)

	return _generate_expr(0)

if __name__ == "__main__":
	try:
		root = generate_tvm_and_np_tree()
		print(root)
		print(root.emit_tvm())
		print(root.emit_np())
	except Exception:
		traceback.print_exc()
		#print("a={0}\nb={1}\n".format(a_tvm,b_tvm))