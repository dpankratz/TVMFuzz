import tvm
from tvm import te,tir
import random
import traceback
import sys 
from probabilistic_selection import ProbabilisticSelection
from expression import *
from generation_node import GenerationNode
from util import dtype_is_float,dtype_is_int
from tvmfuzz_config import TVMFuzzConfig


_expr_selection = ProbabilisticSelection([
	(TVMFuzzConfig.knary_op_weight, Any),
	(TVMFuzzConfig.knary_op_weight, All),

	(TVMFuzzConfig.trinary_op_weight, Select),
	(TVMFuzzConfig.trinary_op_weight, IfThenElse),

	(TVMFuzzConfig.binary_op_weight, Add),
	(TVMFuzzConfig.binary_op_weight, Sub),
	(TVMFuzzConfig.binary_op_weight, Mul),
	(TVMFuzzConfig.binary_op_weight, BitwiseAnd),
	(TVMFuzzConfig.binary_op_weight, BitwiseOr),
	(TVMFuzzConfig.binary_op_weight, BitwiseXor),
	(TVMFuzzConfig.binary_op_weight, ShiftRight), 
	(TVMFuzzConfig.binary_op_weight, ShiftLeft), 
	(TVMFuzzConfig.binary_op_weight, Min),
	(TVMFuzzConfig.binary_op_weight, Max),
	(TVMFuzzConfig.binary_op_weight, GT),
	(TVMFuzzConfig.binary_op_weight, GE),
	(TVMFuzzConfig.binary_op_weight, LT),
	(TVMFuzzConfig.binary_op_weight, LE),
	(TVMFuzzConfig.binary_op_weight, EQ),
	(TVMFuzzConfig.binary_op_weight, NE),
	(TVMFuzzConfig.binary_op_weight, Pow),
	(TVMFuzzConfig.binary_op_weight, FloorMod),
	(TVMFuzzConfig.binary_op_weight, FloorDiv),
	(TVMFuzzConfig.binary_op_weight, TruncMod),
	(TVMFuzzConfig.binary_op_weight, TruncDiv),

	(TVMFuzzConfig.unary_op_weight, Neg),
	(TVMFuzzConfig.unary_op_weight, BitwiseNeg),
	(TVMFuzzConfig.unary_op_weight, Abs),
	(TVMFuzzConfig.unary_op_weight, Floor),

	(TVMFuzzConfig.literal_weight, IntLit),
	(TVMFuzzConfig.literal_weight, BoolLit),

	(TVMFuzzConfig.existing_var_weight, ExistingVar),

	(TVMFuzzConfig.new_var_weight, NewVar),

	(TVMFuzzConfig.terminal_weight, None) #Terminate recursion		
])

_terminal_selection = ProbabilisticSelection([
	(TVMFuzzConfig.literal_weight, IntLit),
	(TVMFuzzConfig.literal_weight, BoolLit),
	(TVMFuzzConfig.existing_var_weight, ExistingVar)
])


def generate_tvm_and_np_tree():

	def _generate_expr(depth):
		
		expr = _expr_selection.select()
		if(depth > TVMFuzzConfig.expr_depth_limit or expr == None):
			#max number of nodes is given by k ** depth where k is the maximum k-nary expr,
			term = _terminal_selection.select()
			return GenerationNode(term)

		ops = []
		nargs = expr.nargs
		if nargs == -1:
			#variable number of args
			nargs = random.randint(1,5)
		for _ in range(nargs):
			ops.append(_generate_expr(depth + 1))
		return GenerationNode(expr,ops)

	return _generate_expr(0)
