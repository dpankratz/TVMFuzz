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


	(_UNARY_WEIGHT, Neg),
	(_UNARY_WEIGHT, BitwiseNeg),

	(_DIV_WEIGHT, FloorDiv),
	(_DIV_WEIGHT, FloorMod),
	(_DIV_WEIGHT, IndexDiv),
	(_DIV_WEIGHT, IndexMod),
	(_DIV_WEIGHT, TruncDiv),
	(_DIV_WEIGHT, TruncMod),
	(_LIT_WEIGHT, IntLit),
	(_VAR_WEIGHT, ExistingVar),
	
	(1 , NewVar),

	(4, None) #Terminate recursion

])

_nonrecursive_selection = ProbabilisticSelection([
	(1, IntLit)
])


depth = 0
a = None
b = None


def generate_expr():
	def _generate_expr():
		global a,b,depth
		depth += 1
		expr = _expr_selection.select()
		if(depth > sys.getrecursionlimit() / 2 or expr == None):
			return _nonrecursive_selection.select().apply()
		
		if(expr.nargs == 0):
			return expr.apply()
		elif(expr.nargs == 1):
			a = _generate_expr()
			b = None
			return expr.apply(a)
		elif(expr.nargs == 2):
			a = _generate_expr()
			b = _generate_expr()
			return expr.apply(a,b)

	depth = 0
	return _generate_expr()

if __name__ == "__main__":
	try:
		print(generate_expr())
	except Exception:
		traceback.print_exc()
		print("a={0}\nb={1}\n".format(a,b))
		