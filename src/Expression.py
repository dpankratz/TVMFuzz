import tvm
from ProbabilisticSelection import ProbabilisticSelection
import random,math
from SymbolTable import SymbolTable

def _suppress_zero(rhs):
	if(rhs == 0):
		return 1
	if(isinstance(rhs,(tvm.expr.IntImm,tvm.expr.FloatImm)) and math.ceil(rhs.value) == 0):
		return 1
	return rhs

def _force_int(e):
	if(isinstance(e,int)):
		return e
	if(isinstance(e,tvm.expr.ExprOp) and 'int' in e.dtype):
		return e
	return tvm.expr.Cast('int32',e)


class UnaryOp(object):
	nargs = 1
	def filter(e):
		return True

class Neg(UnaryOp):
	def apply(e):
		return -e

class BitwiseNeg(UnaryOp):
	def apply(e):
		return ~e


class BinaryOp(object):
	nargs = 2

	def filter(lhs,rhs):
		return True

	def apply(lhs,rhs):
		pass

class Add(BinaryOp):
	def apply(lhs,rhs):
		return lhs + rhs

class Sub(BinaryOp):
	def apply(lhs,rhs):
		return lhs - rhs

class Mul(BinaryOp):
	def apply(lhs,rhs):
		return lhs * rhs

class Div(BinaryOp):
	def apply(lhs,rhs):
		return lhs / _suppress_zero(rhs)


class FloorDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.floordiv(a,b)

class FloorMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.floormod(a,b)

class TruncDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.truncdiv(a,b)

class TruncMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.truncmod(a,b)

class IndexDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.indexdiv(a,b)

class IndexMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.indexmod(a,b)		

class BitwiseAnd(BinaryOp):
	def apply(lhs,rhs):
		return lhs & rhs

class BitwiseOr(BinaryOp):
	def apply(lhs,rhs):
		return lhs | rhs

class BitwiseXor(BinaryOp):
	def apply(lhs,rhs):
		return lhs ^ rhs


class Literal(object):
	nargs = 0

	def filter():
		return True

class IntLit(Literal):
	def apply():
		return random.randint(-100,100)

class NewVar(Literal):

	_var_selection = ProbabilisticSelection([
			(1,lambda : tvm.var(name="f"+str(len(SymbolTable.variables)),dtype='float32')),
			(1,lambda : tvm.var(name="i"+str(len(SymbolTable.variables)),dtype='int32'))
		])

	def apply():
		new_var = NewVar._var_selection.select()()
		SymbolTable.variables.append(new_var)
		return  new_var

class ExistingVar(Literal):

	def apply():
		if(len(SymbolTable.variables) == 0):
			return NewVar.apply()
		return SymbolTable.variables[random.randint(0,len(SymbolTable.variables) - 1)]

