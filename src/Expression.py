"""
Representation of exprs that can be combined to produce random exprs or random numpy exprs
Tvm builds the ast by overriding python operators between tir.expr.ExprOp and python types

To have a matching numpy implementation we recreate the AST as a tree of lambda calls such that populating the variables and calling the root node
will return the same output as the Expr

"""


import tvm
from tvm import te,tir
from ProbabilisticSelection import ProbabilisticSelection
import random
from math import floor,ceil
from SymbolTable import SymbolTable
from Util import *

def _suppress_zero(rhs):
	if(rhs == 0):
		return 1
	if(isinstance(rhs,(tir.expr.IntImm,tir.expr.FloatImm)) and ceil(rhs.value) == 0):
		return 1
	return rhs

def _force_int(e, np=False):
	if (dtype_is_int(e)):
		return e
	return tir.expr.Cast('int32',e)

def _force_float(e,np=False):
	if (dtype_is_float(e)):
		return e
	return tir.expr.Cast('float32',e)

class UnaryOp(object):
	nargs = 1
	def filter(e):
		return True

	def apply(e):
		raise NotImplementedError

	def apply_np(e):
		return lambda : UnaryOp.apply_np(e)

class Neg(UnaryOp):
	def apply(e):
		return -e

	def apply_np(e):
		return lambda : -e()

class BitwiseNeg(UnaryOp):
	def apply(e):
		return ~_force_int(e)

	def apply_np(e):
		return lambda : ~int(e())

class Abs(UnaryOp):
	def apply(e):
		return tir.abs(e)

	def apply_np(e):
		return lambda : abs(e())

class Not(UnaryOp):
	"""
	Not supported by tvm 
	"""





class BinaryOp(object):
	nargs = 2

	def filter(lhs,rhs):
		return True

	def apply(lhs,rhs):
		raise NotImplementedError

	def apply_np(lhs,rhs):
		raise NotImplementedError

class Add(BinaryOp):
	def apply(lhs,rhs):
		return lhs + rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() + rhs()

class Sub(BinaryOp):
	def apply(lhs,rhs):
		return tir.Select(tir.expr.Cast('bool',rhs > lhs), rhs - lhs, lhs - rhs) #Prevent underflow 

	def apply_np(lhs,rhs):
		return lambda : Sub._select(lhs(),rhs())

	def _select(lhs,rhs):
		#Use helper function to avoid evaluating each expression multiple times
		return lhs - rhs if lhs > rhs else rhs - lhs

class Mul(BinaryOp):
	def apply(lhs,rhs):
		return lhs * rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() * rhs()

class Div(BinaryOp):
	"""
	Not supported by tvm, use truncdiv or floordiv
	"""

class Mod(BinaryOp):
	def apply(lhs,rhs):
		return _force_int(lhs) % _suppress_zero(_force_int(rhs))

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) % _suppress_zero(int(rhs()))

class Min(BinaryOp):
	def apply(lhs,rhs):
		return tir.min(lhs,rhs)

	def apply_np(lhs,rhs):
		return lambda : min(lhs(),rhs())

class Max(BinaryOp):
	def apply(lhs,rhs):
		return tir.max(lhs,rhs)

	def apply_np(lhs,rhs):
		return lambda : max(lhs(),rhs())


#import DIV information https://github.com/apache/incubator-tvm/issues/3977
class FloorDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _suppress_zero(_force_int(rhs))
		return tir.floordiv(a,b)

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) // _suppress_zero(int(rhs()))

class FloorMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _suppress_zero(_force_int(rhs))
		return tir.floormod(a,b)

	def apply_np(lhs,rhs):
		return lambda : floor(int(lhs()) % _suppress_zero(int(rhs())))

class TruncDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _suppress_zero(_force_int(rhs))
		return tir.truncdiv(a,b)

	def apply_np(lhs,rhs):
		return lambda : int(int(lhs()) / _suppress_zero(int(rhs())))

class TruncMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _suppress_zero(_force_int(rhs))
		return tir.truncmod(a,b)

	def apply_np(lhs,rhs):
		return lambda : floor(int(lhs()) % abs(_suppress_zero(int(rhs()))))

class IndexDiv(BinaryOp):
	"""
	Implemented as floordiv in tvm  
	"""

class IndexMod(BinaryOp):
	"""
	Implemented as floormod in tvm
	"""

class BitwiseAnd(BinaryOp):
	def apply(lhs,rhs):
		return _force_int(lhs) & _force_int(rhs)

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) & int(rhs())

class BitwiseOr(BinaryOp):
	def apply(lhs,rhs):
		return  _force_int(lhs) | _force_int(rhs)

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) | int(rhs())

class BitwiseXor(BinaryOp):
	def apply(lhs,rhs):
		return _force_int(lhs) ^ _force_int(rhs)

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) ^ int(rhs())

class ShiftRight(BinaryOp):
	def apply(lhs,rhs):
		return  _force_int(lhs) >> tir.abs(_force_int(rhs))

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) >> abs(int(rhs()))

class ShiftLeft(BinaryOp):
	def apply(lhs,rhs):
		return  _force_int(lhs) << tir.abs(_force_int(rhs))

	def apply_np(lhs,rhs):
		return lambda : int(lhs()) << abs(int(rhs()))

class GT(BinaryOp):
	def apply(lhs,rhs):
		return lhs > rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() > rhs()	

class GE(BinaryOp):
	def apply(lhs,rhs):
		return lhs >= rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() >= rhs()	

class LT(BinaryOp):
	def apply(lhs,rhs):
		return lhs < rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() < rhs()	

class LE(BinaryOp):
	def apply(lhs,rhs):
		return lhs <= rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() <= rhs()	

class EQ(BinaryOp):
	def apply(lhs,rhs):
		res = lhs == rhs
		if (isinstance(res,bool)):
			return res
		return res.asobject() # EqualOp is deferred eqOp in python

	def apply_np(lhs,rhs):
		return lambda : lhs() == rhs()

class NE(BinaryOp):
	def apply(lhs,rhs):
		res = lhs != rhs
		if (isinstance(res,bool)):
			return res
		return res.asobject()

	def apply_np(lhs,rhs):
		return lambda : lhs() != rhs()

class Pow(BinaryOp):
	def apply(lhs,rhs):
		#This is tir.pow in c++ but tir.power in python
		return tir.power(_force_float(lhs),_force_int(rhs))

	def apply_np(lhs,rhs):
		return lambda : float(lhs()) ** int(rhs())

class Literal(object):
	nargs = 0

	last_random = None

	def filter():
		return True

	def apply():
		raise NotImplementedError

	def apply_np():
		raise NotImplementedError

class IntLit(Literal):
	def apply():
		IntLit.last_random = random.randint(-100,100)
		return IntLit.last_random

	def apply_np():
		return lambda : IntLit.last_random

class BoolLit(Literal):
	def apply():
		BoolLit.last_random = random.random() >= 0.5
		return BoolLit.last_random

	def apply_np():
		return lambda : BoolLit.last_random

class NewVar(Literal):

	last_random = None

	_var_selection = ProbabilisticSelection([
			(1,lambda : te.var(name="f"+str(len(SymbolTable.variables)),dtype='float32')),
			(1,lambda : te.var(name="i"+str(len(SymbolTable.variables)),dtype='int32'))
		])

	def apply():
		new_var = NewVar._var_selection.select()()
		SymbolTable.variables.append(new_var)
		NewVar._last_random = new_var.name
		return new_var

	def apply_np():
		#this access needs to happen outside of the lambda to materialize the literal since otherwise it will accessing a future value of _last_random
		name = NewVar._last_random
		return lambda : SymbolTable.binds[name]


class ExistingVar(Literal):

	last_random = None

	def apply():
		if(len(SymbolTable.variables) == 0):
			new_var = NewVar.apply()
			ExistingVar._last_random = new_var.name
			return new_var
		rand_var =  SymbolTable.variables[random.randint(0,len(SymbolTable.variables) - 1)]
		ExistingVar._last_random = rand_var.name
		return rand_var

	def apply_np():
		#this access needs to happen outside of the lambda to materialize the literal since otherwise it will accessing a future value of _last_random
		name = ExistingVar._last_random
		return lambda : SymbolTable.binds[name]


if __name__ == "__main__":
	a_tvm = IntLit.apply()
	a_np = IntLit.apply_np()
	b_tvm = ExistingVar.apply()
	b_np = ExistingVar.apply_np()
	c_tvm = Add.apply(a_tvm,b_tvm)
	c_np = Add.apply_np(a_np,b_np)

	i0 = -1
	f0 = -1.0
	print(c_tvm)
	print(c_np())