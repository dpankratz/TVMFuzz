"""
Representation of tvm exprs that can be combined to produce random tvm exprs or random numpy exprs
Tvm builds the ast by overriding python operators between tvm.expr.ExprOp and python types

To have a matching numpy implementation we recreate the AST as a tree of lambda calls such that populating the variables and calling the root node
will return the same output as the tvm Expr

"""


import tvm
from ProbabilisticSelection import ProbabilisticSelection
import random
from math import floor,ceil
from SymbolTable import SymbolTable

def _suppress_zero(rhs):
	if(rhs == 0):
		return 1
	if(isinstance(rhs,(tvm.expr.IntImm,tvm.expr.FloatImm)) and ceil(rhs.value) == 0):
		return 1
	return rhs

def _force_int(e, np=False):
	if(isinstance(e,int)):
		return e
	if(isinstance(e,tvm.expr.ExprOp) and 'int' in e.dtype):
		return e
	return tvm.expr.Cast('int32',e)

def _force_float(e,np=False):
	if(isinstance(e,float)):
		return e
	if(isinstance(e,tvm.expr.ExprOp) and 'float' in e.dtype):
		return e
	return tvm.expr.Cast('float32',e)


#TODO: Numpy promotion e.g. 10 & 10.0 should turn into 10.0 & 10.0 according to TVM rules

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
		return ~e

	def apply_np(e):
		return lambda : ~e()

class Abs(UnaryOp):
	def apply(e):
		return tvm.abs(e)

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
		return lhs - rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() - rhs()

class Mul(BinaryOp):
	def apply(lhs,rhs):
		return lhs * rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() * rhs()

class Div(BinaryOp):
	def apply(lhs,rhs):
		return lhs / _force_float(_suppress_zero(rhs))

	def apply_np(lhs,rhs):
		return lambda : lhs() / _force_float(_suppress_zero(rhs()))

class Mod(BinaryOp):
	def apply(lhs,rhs):
		return _force_int(lhs) % _force_int(_suppress_zero(rhs))

	def apply_np(lhs,rhs):
		return lambda : _force_int(lhs()) % _force_int(_suppress_zero(rhs()))

class Min(BinaryOp):
	def apply(lhs,rhs):
		return tvm.min(lhs,rhs)

	def apply_np(lhs,rhs):
		return lambda : min(lhs(),rhs())

class Max(BinaryOp):
	def apply(lhs,rhs):
		return tvm.max(lhs,rhs)

	def apply_np(lhs,rhs):
		return lambda : max(lhs(),rhs())


#import DIV information https://github.com/apache/incubator-tvm/issues/3977
class FloorDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.floordiv(a,b)

	def apply_np(lhs,rhs):
		return lambda : floor(int(lhs()) / int(_suppress_zero(rhs())))

class FloorMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.floormod(a,b)

	def apply_np(lhs,rhs):
		return lambda : floor(int(lhs()) % int(_suppress_zero(rhs())))

class TruncDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.truncdiv(a,b)

	def apply_np(lhs,rhs):
		return lambda : int(int(lhs()) / int(_suppress_zero(rhs())))

class TruncMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.truncmod(a,b)

	def apply_np(lhs,rhs):
		return lambda : floor(int(lhs()) % int(_suppress_zero(rhs())))

class IndexDiv(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.indexdiv(a,b)

	def apply_np(lhs,rhs):
		return lambda : int(int(lhs()) / int(_suppress_zero(rhs())))

class IndexMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _force_int(_suppress_zero(rhs))
		return tvm.indexmod(a,b)		

	def apply_np(lhs,rhs):
		return lambda : floor(int(lhs()) % int(_suppress_zero(rhs())))

class BitwiseAnd(BinaryOp):
	def apply(lhs,rhs):
		return lhs & rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() & rhs()

class BitwiseOr(BinaryOp):
	def apply(lhs,rhs):
		return lhs | rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() | rhs()

class BitwiseXor(BinaryOp):
	def apply(lhs,rhs):
		return lhs ^ rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() ^ rhs()

class ShiftRight(BinaryOp):
	def apply(lhs,rhs):
		return lhs >> rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() >> rhs()

class ShiftLeft(BinaryOp):
	def apply(lhs,rhs):
		return rhs << lhs

	def apply_np(lhs,rhs):
		return lambda : lhs() << rhs()

class GT(BinaryOp):
	def apply(lhs,rhs):
		return rhs > lhs

	def apply_np(lhs,rhs):
		return lambda : lhs() > rhs()	

class GE(BinaryOp):
	def apply(lhs,rhs):
		return rhs >= lhs

	def apply_np(lhs,rhs):
		return lambda : lhs() >= rhs()	

class LT(BinaryOp):
	def apply(lhs,rhs):
		return rhs < lhs

	def apply_np(lhs,rhs):
		return lambda : lhs() < rhs()	

class LE(BinaryOp):
	def apply(lhs,rhs):
		return rhs <= lhs

	def apply_np(lhs,rhs):
		return lambda : lhs() <= rhs()	

class EQ(BinaryOp):
	def apply(lhs,rhs):
		return rhs == lhs

	def apply_np(lhs,rhs):
		return lambda : (lhs() == rhs()).asobject() # EqualOp is deferred eqOp in python

class NE(BinaryOp):
	def apply(lhs,rhs):
		return rhs != lhs

	def apply_np(lhs,rhs):
		return lambda : (lhs() != rhs()).asobject() # EqualOp is deferred 

class Pow(BinaryOp):
	def apply(lhs,rhs):
		#This is tvm.pow in c++ but tvm.power in python
		return tvm.power(_force_float(lhs),_force_float(rhs))

	def apply_np(lhs,rhs):
		return lambda : float(lhs()) ** float(rhs())

class Literal(object):
	nargs = 0

	_last_random = None

	def filter():
		return True

	def apply():
		raise NotImplementedError

	def apply_np():
		raise NotImplementedError

class IntLit(Literal):
	def apply():
		IntLit._last_random = random.randint(-100,100)
		return IntLit._last_random

	def apply_np():
		return lambda : IntLit._last_random

class BoolLit(Literal):
	def apply():
		BoolLit._last_random = random.random() >= 0.5
		return BoolLit._last_random

	def apply_np():
		return lambda : BoolLit._last_random

class NewVar(Literal):

	_last_random = None

	_var_selection = ProbabilisticSelection([
			(1,lambda : tvm.var(name="f"+str(len(SymbolTable.variables)),dtype='float32')),
			(1,lambda : tvm.var(name="i"+str(len(SymbolTable.variables)),dtype='int32'))
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

	_last_random = None

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