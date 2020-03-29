"""
Representation of exprs that can be combined to produce random exprs or random numpy exprs
Tvm builds the ast by overriding python operators between tir.expr.ExprOp and python types

To have a matching numpy implementation we can recreate the AST as a tree of lambda calls 
such that populating the variables and calling the root node will return the same output as the Expr

"""


import tvm
from tvm import te,tir
from probabilistic_selection import ProbabilisticSelection
import random
from math import floor,ceil,log2,log10,trunc
from symboltable import SymbolTable
from util import *
import numpy 

int = numpy.int32
float = numpy.float32
bool = numpy.bool


def _suppress_zero(rhs):
	if(isinstance(rhs,(tir.expr.ExprOp))):
		if(dtype_is_float(rhs)):
			return tir.Select(rhs == 0, 4.0, rhs)
		elif(dtype_is_int(rhs)):
			return tir.Select(rhs == 0, 4, rhs)
	return 4 if rhs == 0 else rhs

def _force_int(e):
	if (dtype_is_int(e)):
		return e
	return tir.expr.Cast('int32',e)

def _force_float(e):
	if (dtype_is_float(e)):
		return e
	return tir.expr.Cast('float32',e)

def _force_not_uint(e):
	if dtype_is_uint(e):
		return tir.expr.Cast('int32',e)
	if isinstance(e,(numpy.bool,numpy.bool_,numpy.uint32)):
		return int(e)
	return e

def _clamp_tvm(e,low,high):
	return tir.min(tir.max(e,low),high)

def _clamp_np(e,low,high):
	return min(max(e,low),high)

class UnaryOp(object):
	nargs = 1
	def filter(e):
		return True

	def apply(e):
		raise NotImplementedError

	def apply_np(e):
		raise NotImplementedError

class Neg(UnaryOp):
	def apply(e):
		return -_force_not_uint(e)

	def apply_np(e):
		return lambda : -_force_not_uint(e())

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

class Floor(UnaryOp):
	def apply(e):
		return tir.floor(e)

	def apply_np(e):
		return lambda : floor(e())

class Ceil(UnaryOp):
	def apply(e):
		return tir.ceil(e)

	def apply_np(e):
		return lambda : ceil(e())


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
		return _force_not_uint(lhs) - _force_not_uint(rhs) #Prevent underflow 

	def apply_np(lhs,rhs):
		return lambda : _force_not_uint(lhs()) - _force_not_uint(rhs())


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
		return lambda : floor(int(lhs()) / _suppress_zero(int(rhs())))

#floor mod information https://github.com/apache/incubator-tvm/topi/include/topi/broadcast.h
class FloorMod(BinaryOp):
	def apply(lhs,rhs):
		a = _force_int(lhs)
		b = _suppress_zero(_force_int(rhs))
		return tir.floormod(a,b)

	def apply_np(lhs,rhs):
		return lambda : FloorMod._floor_mod(int(lhs()),_suppress_zero(int(rhs())))

	def _floor_mod(a,b):
		#avoid unnecssary evaluations with helper function
		return a - (floor(a / b) * b)

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
		return lambda : TruncMod._trunc_mod(int(lhs()), _suppress_zero(int(rhs())))

	def _trunc_mod(a,b):
		return a - (trunc(a / b) * b)

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
		return  tir.abs(_force_int(lhs)) >> tir.min(30,tir.abs(_force_int(rhs)))

	def apply_np(lhs,rhs):
		return lambda : abs(int(lhs())) >> min(int(30),abs(int(rhs())))

class ShiftLeft(BinaryOp):
	def apply(lhs,rhs):
		return  tir.abs(_force_int(lhs)) << tir.min(30,tir.abs(_force_int(rhs)))

	def apply_np(lhs,rhs):
		return lambda : abs(int(lhs())) << min(int(30),abs(int(rhs())))

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
		return tir.power(_suppress_zero(_force_float(lhs)),_force_float(rhs))

	def apply_np(lhs,rhs):
		return lambda : _suppress_zero(float(lhs())) ** float(rhs())

class TrinaryOp(object):
	nargs = 3

	def apply(arg_one,arg_two,arg_three):
		raise NotImplementedError

	def apply_np(arg_one,arg_two,arg_three):
		raise NotImplementedError


class Select(TrinaryOp):

	def apply(condition,true_expr,false_expr):
		return tir.Select(tir.Cast('bool',_clamp_tvm(condition,0,1)),_force_int(true_expr),_force_int(false_expr))

	def apply_np(condition,true_expr,false_expr):
		return lambda : int(true_expr()) if bool(_clamp_np(condition(),0,1)) else int(false_expr())

class IfThenElse(TrinaryOp):
	def apply(condition,true_expr,false_expr):
		return tir.if_then_else(tir.Cast('bool',_clamp_tvm(condition,0,1)),true_expr,false_expr)

	def apply_np(condition,true_expr,false_expr):
		return lambda : true_expr() if bool(_clamp_np(condition(),0,1)) else false_expr()

class KNaryOp(object):
	nargs = -1

	def apply(*args):
		raise NotImplementedError

	def apply_np(*args):
		raise NotImplementedError

class Any(KNaryOp):
	def apply(*args):
		bool_args = list(map(lambda x : tir.Cast('bool',_clamp_tvm(x,0,1)),args))
		return tir.any(*bool_args)

	def apply_np(*args):
		return lambda : Any._any(*args)

	def _any(*args):
		for arg in args:
			if bool(_clamp_np(arg(),0,1)):
				return True
		return False

class All(KNaryOp):
	def apply(*args):
		bool_args = list(map(lambda x : tir.Cast('bool',_clamp_tvm(x,0,1)),args))
		return tir.all(*bool_args)

	def apply_np(*args):
		return lambda : All._all(*args)

	def _all(*args):
		for arg in args:
			if not bool(_clamp_np(arg(),0,1)):
				return False
		return True

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

	last_random = None

	def apply(val = None):
		if val:
			return val
		IntLit.last_random = random.randint(-100,100)
		return tir.IntImm('int32',IntLit.last_random)

	def apply_np(val = None):
		if not val:
			val = IntLit.last_random
		return lambda : int(val)

class BoolLit(Literal):

	last_random = None
	def apply(val = None):
		if val:
			return val
		BoolLit.last_random = random.random() >= 0.5
		return tir.IntImm('bool',BoolLit.last_random)

	def apply_np(val = None):
		if not val:
			val = BoolLit.last_random
		return lambda : bool(val)

class NewVar(Literal):

	last_random = None

	_var_selection = ProbabilisticSelection([
			(1,lambda : te.var(name="f"+str(len(SymbolTable.variables)),dtype='float32')),
			(1,lambda : te.var(name="i"+str(len(SymbolTable.variables)),dtype='int32'))
		])

	def apply(name = None):
		if name:
			var = te.var(name=name,dtype='int32' if name.startswith('i') else 'float32')
			SymbolTable.variables.append(var)
			return SymbolTable.variables[-1]
		new_var = NewVar._var_selection.select()()
		SymbolTable.variables.append(new_var)
		NewVar.last_random = new_var.name
		return new_var

	def apply_np(name = None):
		if not name:
			name = NewVar.last_random
		return lambda : SymbolTable.binds[name]


class ExistingVar(Literal):

	last_random = None

	def apply(name = None):
		if name:
			for var in SymbolTable.variables:
				if(var.name == name):
					return var
			return NewVar.apply(name)

		if(len(SymbolTable.variables) == 0):
			new_var = NewVar.apply()
			ExistingVar.last_random = new_var.name
			return new_var
		rand_var =  SymbolTable.variables[random.randint(0,len(SymbolTable.variables) - 1)]
		ExistingVar.last_random = rand_var.name
		return rand_var

	def apply_np(name = None):
		if not name:
			name = ExistingVar.last_random
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

	SymbolTable.variables = ['i0','f0']
	SymbolTable.binds = {"i0":-1,'f0':-1.0}
	print(c_tvm)
	print(c_np())