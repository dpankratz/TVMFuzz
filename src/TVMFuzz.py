from ExprGeneration import generate_tvm_and_np_expr
from SymbolTable import SymbolTable
import random

tvm_expr,np_expr = generate_tvm_and_np_expr()

print("tvm_expr={0}".format(tvm_expr))

binds = {}
for var in SymbolTable.variables:

	if(var.dtype == 'int32'):
		SymbolTable.binds[var.name] = random.randint(-10,10)
	elif(var.dtype == 'float32'):
		SymbolTable.binds[var.name] = random.random() * 20 - 10
	print(var.name + " = " + str(SymbolTable.binds[var.name]))
	#numpy binds

	#TODO: tvm test bed for exprs and stmts

print(np_expr())
