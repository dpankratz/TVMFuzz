import __init__
import test_bed,tvm
from SymbolTable import SymbolTable
from Expression import *
from datetime import datetime
from tvm import te,tir
import numpy as np


var1 = te.var()
var2 = te.var()
bootstrap = var1 + var2

limit = 6
	
shape = (10000,10000)
c_tvm = tvm.nd.array(np.zeros(shape=shape,dtype='int32'))
root = bootstrap
i = 0
while i < limit:
	root = var2 + tir.floormod(var1,root)
	#print(root)
	
	
	c = te.compute(shape,lambda i,j: root)
	s = te.create_schedule([c.op])
	f_lowered = tvm.lower(s,[var1,var2,c],simple_mode=True)
	print("#" * 20)
	print(f_lowered)
	print("#" * 20)
	#print(type(f_lowered.body.value.a))
	build_start = datetime.now()
	f = tvm.build(s,[var1,var2,c])
	
	#print(i,test_bed.evaluate_tvm_expr(root,SymbolTable.variables,SymbolTable.binds))
	print("build time={0}".format(datetime.now() - build_start))

	run_start = datetime.now()
	f(5,-3,c_tvm)
	print("run time={0}".format(datetime.now() - run_start))

	print(c_tvm.asnumpy()[0][0])

	
	i +=  1
