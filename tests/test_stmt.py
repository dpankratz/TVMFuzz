import __init__
import tvm
import numpy as np
import sys,traceback,math
from Expression import *
from tvm import te,tir
from test_bed import evaluate_tvm_expr

floormod = lambda a,b : a - math.floor(a / b) * b

a = te.var(name='a', dtype='int32')
b = te.var(name='b', dtype='int32')
shape = (1,)
c = te.compute(shape,lambda i: tir.floormod(a,b))
s = te.create_schedule([c.op])
f = tvm.build(s,[a,b,c])
c_tvm = tvm.nd.array(np.zeros(shape,dtype='int32'))
f(16777217 ,2,c_tvm) 
print(c_tvm.asnumpy()[0] == floormod(16777217,2))
