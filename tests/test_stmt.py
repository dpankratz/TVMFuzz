import __init__
import tvm
import numpy as np
import sys,traceback,math
from expression import *
from tvm import te,tir
from test_bed import evaluate_tvm_expr

a =te.var('a','bool')
shape = (1,)
c = te.compute(shape,lambda i: a + a)
s = te.create_schedule([c.op])
f = tvm.build(s,[a,c])
c_tvm = tvm.nd.array(np.zeros(shape,dtype='bool'))
f(1,c_tvm) 

print(c_tvm.asnumpy()[0])
