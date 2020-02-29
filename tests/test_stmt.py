import tvm
import numpy as np
import sys
from tvm import te,tir


shape = (1,	)
a = te.var(name="a",dtype="int32")
b = te.var(name="b",dtype="int32")
c = te.compute(shape,lambda i: a % b)
s = te.create_schedule([c.op])
f = tvm.build(s,[a,b,c])
c_tvm= tvm.nd.array(np.zeros(shape,dtype='int32'))
f(2,0,c_tvm)
print(c_tvm)
