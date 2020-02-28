import tvm
import numpy as np
import sys

import tvm
import numpy as np
import sys


shape = (1,1)
a = tvm.const(dtype='int32',value=10)
c = tvm.compute(shape,lambda i,j: tvm.const(True) + tvm.const(True))
s = tvm.create_schedule([c.op])
f = tvm.build(s,[c])
c_tvm= tvm.nd.array(np.zeros(shape,dtype='bool'))
f(c_tvm)
print(c_tvm)
