
import tvm
from datetime import datetime
from tvm import te,tir
from math import fmod
import numpy as np
from topi import floor_mod


def floormod_intrin(a,b):
	 return a - (tir.Cast(a.dtype,tir.floor(tir.Cast('float64',a) / tir.Cast('float64',b))) * b)

DIM = 1000
HDIM = 500
shape = (DIM,DIM)
c_tvm = tvm.nd.array(np.zeros(shape=shape,dtype='int32'))
c_np = np.zeros(shape)
	
c = te.compute(shape,lambda i,j: tir.floormod(HDIM - i,j + 1) )
d = te.compute(shape,lambda i,j: tir.floormod(HDIM - i,-(j + 1)))
s = te.create_schedule([c.op])
s2 = te.create_schedule([d.op])
f = tvm.build(s,[c])
f2 = tvm.build(s,[d])
f(c_tvm)
out = c_tvm.asnumpy()
for i in range(DIM):
	for j in range(DIM):
		res = out[i][j]
		res2 = floor_mod(DIM/2 - i, j + 1).value
		if res != res2:
			print(i,j,res,res2)
			assert False

print("Done half")

f2(c_tvm)

out = c_tvm.asnumpy()
for i in range(DIM):
	for j in range(DIM):
		res = out[i][j]
		res2 = floor_mod(DIM/2 - i, j + 1).value
		if res != res2:
			print(i,j,res,res2)
			assert False


