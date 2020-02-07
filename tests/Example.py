import tvm
a = tvm.var(dtype='float32')
c = tvm.compute((a,a),lambda i, j : a + a + 2) 
d = tvm.compute((a/2,a/2), lambda i,j : c[i,j])
s = tvm.create_schedule(d.op)

f = tvm.build(s,[c,d])


