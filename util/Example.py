import tvm
a = tvm.var(dtype='float32')
tvm.compute((a,a),lambda i, j : a + a + 2) 