
import sys,tvm,random

from datetime import datetime
from tvm import te,tir
import numpy as np

import matplotlib as plt

SHAPE = (10000,10000)

tvm_vars = []

def create_nested_expr(level, dtype, op = lambda lhs,rhs : lhs + rhs):
	if level == 0:
		tvm_vars.append(te.var(name="i"+str(len(tvm_vars)), dtype = dtype))
		return tvm_vars[-1]
	return op(create_nested_expr(level - 1,dtype,op),-create_nested_expr(level - 1,dtype,op))

def create_floormod_expr(level,dtype):
	return create_nested_expr(level, dtype, op = tir.floormod)

def create_floordiv_expr(level,dtype):
	return create_nested_expr(level, dtype, op = tr.floordiv)

def test_build_time(tvm_expr,tvm_vars,repetitions = 1):
	assert repetitions > 0
	f = None
	times = []
	for i in range(repetitions):
		c = te.compute(SHAPE,lambda i,j: tvm_expr)
		s = te.create_schedule([c.op])
		f = tvm.build(s,tvm_vars + [c])
	return f, times

def test_run_time(tvm_func, num_vars, repetitions = 10):
	assert repetitions > 0
	c_np = np.zeros(SHAPE,dtype='int32')
	c_tvm = tvm.nd.array(c_np)
	times = []
	for i in range(repetitions):
		binds = []
		for i in range(num_vars):
			binds.append( int(1.5 ** i) + 3)
		start_time = datetime.now()
		f(*binds,c_tvm)
		times.append((datetime.now() - start_time).microseconds)
	return times

if __name__ == "__main__":
	time_avgs = []
	time_stds = []

	num_levels = 6
	for level in range(1,num_levels):
		print("#" * 20)
		print("Level={0}".format(level))
		print("#" * 20)

		tvm_vars.clear()
		expr = create_floormod_expr(level, 'int32')
		print("Expr={0}".format(expr))
		f,_ = test_build_time(expr, tvm_vars)
		
		times = test_run_time(f, len(tvm_vars))
		time_avgs.append(np.average(times))
		time_stds.append(np.std(times))
		print("times={0}".format(times))

	f = open("data/run_stats.txt","a")
	for i in [time_avgs,time_stds]:
		f.write(str(i) + "\n")
	f.close()