
import sys,tvm,random

from datetime import datetime
from tvm import te,tir
import numpy as np

import matplotlib as plot

SHAPE = (10000,10000)

tvm_vars = []

def create_floormod_expr(level):
	global tvm_vars
	if level == 0:
		tvm_vars.append(te.var(name="i"+str(len(tvm_vars))))
		return tvm_vars[-1]
	return tir.FloorDiv(create_floormod_expr(level - 1),create_floormod_expr(level - 1))

def test_build_time(tvm_expr,tvm_vars,repetitions = 5):
	assert repetitions > 0
	f = None
	times = []
	for i in range(repetitions):
		c = te.compute(SHAPE,lambda i,j: tvm_expr)
		s = te.create_schedule([c.op])
		start_time = datetime.now()
		f = tvm.build(s,tvm_vars + [c])
		times.append((datetime.now() - start_time).microseconds)
	return f, times

def test_run_time(tvm_func, num_vars, repetitions = 10):
	assert repetitions > 0
	c_np = np.zeros(SHAPE,dtype='int32')
	c_tvm = tvm.nd.array(c_np)
	times = []
	for i in range(repetitions):
		binds = []
		for i in range(num_vars):
			binds.append(i + 1)
		start_time = datetime.now()
		f(*binds,c_tvm)
		times.append((datetime.now() - start_time).microseconds)
	return c_tvm, times

if __name__ == "__main__":
	build_time_avgs = []
	run_times_avgs = []

	num_levels = 9
	for level in range(num_levels):
		print("#" * 20)
		print("Level={0}".format(level))
		print("#" * 20)
		tvm_vars.clear()
		expr = create_floormod_expr(level)
		print("Expr={0}".format(expr))
		f,times = test_build_time(expr, tvm_vars)
		avg = np.average(times)
		print("Build times={0}, arith mean={1}".format(times,avg))
		build_time_avgs.append(avg)
		continue
		output, times = test_run_time(f,len(tvm_vars))
		avg = np.average(times)
		print("Run times={0}, arith mean={1}".format(times,avg))
		run_times_avgs.append(avg)
		print("Output={0}".format(output))

	print("Overall build_time_avgs={0} and run_times_avgs={1}".format(build_time_avgs,run_times_avgs))