# 2/28/2020
# Bug descrption: Overflow and underflow in TVM constant folding are incorrect due to IntImm always using int64_t 
# PR: Decided not to submit since this is a very deeply entrenched issue in TVM
import tvm
from tvm import tir,te
import numpy as np

def test_underflow():
	int_min = tir.const(-(1 << 31),'int32')
	constant_fold_res = (int_min - 1).value
	int_min_np = np.int32(-(1 << 31))
	res_np = int_min_np - np.int32(1)

	print(constant_fold_res,res_np)

	assert constant_fold_res == res_np


if (__name__ == "__main__"):
	test_underflow()
