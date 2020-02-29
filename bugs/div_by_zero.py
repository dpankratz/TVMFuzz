# 2/29/2020
# Bug descrption: Python frontend crashes with no error message when trying to mod by zero
# PR: 


from tvm import TVMError,tir,te
import traceback

def test_div_by_zero():
	a = te.var(name='a',dtype='int32')
	b = te.var(name='b',dtype='int32')
	
	zero = tir.const(0)
	two = tir.const(2)
	fzero = tir.const(0.0)
	ftwo = tir.const(2.0)
	for int_bin_op in [lambda a,b: a % b, lambda a,b: tir.floordiv(a,b), lambda a,b: tir.truncdiv(a,b),
					lambda a,b: tir.floormod(a,b), lambda a,b: tir.truncmod(a,b) ]:
		try:
			print(str(int_bin_op(a,zero)) + " a by zero")
			int_bin_op(a,zero)
		except TVMError:
			pass
			

		try:	
			print(str(int_bin_op(two,zero)) + "two by zero")
			int_bin_op(two,zero)
		except TVMError:
			pass
	
	for float_bin_op in [lambda a,b: tir.div(a,b), lambda a,b: tir.truncmod(a,b)]:
		try:
			print(str(float_bin_op(a,fzero)) + " a by fzero")
			float_bin_op(a,fzero)
		except TVMError:
			pass


		try:
			print(str(float_bin_op(ftwo,fzero)) + "two by fzero")
			float_bin_op(ftwo,fzero)
		except TVMError:
			pass
		
if __name__ == "__main__":
	test_div_by_zero()