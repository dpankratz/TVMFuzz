# 2/9/2020
# Bug descrption: Crash when trying to perform bitwise operation involving int on LHS and tvm ExprOP on RHS.
# PR: https://github.com/apache/incubator-tvm/pull/4852

from tvm import te
a = te.var()
#reported
b = 10 ^ a #crashes
b = 10 | a #crashes
b = 10 & a #crashes
b = 10 >> a
b = 10 % a
b = 10 << a #crashes

