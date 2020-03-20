# 3/20/2020
# Bug descrption: Bitshifting by an out of bounds value emits no code
# PR: https://github.com/apache/incubator-tvm/pull/5115

import tvm

from tvm import te,tir

x = te.var()

for test in [lambda lhs, rhs : lhs << rhs,
                lambda lhs, rhs : lhs >> rhs]:
    #negative case
    for testcase in [(x,-1), (x,32)]:
        try:
            test(*testcase)
            assert False
        except tvm.TVMError:
            pass

    #positive case
    for testcase in [(x,0), (x,16), (x,31)]:
        test(*testcase)