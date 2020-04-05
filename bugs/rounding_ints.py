# 2/29/2020
# Bug descrption: LLVM backend crashes when trying to round ints
# PR: https://github.com/apache/incubator-tvm/pull/5026

import tvm

def test_round_intrinsics_on_int():
    i = tvm.te.var("i", 'int32')
    for op in [tvm.tir.round, tvm.tir.trunc, tvm.tir.ceil,
                            tvm.tir.floor, tvm.tir.nearbyint]:
        assert op(tvm.tir.const(10,'int32')).value == 10
        assert op(tvm.tir.const(True,'bool')).value == True
        assert op(i).same_as(i)

if __name__ == "__main__":
	test_round_intrinsics_on_int()