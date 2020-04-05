# 3/27/2020
# Bug descrption: Tvm creates consts without enforcing type bound conditions.
# PR: https://github.com/apache/incubator-tvm/pull/5156

import tvm
from tvm import tir,te

def test_const_bounds_int():
    for signed_bits in [8,16,32]:
        dtype = 'int' + str(signed_bits)
        int_max = (1 << (signed_bits - 1))- 1
        int_min = - (1 << (signed_bits - 1))
        try:
            tvm.tir.const(int_min - 1,dtype)
            assert False
        except tvm.TVMError:
            pass

        try:
            tvm.tir.const(int_max + 1,dtype)
            assert False
        except tvm.TVMError:
            pass

        assert tvm.tir.const(int_min,dtype).value == int_min
        assert tvm.tir.const(int_max, dtype).value == int_max
        assert tvm.tir.const(1, dtype).value == 1
        assert tvm.tir.const(0, dtype).value == 0


def test_const_bounds_uint():
    for unsigned_bits in [1,8,16,32]:
        dtype = 'uint' + str(unsigned_bits)
        uint_max = (1 << unsigned_bits) - 1
        try:
            tvm.tir.const(dtype, uint_max + 1)
            assert False
        except tvm.TVMError:
            pass

        assert tvm.tir.const(uint_max,dtype).value == uint_max
        assert tvm.tir.const(1, dtype).value == 1
        assert tvm.tir.const(0, dtype).value == 0

    large_uint_imm = tvm.tir.const((1 << 64) - 1, 'uint64')
    assert large_uint_imm.name == "tvm_large_uint_imm"
    low = large_uint_imm.args[0].value
    high = large_uint_imm.args[1].value
    assert low  | (high << 32) == (1 << 64) - 1

if __name__ == "__main__":
	test_const_bounds_int()
	test_const_bounds_uint()
                