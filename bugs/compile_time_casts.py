# 3/27/2020
# Bug descrption: Tvm simplifes casts illegally
# PR: https://github.com/apache/incubator-tvm/pull/5156

import tvm
from tvm import tir,te

def test_illegal_cast():
    analyzer = tvm.arith.Analyzer()
    try:
        analyzer.rewrite_simplify(1 * tir.Cast('bool',77))
        assert False
    except tvm.TVMError:
        pass
    try:
        analyzer.rewrite_simplify(1 * tir.Cast('int8',171))
        assert False
    except tvm.TVMError:
        pass
    try:
        analyzer.rewrite_simplify(1 * tir.Cast('int32',2 ** 32))
        assert False
    except tvm.TVMError:
        pass

if __name__ == "__main__":
    test_illegal_cast()
