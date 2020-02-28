from tvm.runtime import DataType, TypeCode

import tvm

#from tvm/python/tvm/tir/expr.py
def dtype_is_int(value):
    if isinstance(value, int):
        return True
    return (isinstance(value, tvm.expr.ExprOp) and
            DataType(value.dtype).type_code == TypeCode.INT)

def dtype_is_uint(value):
	return (isinstance(value, tvm.expr.ExprOp) and
            DataType(value.dtype).type_code == TypeCode.UINT)

#from tvm/python/tvm/tir/expr.py
def dtype_is_float(value):
    if isinstance(value, float):
        return True
    return (isinstance(value, tvm.expr.ExprOp) and
            DataType(value.dtype).type_code == TypeCode.FLOAT)

def get_literal_value(e):
	if(isinstance(e,(int,float))):
		return e
	if(isinstance(e,((tvm.expr.IntImm,tvm.expr.FloatImm)))):
		return e.value
	return None