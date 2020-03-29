from tvm.runtime import DataType, TypeCode
from tvm import te,tir
import tvm

#from tvm/python/tvm/tir/expr.py
def dtype_is_int(value):
    if isinstance(value, int):
        return True
    return (isinstance(value, tir.expr.ExprOp) and
            DataType(value.dtype).type_code == TypeCode.INT)

def dtype_is_uint(value):
    return (isinstance(value, tir.expr.ExprOp) and
            DataType(value.dtype).type_code == TypeCode.UINT)

#from tvm/python/tvm/tir/expr.py
def dtype_is_float(value):
    if isinstance(value, float):
        return True
    return (isinstance(value, tir.expr.ExprOp) and
            DataType(value.dtype).type_code == TypeCode.FLOAT)

def get_literal_value(e):
    """ Get literal value from expr

    Parameter
    ---------
    e : object
        Expression

    Returns
    -------
    lit : int,float,bool,None
        Returns the value of the literal
        or None if expr does not have literal value
    """

    if(isinstance(e,(int,float))):
        return e
    if(isinstance(e,((tir.expr.IntImm,tir.expr.FloatImm)))):
        return e.value
    return None

