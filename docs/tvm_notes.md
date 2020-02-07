## Frontend

Entry point is `tvm.build(inputs=[tvm.Schedule],args=[Buffer/Tensor/Var], target=tvm.target.Target/str, target_host=tvm.target.Target/str,name=str,binds=dict{??})`
	-can first arg be a list?
	-can second arg include size_var, const, etc.

`tvm.Schedule` is created with `tvm.create_schedule([tvm.tensor.ComputeOp])`
`tvm.tensor.ComputeOp` is `op` field of `tvm.tensor.Tensor`
`tvm.tensor.Tensor` is created by `tvm.compute(shape=[tvm.expr],fcompute= indices -> value,name=str,tag=str,attrs=dict{??})`
`indices->value` is a function of the form `lambda a_0 .... a_i : RHS op LHS`

# Compute function
fcompute is created by the function `tvm.convert` from a lambda function or in the case of it being an intrinsic call is handled by `_api_internal._TensorComputeOp`

The interesting case is the lambda function as it is formed by a conglomeration of tvm objects and python instrinics. For example a valid fcompute is:
```
a = tvm.var(dtype='float32')
tvm.compute((a,a),lambda i, j : a + a + 2) 
```
To extract the code of a lambda function in python the following code is used:
```
a = lambda b : b + 1
a.__code__
```

This behaviour is allowed by the function `convert_to_object` found in `tvm/_ffi/object_generic.py`. It leverages the python module `numbers` to constify ints. It also supports bools.

The next important line is the following found in `compute` in `tvm/api.py` where `fcompute` is the lambda function:
`body = fcompute(*[v.var for v in dim_var])`

The way the AST is constructed is using operator overriding in `tvm.expr.Expr`. For example consider the following code:
```
a = tvm.var(dtype='float32')
e = a + 1
print(type(e))
# prints tvm.expr.Add
print(type(e.b))
# prints tvm.expr.IntImm
print(type(e.a))
# prints tvm.expr.Var
```

# Dependant tensors

TVM allows previously computed tensors to be used in subsequent computations as follows:

```
a = tvm.var(dtype='float32')
c = tvm.compute((a,a),lambda i, j : a + a + 2) 
d = tvm.compute((a/2,a/2), lambda i,j : c[i,j])
```

Interestingly the manner in which this is handled is as follows:
```
a = tvm.var(dtype='float32')
c = tvm.compute((a,a),lambda i, j : a + a + 2) 
print(type(c[i,j])) 
#prints tvm.tensor.TensorSlice
print((c[i,j] + 1).a)
#prints compute(i,j)
print(type((c[i,j] + 1).a))
#prints tvm.expr.Call
```

As can be there are function calls that also contain special meaning in TVM.

# Generation strategy

As seen in the **Compute function** section exprs can be developed through operator overriding between `tvm.expr.Expr`s. Thus discovering all expr nodes and operators gives the possibility of generated all possible tvm Exprs. The place to look for this is `tvm/expr.py` which contains subclasses of tvm.expr.Expr as well as the definitions for operation overriding. 

Given the capability to generate arbitrary expressions it remains to generate arbitrary sequences of computations.

Thus tvm front-end is really object oriented programming and operator overriding in python. 