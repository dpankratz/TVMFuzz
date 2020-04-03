## Ground-truth

To generate a ground truth expr alongside a TIR expression it is sufficent to implement equivalent operators to the TIR operators and then compose them in the same order. However, this strategy has the limitation that it very tricky to re-execute an arbitrary sub expression which is useful in narrowing down where the bug occurred.

To remedy this instead of directly creating TIR and Python ground-truth expression, a tree structure is created that has the capability of emitting the expression at arbitrary points in the tree. This can be found in `generation_node.py`.

## Equivalent Python Expressions

This part can be very easy in the case of operators like `+` or more tricky in cases of operators like `floormod` or `%` which have no Numpy or Python equivalent. Furthermore it is important to use the same datatypes that TVM ends up using when evaluating the expression such as (int32, float32, bool, etc). Due to Python using arbitrarily sized integers, TVMFuzz uses numpy datatypes (`Numpy.int32`, `Numpy.float32`, `Numpy.bool`, etc.)  to match the behaviour of TVM more closely. 

The other important feature when generating a ground-truth program is the capability of specifying variable bindings and then evaluating the expression using those bindings perhaps many times. Therefore instead of creating a Python expression like `a + 1` which would take the value of `a` immediately and return a literal, TVMFuzz emits a tree of statements like `lambda : a + 1`. Then the expression can have its variables redefined in a manner similar to this example:
```
root = lambda : a + 1
a = 2
root() #3
a = 4
root() #5 
```

More precisely TVMFuzz would produce a tree of statements like `lambda : lhs() + rhs()` since each subexpression is itself a `lambda` statement. So in the above example `lhs()` would return a variable access to `a` and `rhs()` would return `1`.

## Comparing to TVM 

Assuming a Generation Node tree has been constructed then TVM and the ground-truth can compared as follows:
```
tvm_expr = root_node.emit_tvm()
np_expr = root_node.emit_np()
tvm_result = util.evaluate_tvm_expr(tvm_expr)
np_result = np_expr()
print("Are equal = " + str(np_result == tvm_result))
```
This simplifies the situation slightly (how to handle crashes? comparing `nan` and `inf`?) but shows the general idea. 

## Defining an Operator

To define a new operator in TVMFuzz it is as simple as creating a representation of the TIR operator and the python equivalent. For example the operator `+` is defined as follows in TVMFuzz:

```
class Add(BinaryOp):
	def apply(lhs,rhs):
		return lhs + rhs

	def apply_np(lhs,rhs):
		return lambda : lhs() + rhs()
```

For operators with more restrictions then it may be necessary to cajole the inputs to be valid. See [Stratey Generation](https://github.com/dpankratz/TVMFuzz/blob/master/docs/expr_generation.md) for more discussion. 