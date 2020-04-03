## Improving TIR for FloorMod and FloorDiv

Currently [FloorMod](https://github.com/apache/incubator-tvm/blob/master/src/tir/transforms/lower_intrin.cc#L164) and [FloorDiv](https://github.com/apache/incubator-tvm/blob/master/src/tir/transforms/lower_intrin.cc#L114) produce a massive amount of IR for the case where the signs of the operators are unknown.

This was causing TVMFuzz to appear to hang when these operators were near the root of the tree due to how many copies of the LHS and RHS were created and operated on. 

Rather than expressing them using a runime expression it's possible to instead use the `floor` instrinsic in TIR which requires no control flow and many fewer copies of the operands.

## Organization

The files in this directory allow the changes to these operators to be tested for build-time, runtime, and correctness. Further the directories `data` and `figures` show the results already accumulated for `floormod`.

## FloorMod Code 

```
if (dtype.bits() <= 32){
  /* NOTE:
  This must be restricted to int32 or less since floats can losslessly represent integers
  only if the number of bits in the mantissa exceeds the number of bits in the integer.
  Therefore a double (53 bit mantissa) for int32, float (24 bit mantissa) for int16, etc.
  Since TVM is unaware of a float128 type, int64 is not supported. 
  */
  
  // a - floor(a / b) * b
  auto fdtype = DataType::Float(dtype.bits() * 2,dtype.lanes());
  auto div = tir::CastNode::make(fdtype,op->a)
        / tir::CastNode::make(fdtype,op->b);
  auto f = tvm::floor(div);
  auto floor_lowered = tir::CastNode::make(dtype,VisitExpr_(f.as<CallNode>()));

  return op->a - (floor_lowered * op->b);
}
```

## FloorDiv Code

```
if (dtype.bits() <= 32){
	/* NOTE:
	This must be restricted to int32 or less since floats can losslessly represent integers
	only if the number of bits in the mantissa exceeds the number of bits in the integer.
	Therefore a double (53 bit mantissa) for int32, float (24 bit mantissa) for int16, etc.
	Since TVM is unaware of a float128 type, int64 is not supported. 
	*/

	// floor(a / b)
	auto fdtype = DataType::Float(dtype.bits() * 2,dtype.lanes());
	auto div = tir::CastNode::make(fdtype,op->a)
	      / tir::CastNode::make(fdtype,op->b);
	auto f = tvm::floor(div);
	return tir::CastNode::make(dtype,VisitExpr_(f.as<CallNode>()));
}
```