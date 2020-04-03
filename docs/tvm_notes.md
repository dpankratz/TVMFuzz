## TIR

This fuzzer is primarily aimed at testing expressions in TIR (TVM Intermediate Representation). TIR is concerned with the definition and scheduling of operators inspired by the Halide design. Therefore it is not uncommon to be writing expressions over scalars which are later expanded to compute a tensor 'pixelwise'. Therefore bugs involving scalar expressions are directly relevant to expression of tensor computations. 

## Frontend

The most basic usage of expressions in TVM is in a statement like this `tvm.compute(shape, lambda i : i + 1)` which computes a vector with elements `(1,...,n)`. The expression provided to the compute function is created as the overall statement is interpreted by the Python environment. In this case `i + 1` is combining a `te.Var` with a Python `int` using the `Expr.__add__` Python operator overriding feature.

Interestingly some amount of simplification and constant folding occurs at this stage such as `tir.const(13) << 4` becomes a `tir.IntImm` with value 208. 

Therefore to test the TVM frontend of creating and combining expressions it is not necessary to output a complete program and rather is possible to quickly create standalone expressions which are *compiled* by simply being interpreted within Python.

A number of front end bugs were discoved by simply combining operators such as `<<, >>, &, %` in an unexpected order with python literals. For example `tvm.const(1) % 2` was acceptable but `2 % tvm.const(1)` produced a crash due to `__rmod__` not being defined.

## Middle-end

TIR implements a number of compiler passes are run when the commands `tvm.lower` or `tvm.build` are invoked. In the context of this tool, and disregarding crashes, this is tested by comparing the output of generated binary to the ground-truth program.

A middle end bug discovered is that `1 * tir.Cast('bool', 77)` gets changed to `77` by the `RewriteSimplify` pass. This is due to the pass creating a constant with dtype `bool` without ensuring the value provided to the constant is within sane ranges. Subsequently the `1 * ((bool)77)` promotes the `77` back to an `int32` so the end result is `1 * 77 -> 77`.

## Backend 

Similar to middle-end the LLVM backend can be tested by taking the function produced by `tvm.build`, running it for a given input, and comparing the result to the ground-truth program. 

A bug discovered in the LLVM backend is that expressions like `2 << 32` with dtype `int32` becomes `undef`. This results in other expressions also becoming `undef` because of rules like `undef + 5 -> undef`. The end result is that all the code from the `tvm.compute` statement is removed in the LLVM backend and the function from `tvm.build` does nothing. 