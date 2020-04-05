## Bugs

In this directory are testcases for bugs found by TVMFuzz. They are designed such that they should crash or raise an assertion error to indicate the bug unless the bug has been patched. In this sense they can be considered regression tests.

The only exception is the directory `floormod` which was a special case where the operators `floormod` and `floordiv` produce a massive amount of IR and effectively crash TVM by taking potentially hours to build a program.

## Bug Types

| Class  | Type | Location |
| ------------- | ------------- | ------------- |
| floormod  | crash | middleend |
| backend_float_bitwise.py  | crash | backend |
| bitshift_bounds.py  | wrong code generation | backend |
| compile_time_casts.py  | wrong code generation | middleend |
| constant_fold_div_by_zero.py  | crash | frontend |
| constant_fold_underflow.py  | wrong code generation | frontend|
| crashing_ops.py  | crash | frontend |
| float_bitshift.py  | wrong code generation | backend |
| out_of_bounds_consts.py  | wrong code generation | frontend |
| rounding_ints.py  | crash | middleend |
