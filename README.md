## Usage

To invoke the fuzzer simply run `python tvmfuzz.py`. This will generate a random `GenerationNode` tree which is capable of producing TVM and ground-truth programs.
A sample output of the fuzzer looks like this:

```
timestamp =1584920517.211394

tree=BitwiseAnd.apply(IntLit.apply(-55),EQ.apply(FloorDiv.apply(BoolLit.apply(True),BoolLit.apply(True)),EQ.apply(ExistingVar.apply('f0'),IntLit.apply(-73))))

tvm expr=bitwise_and(-55, int32((1 == int32((f0 == -73f)))))

f0 = 2.7451091482651506
np result=0
tvm result=0
equal=True
```

**values**
- The `timestamp` value is used to re-run the fuzzer on a certain seed. For example `python tvmfuzz.py 1584920517.211394`.
- The `tree` value shows the generated tree. This is roundtripable in the python interpreter. For example in the above example you can reproduce the tvm_expr as follows:
	```
	from expression import *
	print(BitwiseAnd.apply(IntLit.apply(-55),EQ.apply(FloorDiv.apply(BoolLit.apply(True),BoolLit.apply(True)),EQ.apply(ExistingVar.apply('f0'),IntLit.apply(-73)))))
	#prints bitwise_and(-55, int32((1 == int32((f0 == -73f)))))
	```
- The `f0` shows the value of the binding used to run the program.
- The `np result` and `tvm result` values are the respective results of running the ground-truth program and tvm program.
- The `equal` field shows whether the tvm result matched the ground-truth.

The goal of the output is to give sufficient information to continue investing a bug. Assuming there was a mismatch additional information is generated:

```
timestamp =1584920527.353658

tree=Neg.apply(Max.apply(Max.apply(Add.apply(NE.apply(FloorMod.apply(FloorMod.apply(FloorDiv.apply(NE.apply(NewVar.apply('f0'),ExistingVar.apply('f0')),Sub.apply(LE.apply(FloorDiv.apply(IntLit.apply(-23),IntLit.apply(-96)),IntLit.apply(-87)),Select.apply(NE.apply(ExistingVar.apply('f0'),ExistingVar.apply('f0')),NewVar.apply('i1'),BitwiseNeg.apply(ExistingVar.apply('i1'))))),BoolLit.apply(False)),BoolLit.apply(False)),Neg.apply(IntLit.apply(53))),BoolLit.apply(True)),ShiftLeft.apply(BitwiseOr.apply(IntLit.apply(76),Abs.apply(BoolLit.apply(True))),IntLit.apply(94))),LT.apply(Select.apply(ShiftRight.apply(GT.apply(Neg.apply(IntLit.apply(-66)),ExistingVar.apply('f0')),ShiftLeft.apply(EQ.apply(ExistingVar.apply('i1'),ExistingVar.apply('f0')),IntLit.apply(-72))),ExistingVar.apply('f0'),FloorDiv.apply(BoolLit.apply(False),GE.apply(BoolLit.apply(True),IntLit.apply(70)))),BitwiseAnd.apply(ExistingVar.apply('i1'),Abs.apply(ExistingVar.apply('i1'))))))

tvm expr=(max(max((int32((floormod(floormod(floordiv(int32((f0 != f0)), select((select(bool((select(bool((f0 != f0)), i1, bitwise_not(i1)) > int32((bool)0))), (select(bool((f0 != f0)), i1, bitwise_not(i1)) - int32((bool)0)), (int32((bool)0) - select(bool((f0 != f0)), i1, bitwise_not(i1)))) == 0), 4, select(bool((select(bool((f0 != f0)), i1, bitwise_not(i1)) > int32((bool)0))), (select(bool((f0 != f0)), i1, bitwise_not(i1)) - int32((bool)0)), (int32((bool)0) - select(bool((f0 != f0)), i1, bitwise_not(i1)))))), 4), 4) != -53)) + 1), 82678120448), int32((select(bool(shift_right(select((int32((f0 < 66f)) >= 0), int32((f0 < 66f)), (0 - int32((f0 < 66f)))), select((shift_left(select((int32((float32(i1) == f0)) >= 0), int32((float32(i1) == f0)), (0 - int32((float32(i1) == f0)))), 72) >= 0), shift_left(select((int32((float32(i1) == f0)) >= 0), int32((float32(i1) == f0)), (0 - int32((float32(i1) == f0)))), 72), (0 - shift_left(select((int32((float32(i1) == f0)) >= 0), int32((float32(i1) == f0)), (0 - int32((float32(i1) == f0)))), 72))))), int32(f0), 0) < bitwise_and(i1, select((i1 >= 0), i1, (0 - i1))))))*-1)

f0 = -5.275919855729221
i1 = 6
np result=-1525142128399588498675721043968
tvm result=-1073741824
equal=False
tvm_code=82678120448
np_res = 1525142128399588498675721043968, tvm_res = 1073741824
mismatch=ShiftLeft.apply(BitwiseOr.apply(IntLit.apply(76),Abs.apply(BoolLit.apply(True))),IntLit.apply(94))
	 BitwiseOr np val=77, tvm val=77
	 IntLit np val=94, tvm val=94
```

A testcase reduction algorithm is run to narrow down where in the tree the exprs no longer match. In this example it found the mismatch to be a `ShiftLeft` op. In ths particular example the ground-truth program does not properly matching the overflow behaviour of the TVM program. Nonetheless, it was able to discover the mismatch and report it to the user for further investigation. 

After the mismatch is detected the algorithm produces all the arguments passed to the mismatching OP. In this example it shows that `77 << 94` is the expression that executed. This output also helps show that the mismatch detected is the *minimal* mismatching tree. 

