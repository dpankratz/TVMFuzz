import __init__
from symboltable import SymbolTable
from expression import *
from test_bed import evaluate_tvm_expr,evaluate_np_expr
SymbolTable.recover_from_binds({'f0': -78.89981616893917, 'f1': -92.5815113182746})
tvm_expr=ShiftLeft.apply(IfThenElse.apply(TruncMod.apply(Sub.apply(NE.apply(BoolLit.apply(False),BoolLit.apply(True)),GT.apply(ExistingVar.apply('f0'),ExistingVar.apply('f0'))),ExistingVar.apply('f0')),Mul.apply(ShiftLeft.apply(Pow.apply(IntLit.apply(85),BoolLit.apply(True)),GE.apply(BoolLit.apply(True),BoolLit.apply(False))),BoolLit.apply(True)),Max.apply(BoolLit.apply(True),IntLit.apply(78))),Pow.apply(IntLit.apply(24),IntLit.apply(21)))
np_expr=ShiftLeft.apply_np(IfThenElse.apply_np(TruncMod.apply_np(Sub.apply_np(NE.apply_np(BoolLit.apply_np(False),BoolLit.apply_np(True)),GT.apply_np(ExistingVar.apply_np('f0'),ExistingVar.apply_np('f0'))),ExistingVar.apply_np('f0')),Mul.apply_np(ShiftLeft.apply_np(Pow.apply_np(IntLit.apply_np(85),BoolLit.apply_np(True)),GE.apply_np(BoolLit.apply_np(True),BoolLit.apply_np(False))),BoolLit.apply_np(True)),Max.apply_np(BoolLit.apply_np(True),IntLit.apply_np(78))),Pow.apply_np(IntLit.apply_np(24),IntLit.apply_np(21)))
print(evaluate_tvm_expr(tvm_expr))
print(evaluate_np_expr(np_expr))
