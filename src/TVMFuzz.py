from ExprGeneration import generate_tvm_and_np_tree
from SymbolTable import SymbolTable
from test_bed import evaluate_tvm_expr,compare_results
from generation_node import GenerationNode
import random
import traceback

root = generate_tvm_and_np_tree()

print(root)

tvm_expr = root.emit_tvm()

print(tvm_expr)
np_expr = root.emit_np()


binds = {}
for var in SymbolTable.variables:

	if(var.dtype == 'int32'):
		SymbolTable.binds[var.name] = random.randint(-10,10)
	elif(var.dtype == 'float32'):
		SymbolTable.binds[var.name] = random.random() * 20 - 10
	print(var.name + " = " + str(SymbolTable.binds[var.name]))

tvm_result = evaluate_tvm_expr(tvm_expr,SymbolTable.variables,SymbolTable.binds)
if(tvm_result == None):
	print("tvm error={0}".format(GenerationNode.TVM_CULPRIT))
print("tvm result={0}".format(tvm_result))


np_result = np_expr()
if (np_result == None):
	print("np error={0}".format(GenerationNode.NP_CULPRIT))

print("np result={0}".format(np_result))


if(np_result == None and tvm_result == None):
	print("Both crashed.")
else:
	
	
	print("equal={0}".format(compare_results(np_result,tvm_result)))
	print("mismatch={0}".format(GenerationNode.MISMATCH_CULPRIT))