from expr_generation import generate_tvm_and_np_tree
from symboltable import SymbolTable
from test_bed import evaluate_tvm_expr,compare_results,evaluate_np_expr
from generation_node import GenerationNode
import random, sys, traceback, dill, pickle, datetime

if(len(sys.argv)) == 1:
	seed = datetime.datetime.utcnow().timestamp()
	print("timestamp ={0}\n".format(seed))
else:
	seed = float(sys.argv[1])

random.seed(seed)

root = generate_tvm_and_np_tree()

tvm_expr = root.emit_tvm()

print("tree={0}\n".format(root))

print("tvm expr={0}\n".format(tvm_expr))
np_expr = root.emit_np()



binds = {}
for var in SymbolTable.variables:

	if(var.dtype == 'int32'):
		SymbolTable.binds[var.name] = random.randint(-10,10)
	elif(var.dtype == 'float32'):
		SymbolTable.binds[var.name] = random.random() * 20 - 10
	print(var.name + " = " + str(SymbolTable.binds[var.name]))

np_result = evaluate_np_expr(np_expr)
if (np_result == "Runtime Exception"):
	print("np error={0}".format(GenerationNode.NP_CULPRIT))

print("np result={0}".format(np_result))


tvm_result = evaluate_tvm_expr(tvm_expr,SymbolTable.variables,SymbolTable.binds)
if(tvm_result == "Runtime Exception"):
	print("tvm error={0}".format(GenerationNode.TVM_CULPRIT))
print("tvm result={0}".format(tvm_result))


if(np_result == None and tvm_result == None):
	print("Both crashed.")
else:
	
	is_equal = compare_results(np_result,tvm_result)
	print("equal={0}".format(is_equal))
	if (not is_equal):
		root.find_mismatch()
		print("mismatch={0}".format(GenerationNode.MISMATCH_CULPRIT))
		for arg in GenerationNode.MISMATCH_CULPRIT.m_args:
			print("\t {0} np val={1}, tvm val={2}".format(arg.m_op.__name__,evaluate_np_expr(arg.m_emitted_np_op),evaluate_tvm_expr(arg.m_emitted_tvm_op,SymbolTable.variables,SymbolTable.binds)))
