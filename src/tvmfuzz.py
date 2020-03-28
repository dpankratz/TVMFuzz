from expr_generation import generate_tvm_and_np_tree
from symboltable import SymbolTable
from test_bed import evaluate_tvm_expr,compare_results,evaluate_np_expr
from generation_node import GenerationNode
from termcolor import colored
from util import get_literal_value
import random, sys, traceback, datetime

def run(timestamp = None, repetitions = 1):

	seed = timestamp if timestamp else None


	for _ in range(repetitions):
		if not seed:
			seed = datetime.datetime.utcnow().timestamp()
			print("timestamp ={0}\n".format(seed))

		random.seed(seed)

		seed = None

		root = generate_tvm_and_np_tree()

		tvm_expr = root.emit_tvm()

		print("tree={0}".format(colored(root,"cyan")))

		print("tvm expr={0}".format(colored(tvm_expr,"yellow")))
		np_expr = root.emit_np()
		
		SymbolTable.populate()

		print("SymbolTable.binds={0}".format(SymbolTable.binds))

		print(evaluate_tvm_expr(tvm_expr,SymbolTable.variables,SymbolTable.binds))

		np_result = evaluate_np_expr(np_expr)
		if (np_result == "Runtime Exception"):
			print("np error={0}".format(GenerationNode.NP_CULPRIT))

		print("np result={0}".format(np_result))

		lit_value = get_literal_value(tvm_expr)
		if lit_value:
			tvm_result = lit_value
			print("tvm result={0} found in front end".format(tvm_result))
		else:
			tvm_result = evaluate_tvm_expr(tvm_expr,SymbolTable.variables,SymbolTable.binds)
			if(tvm_result == "Runtime Exception"):
				print("tvm error={0}".format(GenerationNode.TVM_CULPRIT))
			print("tvm result={0}".format(tvm_result))


		if(np_result == None and tvm_result == None):
			print("Both crashed.")
		else:
			
			is_equal = compare_results(np_result,tvm_result)
			print("equal={0}".format(colored("True","green") if is_equal else colored("False","red")))
			if (not is_equal):
				root.find_mismatch()
				print("mismatch={0}".format(colored(GenerationNode.MISMATCH_CULPRIT,"red")))
				for arg in GenerationNode.MISMATCH_CULPRIT.m_args:
					print("\t {0} np val={1}, tvm val={2}".format(arg.m_op.__name__,evaluate_np_expr(arg.m_emitted_np_op),evaluate_tvm_expr(arg.m_emitted_tvm_op,SymbolTable.variables,SymbolTable.binds)))
				if repetitions > 1:
					print("PRESS ENTER TO CONTINUE FUZZING!")
					input()

if __name__ == "__main__":
	import argparse

	arg_parser = argparse.ArgumentParser(description = 'Control the actions of TVMFuzz.')
	arg_parser.add_argument("--Repetitions", metavar="1", type=int, default=1,
								help="Number of programs to generate!")
	arg_parser.add_argument("--Timestamp", metavar = "123456.12345", type=float, default=None,
								help="Use old seed to execute")

	args = arg_parser.parse_args()

	run(args.Timestamp, args.Repetitions)