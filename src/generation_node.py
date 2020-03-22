import traceback
from test_bed import compare_results,evaluate_tvm_expr,evaluate_np_expr
from symboltable import SymbolTable

class GenerationNode(object):
	TVM_CULPRIT = None
	NP_CULPRIT = None
	MISMATCH_CULPRIT = None

	def __init__(self,op,args = []):
		self.m_op = op
		self.m_args = args
		self.m_emitted_np_op = None
		self.m_emitted_tvm_op = None
		self._random = None

	def emit_tvm(self):
		if(self.is_leaf()):
			self.m_emitted_tvm_op = self.m_op.apply()
			self._random = self.m_op.last_random
			return self.m_emitted_tvm_op

		args = []
		for arg in self.m_args:
			try:
				tvm_ir = arg.emit_tvm()
				if(tvm_ir == None):
					return None
				args.append(tvm_ir)
			except Exception:
				traceback.print_exc()
				GenerationNode.TVM_CULPRIT = arg
				return None

		self.m_emitted_tvm_op = self.m_op.apply(*args)
		return self.m_emitted_tvm_op

	def emit_np(self):
		if(self.is_leaf()):
			self.m_op.last_random = self._random
			self.m_emitted_np_op = self.m_op.apply_np()
			return self.m_emitted_np_op

		args = []
		for arg in self.m_args:
			try:
				np_expr = arg.emit_np()
				if(np_expr == None):
					return None
				args.append(np_expr)
			except Exception:
				traceback.print_exc()
				GenerationNode.NP_CULPRIT = arg
				return None
		self.m_emitted_np_op = self.m_op.apply_np(*args)
		return self.m_emitted_np_op

	def find_mismatch(self):
		for arg in self.m_args:
			if (not arg.find_mismatch()):
				return False


		tvm_res = evaluate_tvm_expr(self.m_emitted_tvm_op,SymbolTable.variables,SymbolTable.binds,suppress_errors = True)
		np_res = evaluate_np_expr(self.m_emitted_np_op)
		is_equal = compare_results(tvm_res,np_res)
		if (not is_equal):
			print("tvm_code={0}".format(self.m_emitted_tvm_op))
			print("np_res = {0}, tvm_res = {1}".format(np_res,tvm_res))
			GenerationNode.MISMATCH_CULPRIT = self			
			return False 

		return True

	def is_leaf(self):
		return len(self.m_args) == 0

	def __str__(self):
		if (len(self.m_args) == 0):
			return self.m_op.__name__ + ("(" + str(self._random) + ")" if self._random != None else "")


		ret = self.m_op.__name__ + "("
		body = ""
		for i in range(len(self.m_args)):
			body = body + str(self.m_args[i]) + ("," if i < len(self.m_args) - 1 else "")
		ret += body + ")"
		return ret