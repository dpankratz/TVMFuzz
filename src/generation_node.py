import traceback
from test_bed import compare_results,evaluate_tvm_expr


class GenerationNode(object):
	TVM_CULPRIT = None
	NP_CULPRIT = None
	MISMATCH_CULPRIT = None

	def __init__(self,op,args = []):
		self.m_op = op
		self.m_args = args
		self.m_emitted_np_args = []
		self.m_emitted_tvm_args = []
		self._random = None

	def emit_tvm(self):
		if(self.is_leaf()):
			ret = self.m_op.apply()
			self._random = self.m_op.last_random
			return ret

		self.m_emitted_tvm_args = []
		for arg in self.m_args:
			try:
				tvm_ir = arg.emit_tvm()
				if(tvm_ir == None):
					return None
				self.m_emitted_tvm_args.append(tvm_ir)
			except Exception:
				traceback.print_exc()
				GenerationNode.TVM_CULPRIT = arg
				return None

		return self.m_op.apply(*self.m_emitted_tvm_args)

	def emit_np(self):
		if(self.is_leaf()):
			self.m_op.last_random = self._random
			return self.m_op.apply_np()

		self.m_emitted_np_args = []
		for arg in self.m_args:
			try:
				np_expr = arg.emit_np()
				if(np_expr == None):
					return None
				self.m_emitted_np_args.append(np_expr)
			except Exception:
				traceback.print_exc()
				GenerationNode.NP_CULPRIT = arg
				return None

		return self.m_op.apply_np(*self.m_emitted_np_args)

	def find_mismatch(self):
		if(self.is_leaf()):
			return True

		for arg in self.m_args:
			if (not arg.find_mismatch()):
				return False

		assert len(self.m_emitted_np_args) == len(self.m_emitted_tvm_args)
		for i in range(len(self.m_emitted_np_args)):
			tvm_expr = self.m_emitted_tvm_args[i]
			np_expr = self.m_emitted_np_args[i]
			tvm_res = evaluate_tvm_expr(tvm_expr,SymbolTable.variables,SymbolTable.binds)
			np_res = np_expr()
			if (not compare_results(tvm_res,np_res)):
				GenerationNode.MISMATCH_CULPRIT = np_expr				
				return False

		return True




		
	def is_leaf(self):
		return len(self.m_args) == 0

	def __str__(self):
		if (len(self.m_args) == 0):
			return self.m_op.__name__


		ret = self.m_op.__name__ + "("
		body = ""
		for i in range(len(self.m_args)):
			body = body + str(self.m_args[i]) + ("," if i < len(self.m_args) - 1 else "")
		ret += body + ")"
		return ret