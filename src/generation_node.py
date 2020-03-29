import traceback

from termcolor import colored
from test_bed import compare_results,evaluate_tvm_expr,evaluate_np_expr
from symboltable import SymbolTable

class GenerationNode(object):
	""" Composing instances of this class form a Tree which can emit python or tvm code

	Attributes
	----------
	m_op : type
		The operator to apply e.g. expression.Add
	m_args : list of GenerationNode
		The arguments to provide to m_op
	m_emitted_np_op : function
		The numpy expression emitted from this node
	m_emitted_tvm_op : tvm.expr.PrimExpr
		The tvm expression emitted from this node

	Parameters
	----------
	op : type
		Operator to apply. E.g. expression.Add
	args : list of GeneationNode
		Args to provide to op

	"""

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
		""" Create a TVM expr with the current node as the root.

		Returns
		-------
		tvm_expr : tir.expr.PrimExpr
			The expr built recursively from this noe
		"""

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
		""" Creates a numpy expr with the current node as the root.

		Returns
		-------
		np_expr : function
			The function to build the numpy expr
		"""
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
		""" Detect where TVM and NP programs compute different results
		
		Sets GenerationNode.MISMATCH_CULPRIT to GenerationNode causing
		mismatch.
		
		Returns
		------
		is_mismatch : bool
			True if mismatch was found. False otherwise.

		"""
		for arg in self.m_args:
			if (not arg.find_mismatch()):
				return False


		tvm_res = evaluate_tvm_expr(self.m_emitted_tvm_op,suppress_errors = True)
		np_res = evaluate_np_expr(self.m_emitted_np_op)
		is_equal = compare_results(tvm_res,np_res)
		if (not is_equal):
			print("tvm_code={0}".format(colored(self.m_emitted_tvm_op,"yellow")))
			print("np_res = {0}, tvm_res = {1}".format(np_res,tvm_res))
			GenerationNode.MISMATCH_CULPRIT = self			
			return False 

		return True

	def is_leaf(self):
		return len(self.m_args) == 0

	def __str__(self):
		if (len(self.m_args) == 0):
			last_random_str = ""
			if isinstance(self._random,str):
				last_random_str += "\'" + self._random + "\'"
			else: 
				last_random_str = str(self._random)
			return self.m_op.__name__ + (".apply(" + last_random_str + ")" if self._random != None else "")


		ret = self.m_op.__name__ + ".apply("
		body = ""
		for i in range(len(self.m_args)):
			body = body + str(self.m_args[i]) + ("," if i < len(self.m_args) - 1 else "")
		ret += body + ")"
		return ret