import random
from tvm import te,tir
from tvmfuzz_config import TVMFuzzConfig

class SymbolTable:
	"""
	Stores the create variable metadata
	"""
	variables = []
	binds = {}

	@staticmethod
	def populate():
		for var in SymbolTable.variables:
			if(var.dtype == 'int32'):
				SymbolTable.binds[var.name] = random.randint(TVMFuzzConfig.bind_min_value,TVMFuzzConfig.bind_max_value)
			elif(var.dtype == 'float32'):
				bind_range = TVMFuzzConfig.bind_max_value - TVMFuzzConfig.bind_min_value
				SymbolTable.binds[var.name] = random.random() * bind_range - (bind_range / 2)
			else:
				raise ValueError("Unable to populate variable {0} with type {1}.".format(var.name,var.dtype))

	@staticmethod
	def recover_from_binds(binds):
		SymbolTable.variables.clear()

		for name,val in binds.items():
			if(name[0] == 'i'):
				SymbolTable.variables.append(te.var(name=name,dtype='int32'))
			elif(name[0] == 'f'):
				SymbolTable.variables.append(te.var(name=name,dtype='float32'))
			else:
				raise ValueError("Unable to recover variable {0}".format(var.name))

		SymbolTable.binds = binds
