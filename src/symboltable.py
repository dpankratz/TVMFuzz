import random
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