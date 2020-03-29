import random

class ProbabilisticSelection:
	""" Randomly select options from a weight table
	
	The probability that a given item is selected is
	calculated as its weight over the sum of all weights
	in the table. 

	Parameters
	----------
	choices : dict{int, object}
		Dict of weights and options

	"""
	def __init__(self,choices):
		total = 0.0
		for weight,_ in choices:
			total += weight

		running_total = 0.0
		self.probs = []
		for weight,val in choices:
			running_total += weight
			prob = running_total / total
			self.probs.append((prob,val))


	def select(self):
		""" Randomly select item from table
		
		Returns
		-------
		selection : object
			Object that was selected

		"""
		rand = random.random()
		begin = 0
		for end,val in self.probs:
			if rand >= begin and rand <= end:
				return val
			begin = end

		raise ValueError("Rand value {0} fell through".format(rand))

if(__name__ == "__main__"):
	ps = ProbabilisticSelection([
		(1,1),
		(1,2),
		(1,3),
		(1,4)
	])

	def select(self,rand_val):
		rand = rand_val
		begin = 0
		for end,val in self.probs:
			if rand >= begin and rand <= end:
				return val
			begin = end

		raise ValueError("Rand value {0} fell through".format(rand))

	
	assert(select(ps,0) == 1)
	assert(select(ps,0.3) == 2)
	assert(select(ps,0.49) == 2)
	assert(select(ps,1) == 4)
	assert(select(ps,0.6) == 3)