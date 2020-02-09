import random

class ProbabilisticSelection:

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
		rand = random.random()
		begin = 0
		self.last_selection = 0
		for end,val in self.probs:
			if rand >= begin and rand <= end:
				return val
			begin = end

		raise ValueError("Rand value {0} fell through".format(rand))
