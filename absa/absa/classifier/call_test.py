class Call_test():
	def __init__(self, name):
		self.name = name
	def __call__(self, say):
		print(f'{self.name} say {say}')
		