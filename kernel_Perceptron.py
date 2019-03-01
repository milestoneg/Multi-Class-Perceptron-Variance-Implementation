import numpy as np
from functools import reduce

class kernel_Perceptron(object):
	def __init__(self, input_num, data, kernel = "poly", d = 1):
		self.weights  = [0.0 for _ in range(input_num)]
		self.bias = 0.0
		self.data = data
		self.kernel = kernel
		self.d = d

	def __str__(self):
		return "weights:%s \n bias:%f"%(list(self.weights), self.bias)

	def predict(self, input_vector):
		if (self.kernel == "poly"):
			K = self.poly_kernel(self.data,input_vector,self.d)
			cal = (self.weights @ K)
		else:
			K = self.gaussian_kernel(self.data,input_vector,2)
			print(K.shape)
			cal = self.weights @ K

		return self.activator_sign(cal)
	
	def train(self, input_vectors, labels, interation, rate):
		for i in range(interation):
			self.interationfuc(input_vectors, labels, rate)
			return self.weights
			
	def interationfuc(self, input_vectors, labels, rate):
		#zipped_dataset = zip(input_vectors,labels)
		for i in input_vectors:
			#print("weights before before:"+ str(list(self.weights)))
			predict = self.predict(input_vectors[i])
			#print("weights before:"+ str(list(self.weights)))
			self.update(input_vectors[i], predict, labels[i], rate, i)
			#print("weights after:"+ str(list(self.weights)))

	def update(self, input_vec, predict, label, rate, index):
		if predict == label:
			self.weights[index] = 0
		else:
			if (self.kernel =="poly"):
				self.weights[index] = label
				self.weights[index] += label*poly_kernel(input_vec,input_vec,self.d)
			else:
				self.weights[index] += label*gaussian_kernel(input_vec,input_vec,2)

	def poly_kernel(self, p, q, d):
		return (p @ q.T)**d

	def gaussian_kernel(self, p, q, c):
		K = np.exp(-c * np.linalg.norm(p - q)**2)
		return K

	def activator_sign(self, x):
		if x > 0:
			return 1
		elif x == 0:
			return 0
		else:
			return -1



'''
def f(x):
	if x>0:
		return 1
	else:
		return 0



def start_train():
	print("start_train called!")
	p = kernel_Perceptron(2)
	input_vecs, lable = data_generator()
	p.train(input_vecs,lable, 10, 0.1)
	return p

if __name__ == '__main__':
	p = start_train()
	print(p)
	print("Predict result: " + str(p.predict([1,1])))
'''