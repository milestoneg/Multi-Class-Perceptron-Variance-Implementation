import numpy as np
import math
from functools import reduce
from numba import jit

class kernel_Perceptron(object):
	def __init__(self, input_num, data, labels, d, kernel = "poly"):
		#self.weights  = np.array([0.0 for _ in range(input_num)]) np.zeros(input_num)
		self.weights  = np.zeros(input_num)
		self.bias = 0.0
		self.data = data
		self.labels = labels
		self.kernel = kernel
		self.d = d
		self.mistakes = 0

	def __str__(self):
		return "weights:%s \n bias:%f"%(list(self.weights), self.bias)

	@jit(parallel= True)
	def predict(self, input_vector):
		if (self.kernel == "poly"):
			
			K = self.poly_kernel(self.data, input_vector, self.d)
			cal = self.weights @ K 
			
			'''
			cal=0
			for i in range(len(self.data)):
				cal  = cal + self.weights[i]*self.poly_kernel(self.data[i],input_vector,self.d)
			'''
		else:
			'''
			cal = 0
			for i in range(len(self.data)):
				cal = cal + self.gaussian_kernel(self.data[i], input_vector, self.d)*self.weights[i]
			'''
			input_vector_mtx = np.zeros(shape = (len(self.data),len(input_vector)))
			for i in range(len(input_vector_mtx)):
				input_vector_mtx[i] = input_vector
			K = self.gaussian_kernel(self.data,input_vector_mtx,self.d)
			cal = self.weights @ K
			
		#return self.activator_sign(cal)
		return cal
	
	def train(self, input_vectors, labels):
		self.mistakes = 0
		self.interationfuc(input_vectors, labels)
		return self.mistakes

	def interationfuc(self, input_vectors, labels):
		#zipped_dataset = zip(input_vectors,labels)
		for i in range(len(input_vectors)):
			predict = self.predict(input_vectors[i])
			self.update(input_vectors[i], predict, i, labels)
		#print("----------")

	@jit
	def update(self, input_vec, predict, index, labels):
		if self.activator_sign(predict) != labels[index]:
			self.mistakes = self.mistakes + 1
			self.weights[index] += labels[index]
			

	@jit
	def poly_kernel(self, p, q, d):
		return (p @ q.T)**d

	@jit
	def gaussian_kernel(self, p, q, c):
		K = np.exp(-c * np.linalg.norm(p - q, axis = 1)**2)
		return K

	def activator_sign(self, x):
		if x > 0:
			return 1
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