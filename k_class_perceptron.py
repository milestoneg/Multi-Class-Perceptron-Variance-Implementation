from kernel_Perceptron_1 import kernel_Perceptron
import copy
import numpy as np
from numba import jit
import time

class k_class_perceptron(object):
	def __init__(self, input_num, data, labels, class_num, d,kernel = "poly"):
		self.weights  = [0.0 for _ in range(input_num)]
		self.bias = 0.0
		self.data = data
		self.labels = labels
		self.class_num = class_num
		self.kernel = kernel
		self.d = d
		self.classfier_list = [kernel_Perceptron(input_num, data, labels, d, kernel) for _ in range(class_num)]


	def predict(self, input_vector):
		digits = [0,1,2,3,4,5,6,7,8,9]
		confident = [0.0 for _ in range(self.class_num)]
		for i in range(len(self.classfier_list)):
			confident[i] = self.classfier_list[i].predict(input_vector)
		result = digits[confident.index(max(confident))]
		return confident, result
	
	@jit
	def train(self, input_vectors, interation):
		digits = [0,1,2,3,4,5,6,7,8,9]
		mistakes_list = []
		for index in range(interation):
			mistakes = 0
			S = time.time()
			for i in range(len(self.classfier_list)):
				#construct label set for each classifier
				labels_copy = copy.copy(self.labels)
				for d in range(len(labels_copy)):
					if(labels_copy[d] != digits[i]):
						labels_copy[d] = -1
					else:
						labels_copy[d] = 1
				mistakes = mistakes + self.classfier_list[i].train(input_vectors,labels_copy)
				E = time.time()
			mistakes_list.append(mistakes)
			#print(mistakes)
			print("Epoch {0}, mistakes {1}, time {2}".format(index,np.sum(mistakes),E-S))
		return mistakes_list
