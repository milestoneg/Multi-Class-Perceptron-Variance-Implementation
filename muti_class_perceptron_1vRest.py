"""
Created on·1st·Dec·2018

@author: Yuan Gao ucabyg5 18064382
"""
#-----------------------------------------------------------
from kernel_Perceptron_1 import kernel_Perceptron
import copy
import numpy as np
from numba import jit
import time
import numexpr as ne

class k_class_perceptron_1vRest(object):
	def __init__(self, input_num, train_set, labels, class_num, d, kernel = "poly"):
		self.weights  = [0.0 for _ in range(input_num)]
		self.class_num = class_num
		self.train_set = train_set
		self.labels = labels
		self.kernel = kernel
		self.d = d
		self.classifier_mtx = np.zeros(shape= (class_num, input_num))
	
	def predict(self, input_vector):
		if(self.kernel == "poly"):
			K = self.poly_kernel(input_vector, self.train_set, self.d)
		else:
			input_vector_mtx = np.tile(input_vector, (len(self.train_set),1))
			K = self.gaussian_kernel(input_vector_mtx, self.train_set, self.d)
		confident = self.classifier_mtx @ K
		confident = confident.tolist()
		result = confident.index(max(confident))
		return confident, result

	def predict_mtx(self, test_set):
		if(self.kernel == "poly"):
			K = self.poly_kernel(test_set, self.train_set, self.d)
		else:
			K = self.gaussian_kernel(test_set, self.train_set, self.d)
		confident_mtx = self.classifier_mtx @ K.T
		print((confident_mtx.T).shape)
		return confident_mtx.T

	def train(self, interation):
		if(self.kernel == "poly"):
			K_train = self.poly_kernel(self.train_set, self.train_set, self.d)
		else:
			K_train = self.gaussian_kernel(self.train_set, self.train_set, self.d)
		error_list = []
		for s in range(interation):
			start_time = time.time()
			error = 0
			for i in range(len(self.train_set)):
				confidence = self.classifier_mtx @ K_train[i]
				for index in range(len(confidence)):
					if(confidence[index]>0):
						if(index != self.labels[i]):
							error += 1
							self.classifier_mtx[index][i] += -1
					if(confidence[index]<=0):
						if(index == self.labels[i]):
							error += 1
							self.classifier_mtx[index][i] += 1
			error_list.append(error)
			end_time = time.time()
			print("Epoch {0}, mistakes {1}, time {2}".format(s,error,end_time-start_time))
		return error_list

	@jit
	def poly_kernel(self, p, q, d):
		return (p @ q.T)**d

	@jit
	def gaussian_kernel(self, p, q, c):
		X = p
		X_norm = np.sum(X ** 2, axis = -1)
		Y = q
		Y_norm = np.sum(Y ** 2, axis = -1)
		K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
			'A' : X_norm[:,None],
			'B' : Y_norm[None,:],
			'C' : np.dot(X, Y.T),
			'g' : c
		})
		return K

