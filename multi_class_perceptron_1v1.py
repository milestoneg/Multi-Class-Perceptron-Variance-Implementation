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

class k_class_perceptron_1v1(object):
	def __init__(self, input_num, train_set, labels, class_num, d, kernel = "poly"):
		self.weights  = [0.0 for _ in range(input_num)]
		self.class_num = class_num
		self.train_set = train_set
		self.labels = labels
		self.kernel = kernel
		self.d = d
		self.classifier_mtx = np.zeros(shape= (int(class_num*(class_num-1)/2), input_num))
		self.index_list = []
		for i in range(class_num):
			for j in range(i+1, class_num):
				index = (i,j)
				self.index_list.append(index)
	
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
		
		result_mtx = self.classifier_mtx @ K.T
		print(result_mtx.shape)
		
		result_mtx = result_mtx.T
		
		confident_mtx = np.zeros(shape = (len(test_set),self.class_num))
		for i in range(len(result_mtx)):
			confident_list = np.zeros(shape = (10,1))
			for g in range(len(result_mtx[i])):
				classification = self.index_list[g]
				if(result_mtx[i][g]>0):
					confident_list[classification[0]] += 1
				else:
					confident_list[classification[1]] += 1
			confident_mtx[i] = confident_list.T
		return confident_mtx

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
				for index in self.index_list:
					if index[0] == self.labels[i]:
						position = self.index_list.index(index)
						if(confidence[position]<=0):
							error +=1
							self.classifier_mtx[position][i] += 1
					if index[1] == self.labels[i]:
						position = self.index_list.index(index)
						if(confidence[position]>0):
							error += 1
							self.classifier_mtx[position][i] += -1
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

