"""
Created on·1st·Dec·2018

@author: Yuan Gao ucabyg5 18064382
"""
#-----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#from k_class_perceptron import k_class_perceptron
from sklearn.model_selection import train_test_split, KFold
from prettytable import PrettyTable
from muti_class_perceptron_1vRest import k_class_perceptron_1vRest
from multi_class_perceptron_1v1 import k_class_perceptron_1v1
#from skmultilearn.adapt import MLkNN
#from kernel_Perceptron_1 import kernel_Perceptron
import cos_knn as knn
import svc as svc
import randomforeast as rfc

data = pd.read_table("zipcombo.dat", sep="\s+")
data = np.array(data)
error_images_index = []


def part1_1():
	iteration = 20 #set training epochs
	result_table = PrettyTable(["train errors ± STD","test errors ± STD"])
	train_result_mtx = np.zeros(shape = (7,20))
	test_result_mtx = np.zeros(shape = (7,20))
	#iterate d from 1 to 7 
	for d in range(1,8):
		mistake_train_mtx = np.zeros(shape = (20,iteration))
		mistake_test_mtx = np.zeros(shape = (20,1))
		#run 20 time
		for i in range(20):
			print("---------------")
			print("d = " + str(d)+", round: "+ str(i))
			#split data set
			data_train ,data_test = train_test_split(data,test_size=0.2)
			x_train = data_train[:,1:]
			y_train = data_train[:,0]
			x_test = data_test[:,1:]
			y_test = data_test[:,0]


			#train perceptron
			#get training error and test error
			kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10,d)
			mistake_train = kp.train(iteration)
			mistake_train_mtx[i] = mistake_train
			mistake_test = test_classifiers(x_test,y_test,kp)
			mistake_test_mtx[i] = mistake_test
		
		#store data
		mistake_train_mtx = np.sum(mistake_train_mtx, axis=1)
		mistake_test_mtx = np.sum(mistake_test_mtx, axis=1)
		train_result_mtx[d-1] = mistake_train_mtx
		test_result_mtx[d-1] = mistake_test_mtx

	#calculate mean value and std
	for i in range(len(train_result_mtx)):
		mean_train_error = np.sum(train_result_mtx[i])/20
		train_std = np.std(train_result_mtx[i], ddof = 1)
		mean_test_error = np.sum(test_result_mtx[i])/20
		test_std = np.std(test_result_mtx[i], ddof = 1)
		result_table.add_row([str(mean_train_error) + ' ± ' +str(train_std), str(mean_test_error)+ ' ± ' +str(test_std)])
	print(result_table)

def part1_2():
	interation = 5#set training epochs
	result_mtx = np.zeros(shape=(20,2))
	#run 20 times 
	for i in range(20):
		#split data set
		data_train ,data_test = train_test_split(data,test_size=0.2)
		x_train = data_train[:,1:]
		y_train = data_train[:,0]
		x_test = data_test[:,1:]
		y_test = data_test[:,0]
		mistake_list = []
		#iteration d from 1 to 7
		for d in range(1,8):
			print("-----------------------")
			print("now d is: "+ str(d))
			print("-----------------------")
			mistake = k_fold_crossvalidation(5, x_train, y_train, d, interation)
			mistake_list.append(mistake)
		d_star = mistake_list.index(min(mistake_list))+1 #get the best d
		#train perceptron
		#get test error 
		kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10,d_star)
		kp.train(interation)
		test_error = test_classifiers(x_test,y_test,kp)
		result_mtx[i][0] = d_star
		result_mtx[i][1] = test_error
	#calculate mean value and std
	mean_error = np.sum(result_mtx, axis = 0)[1]/20
	error_std = np.std(result_mtx, axis = 0)[1]
	mean_d_star =  np.sum(result_mtx, axis = 0)[0]/20
	d_star_std = np.std(result_mtx, axis = 0)[0]
	print("-------------")
	print("mean test error: "+ str(mean_error) + " ± " + str(error_std))
	print("mean d*: " + str(mean_d_star) + " ± " + str(d_star_std))

def part1_3():
	interation = 15#set epochs set
	confusion_mat = np.zeros(shape=(10,10))
	STD_mat = np.zeros(shape=(10,10))
	single_confusion_mtx_list = []
	#run 20 time
	for i in range(20):
		single_confusion_mtx = np.zeros(shape = (10,10))
		#split data set 
		data_train ,data_test = train_test_split(data,test_size=0.2)
		x_train = data_train[:,1:]
		y_train = data_train[:,0]
		x_test = data_test[:,1:]
		y_test = data_test[:,0]
		mistake_list = []
		#iterate d from 1 to 7
		for d in range(1,8):
			print("-----------------------")
			print("now d is: "+ str(d))
			print("-----------------------")
			mistake = k_fold_crossvalidation(5, x_train, y_train, d, interation)
			mistake_list.append(mistake)
		d_star = mistake_list.index(min(mistake_list))+1#get the best d
		print(d_star)
		#train perceptron
		kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10,d_star)
		kp.train(interation)
		#for each run create a confusion matrix
		for i in range(len(x_train)):
			confident, predict = kp.predict(x_train[i])
			if(int(predict) != int(y_train[i])):
				single_confusion_mtx[int(y_train[i])][int(predict)] += 1
				error_images_index.append(x_train[i])
		single_confusion_mtx_list.append(single_confusion_mtx)
	#calculate sum of 20 single_confusion_mtx
	for i in range(len(single_confusion_mtx_list)):
		confusion_mat += single_confusion_mtx_list[i]
	#calculate STD for each single_confusion_mtx
	tmp = np.zeros(shape=(10,10))
	for row in range(len(tmp)):
		for column in range(len(tmp)):
			element_list = []
			for i in single_confusion_mtx_list:
				element = i[row][column]
				element_list.append(element)
			element_list = np.array(element_list)
			std = np.std(element_list)
			STD_mat[row][column] = std
	print("-----------------------")
	print(confusion_mat)
	print(STD_mat)
	print(error_images)
	
def part1_4():
	printed_images = []
	#print all digits records before
	for digit in error_images:
		for images in printed_images:
			if digit != images:
				plt.imshow(digit.reshape(16,16))
				plt.show()
				printed_images.append(digit)


def part1_5_1():
	iteration = 5#set epochs number
	result_table = PrettyTable(["train errors ± STD","test errors ± STD"])
	train_result_mtx = np.zeros(shape = (7,20))
	test_result_mtx = np.zeros(shape = (7,20))
	#iterate d from -18 to -11
	for d in range(-18,-11):
		mistake_train_mtx = np.zeros(shape = (20,iteration))
		mistake_test_mtx = np.zeros(shape = (20,1))
		#run 20 times
		for i in range(20):
			print("---------------")
			print("d = " + str(2**(d/2))+", round: "+ str(i))
			#split data set
			data_train ,data_test = train_test_split(data,test_size=0.2)
			x_train = data_train[:,1:]
			y_train = data_train[:,0]
			x_test = data_test[:,1:]
			y_test = data_test[:,0]
			#perceptron
			#get train error and test error
			kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10,2**(d/2),"gaussain")
			mistake_train = kp.train(iteration)
			mistake_train_mtx[i] = mistake_train
			mistake_test = test_classifiers(x_test,y_test,kp)
			mistake_test_mtx[i] = mistake_test
		#construct result matrix
		mistake_train_mtx = np.sum(mistake_train_mtx, axis=1)
		mistake_test_mtx = np.sum(mistake_test_mtx, axis=1)
		train_result_mtx[int(d+18)] = mistake_train_mtx
		test_result_mtx[int(d+18)] = mistake_test_mtx
	#calculate mean and std
	for i in range(len(train_result_mtx)):
		mean_train_error = np.sum(train_result_mtx[i])/20
		train_std = np.std(train_result_mtx[i], ddof = 1)
		mean_test_error = np.sum(test_result_mtx[i])/20
		test_std = np.std(test_result_mtx[i], ddof = 1)
		result_table.add_row([str(mean_train_error) + ' ± ' +str(train_std), str(mean_test_error)+ ' ± ' +str(test_std)])
	print(result_table)

def part1_5_2():
	interation = 5#set epochs numbers
	result_mtx = np.zeros(shape=(20,2))
	#run 20 times 
	for i in range(20):
		data_train ,data_test = train_test_split(data,test_size=0.2)
		x_train = data_train[:,1:]
		y_train = data_train[:,0]
		x_test = data_test[:,1:]
		y_test = data_test[:,0]
		mistake_list = []
		#iterate d from -18 to -11
		for d in range(-18,-11):
			print("-----------------------")
			print("now d is: "+ str(d))
			print("-----------------------")
			mistake = k_fold_crossvalidation(5, x_train, y_train, 2**(d/2), interation, "gaussain")
			mistake_list.append(mistake)
		d_star = mistake_list.index(min(mistake_list))-18#calculate best d
		#train perceptron get train error and test error
		kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10,2**(d_star/2),"gaussain")
		kp.train(interation)
		test_error = test_classifiers(x_test,y_test,kp)
		result_mtx[i][0] = d_star
		result_mtx[i][1] = test_error
	#calculate mean and std
	mean_error = np.sum(result_mtx, axis = 0)[1]/20
	error_std = np.std(result_mtx, axis = 0)[1]
	mean_d_star =  np.sum(result_mtx, axis = 0)[0]/20
	d_star_std = np.std(result_mtx, axis = 0)[0]
	print("mean test error: "+ str(mean_error) + " ± " + str(error_std))
	print("mean d*: " + "2^"+ str(mean_d_star/2) + " ± " + str(d_star_std))

def part1_6_1():
	iteration = 10#set epochs number 
	result_table = PrettyTable(["train errors ± STD","test errors ± STD"])
	train_result_mtx = np.zeros(shape = (7,20))
	test_result_mtx = np.zeros(shape = (7,20))
	#iterate d from 1 to 7
	for d in range(1,8):
		mistake_train_mtx = np.zeros(shape = (20,iteration))
		mistake_test_mtx = np.zeros(shape = (20,1))
		for i in range(20):
			print("---------------")
			print("d = " + str(d)+", round: "+ str(i))
			#split data set
			data_train ,data_test = train_test_split(data,test_size=0.2)
			x_train = data_train[:,1:]
			y_train = data_train[:,0]
			x_test = data_test[:,1:]
			y_test = data_test[:,0]


			#train perceptron and get train error and test error
			kp = k_class_perceptron_1v1(len(x_train),x_train,y_train,10,d)
			mistake_train = kp.train(iteration)
			mistake_train_mtx[i] = mistake_train
			mistake_test = test_classifiers(x_test,y_test,kp)
			mistake_test_mtx[i] = mistake_test
		mistake_train_mtx = np.sum(mistake_train_mtx, axis=1)
		mistake_test_mtx = np.sum(mistake_test_mtx, axis=1)
		train_result_mtx[d-1] = mistake_train_mtx
		test_result_mtx[d-1] = mistake_test_mtx
	
	#calculate mean and std
	for i in range(len(train_result_mtx)):
		mean_train_error = np.sum(train_result_mtx[i])/20
		train_std = np.std(train_result_mtx[i], ddof = 1)
		mean_test_error = np.sum(test_result_mtx[i])/20
		test_std = np.std(test_result_mtx[i], ddof = 1)
		result_table.add_row([str(mean_train_error) + ' ± ' +str(train_std), str(mean_test_error)+ ' ± ' +str(test_std)])
	print(result_table)

def part1_6_2():
	interation = 5#set training  epochs 
	result_mtx = np.zeros(shape=(20,2))
	#run 20 times 
	for i in range(20):
		#split data set 
		data_train ,data_test = train_test_split(data,test_size=0.2)
		x_train = data_train[:,1:]
		y_train = data_train[:,0]
		x_test = data_test[:,1:]
		y_test = data_test[:,0]
		mistake_list = []
		#iterate d from 1 to 7
		for d in range(1,8):
			print("-----------------------")
			print("now d is: "+ str(d))
			print("-----------------------")
			mistake = k_fold_crossvalidation(5, x_train, y_train, d, interation)
			mistake_list.append(mistake)
		d_star = mistake_list.index(min(mistake_list))+1#get the best d
		#using the best d to train perceptron and get train error and test error
		kp = k_class_perceptron_1v1(len(x_train),x_train,y_train,10,d_star)
		kp.train(interation)
		test_error = test_classifiers(x_test,y_test,kp)
		result_mtx[i][0] = d_star
		result_mtx[i][1] = test_error
	#calculate mean and std
	mean_error = np.sum(result_mtx, axis = 0)[1]/20
	error_std = np.std(result_mtx, axis = 0)[1]
	mean_d_star =  np.sum(result_mtx, axis = 0)[0]/20
	d_star_std = np.std(result_mtx, axis = 0)[0]
	print("mean test error: "+ str(mean_error) + " ± " + str(error_std))
	print("mean d*: " + str(mean_d_star) + " ± " + str(d_star_std))

def part1_7_svc_1():
	result_table = PrettyTable(["train errors ± STD","test errors ± STD"])
	train_result_mtx = np.zeros(shape = (7,20))
	test_result_mtx = np.zeros(shape = (7,20))
	#iterate d from 1 to 7
	for d in range(1,8):
		mistake_train_mtx = np.zeros(shape = (20,1))
		mistake_test_mtx = np.zeros(shape = (20,1))
		for i in range(20):
			print("---------------")
			print("d = " + str(d)+", round: "+ str(i))
			#split data set
			data_train ,data_test = train_test_split(data,test_size=0.2)
			x_train = data_train[:,1:]
			y_train = data_train[:,0]
			x_test = data_test[:,1:]
			y_test = data_test[:,0]


			#train SVC and get train and test error
			pred, train_error, test_error = svc.SVC(x_test, y_test,x_train, y_train, d)

			mistake_train = train_error
			mistake_train_mtx[i] = mistake_train
			mistake_test = test_error
			mistake_test_mtx[i] = mistake_test
			#print(mistake_test_mtx)
		mistake_train_mtx = np.sum(mistake_train_mtx, axis=1)
		mistake_test_mtx = np.sum(mistake_test_mtx, axis=1)
		train_result_mtx[d-1] = mistake_train_mtx
		test_result_mtx[d-1] = mistake_test_mtx
	#print(test_result_mtx)

	for i in range(len(train_result_mtx)):
		mean_train_error = np.sum(train_result_mtx[i])/20
		train_std = np.std(train_result_mtx[i], ddof = 1)
		mean_test_error = np.sum(test_result_mtx[i])/20
		test_std = np.std(test_result_mtx[i], ddof = 1)
		result_table.add_row([str(mean_train_error) + ' ± ' +str(train_std), str(mean_test_error)+ ' ± ' +str(test_std)])
	print(result_table)

def part1_7_svc_2():
	result_mtx = np.zeros(shape=(20,2))
	#run 20 times 
	for i in range(20):
		data_train ,data_test = train_test_split(data,test_size=0.2)
		x_train = data_train[:,1:]
		y_train = data_train[:,0]
		x_test = data_test[:,1:]
		y_test = data_test[:,0]
		mistake_list = []
		#iterate d from 1 to 7
		for d in range(1,8):
			print("-----------------------")
			print("this is round "+ str(i)+". now d is: "+ str(d))
			print("-----------------------")
			mistake = k_fold_crossvalidation_for_svc(5, x_train, y_train, d)
			mistake_list.append(mistake)
		d_star = mistake_list.index(min(mistake_list))+1#find the best d
		#use best d to train svc and get beset test error and train error
		pred, train_error, test_error = svc.SVC(x_test, y_test,x_train, y_train, d)
		test_error = test_error
		result_mtx[i][0] = d_star
		result_mtx[i][1] = test_error
	mean_error = np.sum(result_mtx, axis = 0)[1]/20
	error_std = np.std(result_mtx, axis = 0)[1]
	mean_d_star =  np.sum(result_mtx, axis = 0)[0]/20
	d_star_std = np.std(result_mtx, axis = 0)[0]
	print("-------------")
	print("mean test error: "+ str(mean_error) + " ± " + str(error_std))
	print("mean d*: " + str(mean_d_star) + " ± " + str(d_star_std))

def part1_7_rfc_1():
	result_table = PrettyTable(["train errors ± STD","test errors ± STD"])
	train_result_mtx = np.zeros(shape = (5,20))
	test_result_mtx = np.zeros(shape = (5,20))
	#iterate d from 20 to 101
	for d in range(20,101,20):
		mistake_train_mtx = np.zeros(shape = (20,1))
		mistake_test_mtx = np.zeros(shape = (20,1))
		for i in range(20):
			print("---------------")
			print("d = " + str(d)+", round: "+ str(i))
			#split data set
			data_train ,data_test = train_test_split(data,test_size=0.2)
			x_train = data_train[:,1:]
			y_train = data_train[:,0]
			x_test = data_test[:,1:]
			y_test = data_test[:,0]


			#train random forest classifier
			pred, train_error, test_error = rfc.RFC(x_test, y_test,x_train, y_train, d)

			mistake_train = train_error
			mistake_train_mtx[i] = mistake_train
			mistake_test = test_error
			mistake_test_mtx[i] = mistake_test
			#print(mistake_test_mtx)
		mistake_train_mtx = np.sum(mistake_train_mtx, axis=1)
		mistake_test_mtx = np.sum(mistake_test_mtx, axis=1)
		train_result_mtx[int(d/20-1)] = mistake_train_mtx
		test_result_mtx[int(d/20-1)] = mistake_test_mtx
	#print(test_result_mtx)

	for i in range(len(train_result_mtx)):
		mean_train_error = np.sum(train_result_mtx[i])/20
		train_std = np.std(train_result_mtx[i], ddof = 1)
		mean_test_error = np.sum(test_result_mtx[i])/20
		test_std = np.std(test_result_mtx[i], ddof = 1)
		result_table.add_row([str(mean_train_error) + ' ± ' +str(train_std), str(mean_test_error)+ ' ± ' +str(test_std)])
	print(result_table)

def part1_7_rfc_2():
	result_mtx = np.zeros(shape=(20,2))
	for i in range(20):
		data_train ,data_test = train_test_split(data,test_size=0.2)
		x_train = data_train[:,1:]
		y_train = data_train[:,0]
		x_test = data_test[:,1:]
		y_test = data_test[:,0]
		mistake_list = []
		for d in range(20,101,20):
			print("-----------------------")
			print("this is round "+ str(i)+". now d is: "+ str(d))
			print("-----------------------")
			mistake = k_fold_crossvalidation_for_rfc(5, x_train, y_train, d)
			mistake_list.append(mistake)
		d_star = mistake_list.index(min(mistake_list))+1
		pred, train_error, test_error = rfc.RFC(x_test, y_test,x_train, y_train, d)
		test_error = test_error
		result_mtx[i][0] = d_star
		result_mtx[i][1] = test_error
	mean_error = np.sum(result_mtx, axis = 0)[1]/20
	error_std = np.std(result_mtx, axis = 0)[1]
	mean_d_star =  np.sum(result_mtx, axis = 0)[0]/20
	d_star_std = np.std(result_mtx, axis = 0)[0]
	print("-------------")
	print("mean test error: "+ str(mean_error) + " ± " + str(error_std))
	print("mean d*: " + str(mean_d_star) + " ± " + str(d_star_std))

def k_fold_crossvalidation(k, data_x, data_y, d, iteration, kernel = "poly"):
	#split data into k segments averagely
	kf=KFold(n_splits=k)
	a = kf.split(data_x)
	mistake_list = []
	for train_data_index, test_data_index in a:
		#split data set
		x_train = data_x[train_data_index]
		x_test = data_x[test_data_index]
		y_train = data_y[train_data_index]
		y_test = data_y[test_data_index]
		#perceptron
		kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10,d, kernel)
		kp.train(iteration)
		mistakes = test_classifiers(x_test,y_test,kp)
		mistake_list.append(mistakes)
		print("-------------------")
	sum = 0
	for i in range(len(mistake_list)):
		sum+= mistake_list[i]
	mean_mistake = sum/len(mistake_list)
	return mean_mistake

def k_fold_crossvalidation_for_svc(k, data_x, data_y, d, kernel = "poly"):
	#split data into k segments averagely
	kf=KFold(n_splits=k)
	a = kf.split(data_x)
	mistake_list = []
	for train_data_index, test_data_index in a:
		#split data set
		x_train = data_x[train_data_index]
		x_test = data_x[test_data_index]
		y_train = data_y[train_data_index]
		y_test = data_y[test_data_index]
		#perceptron
		pred, train_error, test_error = svc.SVC(x_test, y_test,x_train, y_train, d)
		mistakes = test_error
		mistake_list.append(mistakes)
	sum = 0
	for i in range(len(mistake_list)):
		sum+= mistake_list[i]
	mean_mistake = sum/len(mistake_list)
	return mean_mistake

def k_fold_crossvalidation_for_rfc(k, data_x, data_y, d):
	#split data into k segments averagely
	kf=KFold(n_splits=k)
	a = kf.split(data_x)
	mistake_list = []
	for train_data_index, test_data_index in a:
		#split data set
		x_train = data_x[train_data_index]
		x_test = data_x[test_data_index]
		y_train = data_y[train_data_index]
		y_test = data_y[test_data_index]
		#perceptron
		pred, train_error, test_error = rfc.RFC(x_test, y_test,x_train, y_train, d)
		mistakes = test_error
		mistake_list.append(mistakes)
	sum = 0
	for i in range(len(mistake_list)):
		sum+= mistake_list[i]
	mean_mistake = sum/len(mistake_list)
	return mean_mistake

def test_classifiers(test_data,test_labels,perceptron):
	mistake_count = 0
	confident_mtx = perceptron.predict_mtx(test_data)
	mistake_count = 0
	#compare test_data and test_lables and get the mistake count
	for i in range(len(confident_mtx)):
		indivadual_confident_list = confident_mtx[i].tolist()
		predict_label = indivadual_confident_list.index(max(indivadual_confident_list))
		if(int(predict_label) != int(test_labels[i])):
			mistake_count += 1
	return mistake_count
	

def test():
	data_train ,data_test = train_test_split(data,test_size=0.2)
	x_train = data_train[:,1:]
	y_train = data_train[:,0]
	x_test = data_test[:,1:]
	y_test = data_test[:,0]
	kp = k_class_perceptron_1vRest(len(x_train),x_train,y_train,10, 5)
	s = kp.train(10)
	for i in range(len(x_train)):
		confident, predict = kp.predict(x_train[i])
		print("-----------------")
		print(str(predict)+" : "+str(y_train[i]))
		print("-----------------")
		if(int(predict) != int(y_train[i])):
			print("error!!!!")




if __name__ == '__main__':
	part1_5_2()