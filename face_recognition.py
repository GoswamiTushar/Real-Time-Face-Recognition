import cv2 as cv
import numpy as np 
import os


#Eucledian Distance
def distance(v1, v2):
	return np.sqrt(((v1 - v2)**2).sum())


# Algorithm to Recognise Faces
def KNN(training_data, testing_data, K = 5):
	dist_record = []

	#Checking with whole dataset
	for i in range(training_data.shape[0]):
		ix = train[i, :-1] # X values (till last column)
		iy = train[i, -1] # Y values (the last column)

		#Compute the eucledian distance
		d = distance(testing_data , ix)

		dist.append([d, iy]) #appending distance along with class name

	#Sort based on distance and get top k
	dk = sorted(dist, key = lambda x : x[0])[:k]

	
	#Retrieve only the labels
	labels = np.array(dk)[:, -1]


	# Get the frequencies of each label
	output = np.unique(labels, return_counts = True)


	#Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
