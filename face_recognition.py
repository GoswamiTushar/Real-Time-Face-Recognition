import cv2 as cv
import numpy as np 
import os


# -------------------------------------------Eucledian Distance--------------------------------------------------
def distance(v1, v2):
	return np.sqrt(((v1 - v2)**2).sum())

#--------------------------------------------KNN Algorithm-------------------------------------------------------
# Algorithm to Recognise Faces


def KNN(training_data, testing_data, K = 5):
	dist_record = []

	#Checking with whole dataset
	for i in range(training_data.shape[0]):
		ix = training_data[i, :-1] # X values (till last column)
		iy = training_data[i, -1] # Y values (the last column)

		#Compute the eucledian distance
		d = distance(testing_data , ix)

		dist_record.append([d, iy]) #appending distance along with class name

	#Sort based on distance and get top k
	dk = sorted(dist_record, key = lambda x : x[0])[:K]

	
	#Retrieve only the labels
	labels = np.array(dk)[:, -1]


	# Get the frequencies of each label
	output = np.unique(labels, return_counts = True)


	#Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]


##------------------------------------------DATA PREPARATION------------------------------------------------


Face_Data = [] #List to store face data
Data_Path = "./images/" # Path, where to store Face_Data
labels = []
class_id = 0 # Labels for the given file
names = {} # To map id with name



# There could be a lot of npy files, so iterating over them
for fx in os.listdir(Data_Path): 
	
	#if picked file has extension npy
	if fx.endswith(".npy"): 

		#Map between class id and label 
		names[class_id] = fx[:-4]
		
		# Load Data
		data_item = np.load(Data_Path + fx)
		
		#Bunch of face data (could be of different presons as well)
		Face_Data.append(data_item)

		# Create uniques labels for each npy file
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)


Face_Dataset = np.concatenate(Face_Data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1, 1))

# print(Face_Dataset.shape)
# print(face_labels.shape)
trainset = np.concatenate((Face_Dataset, face_labels), axis = 1)


##--------------------------------------------------Testing-------------------------------------------------------
# Capture video for indefinite period of time
cap= cv.VideoCapture(0) 

# haarcascade_frontalface_alt is a pre-build classifier.
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:

	Bool_val, frame = cap.read()

	# Incase if any fault occurs
	if Bool_val == False:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.4, 5)

	for face in faces:
		x,y,w,h = face


		#Get the Region Of Interest

		offset = 10
		face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]
		face_section = cv.resize(face_section, (100, 100))

		face_section = face_section.flatten()

		#Predicted label

		output = KNN(trainset, face_section)
		pred_name = names[int(output)]

		cv.putText(frame, pred_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
		cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 255), 2)
	cv.imshow("Faces", frame)

	key_pressed = cv.waitKey(1) & 0xFF
	# There is no input function so when the key is pressed, it is interpreted as an ASCII value (accepted by waitKey(1))
	# & 0xFF is to convert it into a 8 bit character rather than 64 or 32 bit

	if key_pressed == ord('q'): # ord('q') gives ASCII value of q
		break 

cap.release()
cv.destroyAllWindows()