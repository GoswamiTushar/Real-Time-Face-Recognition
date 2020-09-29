"""
Read and show video stream, capture images
Detect Faces and show bounding box	
Flatten the largest face image and save in a numpy array
Repeat the above for multiple people to generate training data
"""

# Read a Video Stream from Camera(Frame by Frame)
import cv2 as cv
import numpy as np


# Capture video for indefinite period of time
capture = cv.VideoCapture(0) 

Face_Data = [] #List to store face data
Data_Path = "./images/" # Path, where to store Face_Data


# haarcascade_frontalface_alt is a pre-build classifier.
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip_counter = 0 # To capture every tenth face

file_name = input("Enter the your name")

while True:

	Bool_val, frame = capture.read()
	

	Gray_Frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Grayscale Image

	# Incase if any fault occurs
	if Bool_val == False:
		continue

	# Storing coordinates of recognized faces		
	faces = face_cascade.detectMultiScale(Gray_Frame, 1.4, 5) #1.3 is scaling factor (it will shrink the image by 30%) and 5 is no of neighbours
	
	# Sorting face coordinates based on area to findout closer image
	faces = sorted(faces, key = lambda f : f[2] * f[3], reverse = True) # f[2]*f[3] = area covered

	#print(faces) # Print list of [x, y, w, h]


	# Iteration over list of tuples containing (X_coordinate, Y_coordinate, width, height)
	for (x, y, w, h) in faces:
		cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #Drawing rectangle
		# Extract (Crop Out the required face) : Region of Interest
		offset = 10
		face_section = frame[y - offset : y + h +offset, x - offset : x + w + offset]

		# Resizing face_section
		face_section = cv.resize(face_section, (100,100))

		if skip_counter % 10 == 0:
			Face_Data.append(face_section)
			print(len(Face_Data))
			cv.imshow("Face_section ",face_section)

		skip_counter += 1

		
	cv.imshow("Gotchaa!!!", frame) # A rectangle will be drawn over the frame


	
	#cv.imshow('Gray Image', Gray_Frame)
	
	# Wait for user input to terminate video stream
	key_pressed = cv.waitKey(1) & 0xFF
	# There is no input function so when the key is pressed, it is interpreted as an ASCII value (accepted by waitKey(1))
	# & 0xFF is to convert it into a 8 bit character rather than 64 or 32 bit

	if key_pressed == ord('q'): # ord('q') gives ASCII value of q
		break
Face_Data = np.asarray(Face_Data)
Face_Data = Face_Data.reshape((Face_Data.shape[0], -1))


#Save this data into file system
np.save(Data_Path + file_name + ".npy", Face_Data)

print("Data saved successfully at" + Data_Path + file_name + ".npy")
capture.release()
cv.destroyAllWindows()