# Read a Video Stream from Camera(Frame by Frame)
import cv2 as cv


# Capture video for indefinite period of time
capture = cv.VideoCapture(0) 


# haarcascade_frontalface_alt is a pre-build classifier.
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:

	Bool_val, frame = capture.read()

	Gray_Frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	# Incase if any fault occurs
	if Bool_val == False:
		continue

	# Storing coordinates of recognized faces		
	faces = face_cascade.detectMultiScale(Gray_Frame, 1.3, 5) #1.3 is comression ratio and 5 is no of neighbours

	# Iteration over list of tuples containing (X_coordinate, Y_coordinate, width, height)
	for (x, y, w, h) in faces:
		cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #Drawing rectangle
	

	cv.imshow("Gotchaa!!!", frame) # A rectangle will be drawn over the frame
	
	#cv.imshow('Gray Image', Gray_Frame)
	
	# Wait for user input to terminate video stream
	key_pressed = cv.waitKey(1) & 0xFF
	# There is no input function so when the key is pressed, it is interpreted as an ASCII value (accepted by waitKey(1))
	# & 0xFF is to convert it into a 8 bit character rather than 64 or 32 bit

	if key_pressed == ord('q'): # ord('q') gives ASCII value of q
		break

cv.release()
cv.destroyAllWindows()