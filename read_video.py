# Read a Video Stream from Camera(Frame by Frame)
import cv2 as cv


# Capture video for indefinite period of time
capture = cv.VideoCapture(0) 

while True:

	Bool_val, frame = capture.read()

	Gray_Frame =	cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	BGR_Frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

	# Incase if any fault occurs
	if Bool_val == False:
		continue

	cv.imshow("Frame", frame) #Display 
	cv.imshow('Gray Image', Gray_Frame)
	cv.imshow('BGR_Frame', BGR_Frame)

	
	# Wait for user input to terminate video stream
	key_pressed = cv.waitKey(1) & 0xFF
	# There is no input function so when the key is pressed, it is interpreted as an ASCII value (accepted by waitKey(1))
	# & 0xFF is to convert it into a 8 bit character rather than 64 or 32 bit

	if key_pressed == ord('q'): # ord('q') gives ASCII value of q
		break

cv.release()
cv.destroyAllWindows()