import cv2


# =============================================================
def detecting(img, cascade_type):
	face_img = img.copy()
	face_rect = cascade_type.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

	for (x,y,w,h) in face_rect:
		cv2.rectangle(face_img,(x,y),(x+w,y+h),(0, 0, 255), 5) # image, start, end, color, thickness
		# break
	return face_img


# =============================================================
def image_detecting(test_image, model):
	result = detecting(test_image, model)
	while True:
		cv2.imshow('Image Detection',result)
		code = cv2.waitKey(10)
		if code == ord('q'):
			break

	cv2.destroyWindow("Image Detection")

# =============================================================
def camera_detecting():
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		frame = detecting(frame, eye_cascade)
		cv2.imshow('Camera Detection',frame)
		code = cv2.waitKey(10)
		if code == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


# ==============================================================
if __name__=="__main__":
	image_paths = ['./media/face1.png', './media/face2.jpeg', './media/face3.jpg']
	face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')
	test_image = cv2.imread(image_paths[2], 0)

	image_detecting(test_image, face_cascade)
	# camera_detecting()
