import cv2
import numpy as np

# =============================================================
def draw_rec(img, rec_array, color: (int, int, int)):
	for (x,y,w,h) in face_rect:
		cv2.rectangle(img, (x,y), (x+w,y+h), color, 4) # image, start, end, color, thickness
		# break

# =============================================================
def detecting(img, cascade_type):
	face_img = img.copy()
	face_rect = cascade_type.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
	return (face_img, face_rect)


# =============================================================
def resizer(name, x, y):
	# 720 * 1280 : ahmad
	cv2.namedWindow(name, cv2.WINDOW_NORMAL) 
	cv2.resizeWindow(name, x, y)


# =============================================================
def clearify_selection_with_face(face_cord, eyes_cord):
	pass


# =============================================================
if __name__=="__main__":
	face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

	cap = cv2.VideoCapture("./media/ahmad.mp4")
	resizer("video", 360, 640)
	
	while True:
		ret, frame = cap.read()
		if not(ret):
			break

		# image or so little or so big or not in face deleted
		frame, face_cord = detecting(frame, face_cascade)
		frame, eyes_cord = detecting(frame, eye_cascade)
		if len(face_cord):
			clearify_selection_with_face(face_cord, eyes_cord)

		# roi = frame[]
		cv2.imshow("video", frame)
		key = cv2.waitKey(30)
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
