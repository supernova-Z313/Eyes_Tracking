import cv2
import numpy as np

# =============================================================
def draw_rec(img, rec_array, color: (int, int, int)):
	for (x,y,w,h) in rec_array:
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
	# image or so little or so big or under the half of face or not in face deleted
	face_end_x = face_cord[0][0] + face_cord[0][2]
	face_end_y = face_cord[0][1] + face_cord[0][3]
	face_eara = face_cord[0][2] * face_cord[0][3]
	final = []
	for i in eyes_cord:
		eara = i[2]*i[3]
		end_x = i[0] + i[2]
		end_y = i[1] + i[3]

		# ------------------------------------ not in face
		if face_cord[0][0] > start_x:
			continue
		if face_cord[0][1] > start_y:
			continue
		if face_end_x < end_x:
			continue
		if face_end_y < end_y:
			continue

		# ------------------------------------ under the 65%
		if i[1] > (face_cord[0][1] + (face_cord[0][3]*65/100)):
			continue

		# ------------------------------------ so big or little
		if not(2.335 <= (eara*100)/(face_eara) <= 4.111):
			continue



		


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

		frame, face_cord = detecting(frame, face_cascade)
		frame, eyes_cord = detecting(frame, eye_cascade)
		draw_rec(frame, eyes_cord, (0, 255, 0))
		draw_rec(frame, face_cord, (0, 0, 255))

		# better to clearify all and with face only just with face
		if len(face_cord):
			clearify_selection_with_face(face_cord, eyes_cord)
		else:
			# overlap or more than one 
			clearify_selection_with_mo_face()

		# roi = frame[]
		cv2.imshow("video", frame)
		key = cv2.waitKey(1000)
		if key == ord('q'): # quit
			break
		elif key == ord('s'): # stop
			cv2.waitKey(0) # wait until some key pressed

	cv2.destroyAllWindows()
