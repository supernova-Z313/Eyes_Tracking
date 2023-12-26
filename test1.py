import cv2
import numpy as np

# =============================================================
def draw_rec(img, rec_array, color: (int, int, int), thickness=4):
	for (x,y,w,h) in rec_array:
		cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness) # image, start, end, color, thickness
		# break

# =============================================================
def detecting(img, cascade_type):
	face_img = img.copy()
	face_rect = cascade_type.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
	return (face_img, face_rect)

# =============================================================
def resizer(name, x, y, flag=cv2.WINDOW_NORMAL):
	cv2.namedWindow(name, flag)
	if flag != cv2.WINDOW_AUTOSIZE:
		cv2.resizeWindow(name, x, y)

# =============================================================
def clearify_selection_with_face(face_cord, eyes_cord):
	"""
		image or so little or so big or 
		under the half of face or not in face deleted
		[ ] overlap or more than one 	
	"""
	face_end_x = face_cord[0][0] + face_cord[0][2]
	face_end_y = face_cord[0][1] + face_cord[0][3]
	face_eara = face_cord[0][2] * face_cord[0][3]
	final = []
	# print(face_cord[0])
	# print("hello -------------------")
	for ind, i in enumerate(eyes_cord):
		if ind >= 2:
			break
		eara = i[2]*i[3]
		end_x = i[0] + i[2]
		end_y = i[1] + i[3]
		# ------------------------------------ not in face
		if face_cord[0][0] > i[0]:
			continue
		if face_cord[0][1] > i[1]:
			continue
		if face_end_x < end_x:
			continue
		if face_end_y < end_y:
			continue
		# ------------------------------------ under the 65%
		if i[1] > (face_cord[0][1] + (face_cord[0][3]*65/100)):
			continue
		# ------------------------------------ so big or little
		# print(i)
		# print()
		if not(2.546 <= (eara*100)/(face_eara) <= 10.458):
			continue
		# ------------------------------------ so big or little
		final.append(i)
	return final

# =============================================================
def clearify_selection_with_no_face():
	pass

# =============================================================
def find_best_contours(temp, temp_threshold):
	"""
	 1) in the edge 2) be rectangle 3) little or big
	"""
	contours, _ = cv2.findContours(temp_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
	rect = 2.3
	for ind, cnt in enumerate(contours, start=0):
		(c_x, x_y, c_w, c_h) = cv2.boundingRect(cnt)
		# ----------------------- rectangle
		if (c_w/c_h > rect) or (c_h/c_w > rect):
			continue
		# ----------------------- 

		# draw_rec(temp, [[c_x, x_y, c_w, c_h]], (255, 0, 0), 1)
		# cv2.drawContours(left, [cnt], -1, (0, 0, 255), 2)
		if ind > 1:
			break

# =============================================================
def left_right_selection(eyes):
	if clean1[0][0] < clean1[1][0]:
		l = clean1[0]
		r = clean1[1]
	else:
		l = clean1[1]
		r = clean1[0]
	return l, r

# =============================================================
def eyes_frame_seter(frame, lor):
	temp = frame[lor[1]: lor[1]+lor[3], lor[0]: lor[0]+lor[2]]
	temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
	temp = cv2.GaussianBlur(temp, (7, 7), 0)
	_, temp_threshold = cv2.threshold(temp, 45, 255, cv2.THRESH_BINARY_INV)
	return temp, temp_threshold

# =============================================================
def set_name_and_position():
	# 720 * 1280 : ahmad : 360, 640
	# 1080 * 1920 : rastegar : 540, 960
	resizer("video", 360, 640)
	resizer("left", 200, 250) # cv2.WINDOW_AUTOSIZE
	resizer("right", 200, 250) # cv2.WINDOW_AUTOSIZE
	resizer("left_threshold", 200, 250)
	resizer("right_threshold", 200, 250)
	cv2.moveWindow("video", 900, 10)
	cv2.moveWindow("left", 150, 100)
	cv2.moveWindow("right", 400, 100)
	cv2.moveWindow("left_threshold", 150, 450)
	cv2.moveWindow("right_threshold", 400, 450)	

# =============================================================
if __name__=="__main__":
	face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

	cap = cv2.VideoCapture("./media/ahmad.mp4")
	set_name_and_position()

	while True:
		ret, frame = cap.read()
		if not(ret):
			break

		frame, face_cord = detecting(frame, face_cascade)
		frame, eyes_cord = detecting(frame, eye_cascade)
		# draw_rec(frame, eyes_cord, (0, 255, 0))
		draw_rec(frame, face_cord, (0, 0, 255))
		clean1 = []

		if len(face_cord):
			clean1 = clearify_selection_with_face(face_cord, eyes_cord)
			# draw_rec(frame, clean1, (255, 0, 0), 1)

		if len(clean1) == 2:
			l, r = left_right_selection(clean1)

			left, left_threshold = eyes_frame_seter(frame, l)
			right, right_threshold = eyes_frame_seter(frame, r)
			
			find_best_contours(left, left_threshold)
			find_best_contours(right, right_threshold)

			cv2.imshow("left_threshold", left_threshold)
			cv2.imshow("right_threshold", right_threshold)
			cv2.imshow("left", left)
			cv2.imshow("right", right)

		cv2.imshow("video", frame)
		key = cv2.waitKey(30)
		if key == ord('q'): # quit
			break
		elif key == ord('s'): # stop
			cv2.waitKey(0) # wait until some key pressed

	cv2.destroyAllWindows()
