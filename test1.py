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
		image or so little or so big or under the half of face or not in face deleted
		TODO : [ ] overlap or more than one 	
	"""
	face_end_x = face_cord[0][0] + face_cord[0][2]
	face_end_y = face_cord[0][1] + face_cord[0][3]
	face_eara = face_cord[0][2] * face_cord[0][3]
	final = []
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
	rect = 2.1 # 2.3
	little = 1.5
	big = 10
	edge_percent = 22
	o_edge_percent = 100 - edge_percent
	f_h, f_w = temp.shape
	ans = []
	# print("\n---------------")
	for ind, cnt in enumerate(contours, start=0):
		(c_x, c_y, c_w, c_h) = cv2.boundingRect(cnt)
		# ----------------------- rectangle
		if (c_w/c_h > rect) or (c_h/c_w > rect):
			continue
		# ----------------------- little or big
		if not(little < (c_w*c_h*100)/(f_h*f_w) < big):
			continue
		# ----------------------- in the edge
		# (c_x < f_w*edge_percent/100) or (c_x+c_w > f_w*o_edge_percent/100)
		if (c_y < f_h*edge_percent/100) or (c_y+c_h > f_h*o_edge_percent/100):
			continue
		# ----------------------- only 3 of biggest (needed to sort)
		# if ind > 2:
		# 	break
		# -----------------------
		ans.append([c_x, c_y, c_w, c_h])
		# draw_rec(temp, [[c_x, c_y, c_w, c_h]], (255, 0, 0), 1)
		# cv2.drawContours(left, [cnt], -1, (0, 0, 255), 2)
	return ans

# =============================================================
def find_center_point(rectangles):
	mids = {}
	for i in rectangles:
		mids[(i[0]+(i[2]/2), i[1]+(i[3]/2))] = i[2]*i[3]
	temp_x = 0
	temp_y = 0
	erea = 0
	for i in mids:
		erea += mids[i]
		temp_x += i[0]*mids[i]
		temp_y += i[1]*mids[i]
	middle_x = temp_x/erea
	middle_y = temp_y/erea
	return (middle_x, middle_y)

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
def directions(m_x, m_y, shape, words):
	m_x, m_y = int(m_x), int(m_y)
	vertical_top_border = 3.4
	vertical_down_border = 2.9
	horizontal_right_border = 9.6
	horizontal_left_border = 8.9
	ans = None
	# ------------------------ first horizontal second vertical
	if m_x < (shape/2)-(shape*horizontal_left_border/100):
		ans = "Left"
	elif m_x > (shape/2)+(shape*horizontal_right_border/100):
		ans = "Right"
	elif m_y < (shape/2)-(shape*vertical_top_border/100):
		ans = "Top"
	elif m_y > (shape/2)+(shape*vertical_down_border/100):
		ans = "Down"
	else:
		ans = "Center"
	# print(words, ":", ans)
	return ans

# =============================================================
def simple_out_direction(last_list):
	if last_list.count("Top") > 2:
		print("Direction is: Top")
	elif last_list.count("Down") > 2:
		print("Direction is: Down")
	elif last_list.count("Left") > 2:
		print("Direction is: Left")
	elif last_list.count("Right") > 2:
		print("Direction is: Right")
	elif last_list.count("Center") > 2:
		print("Direction is: Center")
	else:
		print("Direction is: Direction Not Equal [Default Value Center]")

# =============================================================
def complex_out_direction(last_list):
	# check if first one was center and 1, 2 are equal ignore first
	if (last_list.count("Top") >= 3) or (last_list[1] == last_list[2] == "Top"):
		print("Direction is: Top")
	elif (last_list.count("Down") >= 3)  or (last_list[1] == last_list[2] == "Down"):
		print("Direction is: Down")
	elif (last_list.count("Left") >= 3) or (last_list[1] == last_list[2] == "Left"):
		print("Direction is: Left")
	elif (last_list.count("Right") >= 3) or (last_list[1] == last_list[2] == "Right"):
		print("Direction is: Right")
	elif last_list.count("Center") >= 3:
		print("Direction is: Center")
	else:
		print("Direction is: Direction Not Equal [Default Value Center]")

# =============================================================
def set_name_and_position():
	# 720 * 1280 : ahmad : 360, 640
	# 1080 * 1920 : rastegar : 540, 960
	resizer("video", 360, 640)
	resizer("left", 200, 250) # cv2.WINDOW_AUTOSIZE
	resizer("right", 200, 250) # cv2.WINDOW_AUTOSIZE
	# resizer("left_threshold", 200, 250)
	# resizer("right_threshold", 200, 250)
	cv2.moveWindow("video", 900, 10)
	cv2.moveWindow("left", 150, 100)
	cv2.moveWindow("right", 400, 100)
	# cv2.moveWindow("left_threshold", 150, 450)
	# cv2.moveWindow("right_threshold", 400, 450)

# =============================================================
if __name__=="__main__":
	face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

	cap = cv2.VideoCapture("./media/ahmad.mp4")
	set_name_and_position()
	last_directions = [0, 0, 0, 0, 0]

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
			
			left_ans = find_best_contours(left, left_threshold)
			right_ans = find_best_contours(right, right_threshold)

			if len(left_ans) and len(right_ans):
				l_m_x, l_m_y = find_center_point(left_ans)
				r_m_x, r_m_y = find_center_point(right_ans)

				l_ans = directions(l_m_x, l_m_y, left.shape[0], "left eye")
				# print(left.shape)
				# print(l_m_x, l_m_y)
				r_ans = directions(r_m_x, r_m_y, right.shape[0], "right eye")
				# print(right.shape)
				# print(r_m_x, r_m_y)
				# print("------------------")

				# ---------------------------- check some last direction ...
				# shift register 5tayi
				last_directions.insert(0, last_directions.pop())

				# ---------------------------
				if l_ans == r_ans:
					last_directions[0] = l_ans
					# print("Direction is:", l_ans)
				else:
					last_directions[0] = "DNE"
					# print("Direction is: Center [Direction Not Equal]")
				# ---------------------------

				simple_out_direction(last_directions)
				# complex_out_direction(last_directions)

				cv2.circle(left, (int(l_m_x), int(l_m_y)), 4, (0, 255, 0), 4)
				cv2.circle(right, (int(r_m_x), int(r_m_y)), 4, (0, 255, 0), 4)

			# cv2.imshow("left_threshold", left_threshold)
			# cv2.imshow("right_threshold", right_threshold)
			cv2.imshow("left", left)
			cv2.imshow("right", right)

		cv2.imshow("video", frame)
		key = cv2.waitKey(20)
		if key == ord('q'): # quit
			break
		elif key == ord('s'): # stop
			cv2.waitKey(0) # wait until some key pressed

	cv2.destroyAllWindows()
