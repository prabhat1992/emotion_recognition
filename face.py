import cv2.cv as cv


face_cascade = "haarcascades/haarcascade_frontalface_alt.xml"
debug = False
show_gabor = True

if show_gabor:
	import gabor

count = 1

#gabor filter parameters
kernel_var = 50
gabor_pulsation = 5
gabor_phase = 0
gabor_psi = 90


def main():
	hc = cv.Load(face_cascade)
	cv.NamedWindow("camera", 1)
	capture = cv.CaptureFromCAM(0)
	cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
	cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FORMAT, cv.IPL_DEPTH_32F)



	print "Press ESC to exit"
	print "capture image with keys 1 to 7"
	print "1: Neutral"
	print "2: anger"
	print "3: disgust"
	print "4: fear"
	print "5: happy"
	print "6: sadness"
	print "7: surprise"

	while True:
		img = cv.QueryFrame(capture)
		#img = cv.LoadImage("data/o_happy_24.jpg")
		
		returned = handel_camera_image(img, hc)

		if returned == None:
			pass
		else:
			(img_f, img_r) = returned

			cv.ShowImage("camera", img_f)
			cv.ShowImage("normalized", img_r)

			if show_gabor:
				kernel_var = 50
				gabor_psi = 90

				gabor_pulsation = 2
				gabor_phase = 0
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor1", img_g_mag)

				gabor_pulsation = 4
				gabor_phase = 0
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor2", img_g_mag)

				gabor_pulsation = 6
				gabor_phase = 0
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor3", img_g_mag)

				# ////////////////////////////////////////

				gabor_pulsation = 2
				gabor_phase = 30
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor4", img_g_mag)

				gabor_pulsation = 4
				gabor_phase = 30
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor5", img_g_mag)

				gabor_pulsation = 6
				gabor_phase = 30
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor6", img_g_mag)

				# ////////////////////////////////////////

				gabor_pulsation = 2
				gabor_phase = 60
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor7", img_g_mag)

				gabor_pulsation = 4
				gabor_phase = 60
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor8", img_g_mag)

				gabor_pulsation = 6
				gabor_phase = 60
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor9", img_g_mag)

				# ////////////////////////////////////////

				gabor_pulsation = 2
				gabor_phase = 90
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor10", img_g_mag)

				gabor_pulsation = 4
				gabor_phase = 90
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor11", img_g_mag)

				gabor_pulsation = 6
				gabor_phase = 90
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor12", img_g_mag)

				# ////////////////////////////////////////

				gabor_pulsation = 2
				gabor_phase = 120
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor13", img_g_mag)

				gabor_pulsation = 4
				gabor_phase = 120
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor14", img_g_mag)

				gabor_pulsation = 6
				gabor_phase = 120
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor15", img_g_mag)

				# ////////////////////////////////////////

				gabor_pulsation = 2
				gabor_phase = 150
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor16", img_g_mag)

				gabor_pulsation = 4
				gabor_phase = 150
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor17", img_g_mag)

				gabor_pulsation = 6
				gabor_phase = 150
				(img_g_mag, img_g) = gabor.Process(img_r, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
				cv.ShowImage("Gabor18", img_g_mag)

				# ////////////////////////////////////////

		key_pressed = cv.WaitKey(1)
		# print "key pressed: " + str(key_pressed)

		if (key_pressed == 27):
			break
		elif (key_pressed == 49):
			save_img("neutral", img, img_r)
		elif (key_pressed == 50):
			save_img("anger", img, img_r)
		elif (key_pressed == 51):
			save_img("disgust", img, img_r)
		elif (key_pressed == 52):
			save_img("fear", img, img_r)
		elif (key_pressed == 53):
			save_img("happy", img, img_r)
		elif (key_pressed == 54):
			save_img("sadness", img, img_r)
		elif (key_pressed == 55):
			save_img("surprise", img, img_r)

	cv.DestroyAllWindows()


def save_img(img_type, img_o, img_f):
	subdir = "data/"
	global count
	cv.SaveImage(subdir + "o_" + img_type + "_" + str(count) + ".jpg", img_o)
	cv.SaveImage(subdir + "f_" + img_type + "_" + str(count) + ".jpg", img_f)
	print "saved " + img_type + ": " + str(count)

	count = count + 1
	

#return the 64 to 64 image
def resize_crop_img(img,x,y,w,h):
	dest_size = 128

	img_o = cv.GetSubRect(img, (x,y,w,h))
	img_r = cv.CreateImage((dest_size, dest_size), 8, 1)
	cv.Resize(img_o, img_r)
	return img_r

# img is the image from the camera
# hc is the haar cascade xml loaded into memory it is used to detect faces

# returns a tuple (img_f, img_face) (or None if more than or less than one face is detected in the image)
# 	img_f is the resized, grayscaled input image with a rectangle around the face
# 	img_face is the normalized image of the face which is the output of resize_crop_img function
def handel_camera_image(img, hc):
	#resize it
	img2 = cv.CreateMat(cv.GetSize(img)[1] / 2, cv.GetSize(img)[0] / 2, cv.CV_8UC3)
	cv.Resize(img, img2)

	#convert to grayscale
	img_gray = cv.CreateImage(cv.GetSize(img2), 8, 1)
	cv.CvtColor(img2, img_gray, cv.CV_RGB2GRAY)

	#set the final image
	img_f = img_gray

	#detect faces from it
	objects = cv.HaarDetectObjects(img_f, hc, cv.CreateMemStorage())
	number_of_faces = len(objects)

	if number_of_faces != 1:
		if debug:
			print "Error! Number of detected faces: " + str(number_of_faces)
		return None
	else:
		for (x, y, w, h), n in objects:
			#annotate the image
			cv.Rectangle(img_f, (x,y), (x+w, y+h), 255)
			if debug:
				print "FACE -> h: " + str(h) + ", w: " + str(w) + ", r(w/h): " + str(float(w)/float(h))
			
			#resize to 64 to 64
			img_r = resize_crop_img(img_f, x,y,w,h)
			return (img_f, img_r)


if __name__ == '__main__':
	main()