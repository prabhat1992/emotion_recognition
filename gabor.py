import cv2.cv as cv
import math

src = 0
src_f = 0
image = 0
dest = 0
dest_mag = 0
# kernelimg=0
# big_kernelimg=0
kernel=0

kernel_size =21
pos_var = 50
pos_w = 5
pos_phase = 0
pos_psi = 90

# returns (dest_mag, dest)
def Process(image, pos_var, pos_w, pos_phase, pos_psi):
	global kernel_size
	if kernel_size%2==0:
		kernel_size += 1

	kernel = cv.CreateMat(kernel_size,kernel_size,cv.CV_32FC1)
	# kernelimg = cv.CreateImage((kernel_size,kernel_size),cv.IPL_DEPTH_32F,1)
	# big_kernelimg = cv.CreateImage((kernel_size*20,kernel_size*20),cv.IPL_DEPTH_32F,1)
	src = cv.CreateImage((image.width,image.height),cv.IPL_DEPTH_8U,1)
	src_f = cv.CreateImage((image.width,image.height),cv.IPL_DEPTH_32F,1)

	# src = image #cv.CvtColor(image,src,cv.CV_BGR2GRAY) #no conversion is needed
	if cv.GetElemType(image) == cv.CV_8UC3:
		cv.CvtColor(image,src,cv.CV_BGR2GRAY)
	else:
		src = image
	

	cv.ConvertScale(src,src_f,1.0/255,0)
	dest = cv.CloneImage(src_f)
	dest_mag = cv.CloneImage(src_f)

	var = pos_var/10.0
	w = pos_w/10.0
	phase = pos_phase*cv.CV_PI/180.0
	psi = cv.CV_PI*pos_psi/180.0

	cv.Zero(kernel)
	for x in range(-kernel_size/2+1,kernel_size/2+1):
		for y in range(-kernel_size/2+1,kernel_size/2+1):
			kernel_val = math.exp( -((x*x)+(y*y))/(2*var))*math.cos( w*x*math.cos(phase)+w*y*math.sin(phase)+psi)
			cv.Set2D(kernel,y+kernel_size/2,x+kernel_size/2,cv.Scalar(kernel_val))
			# cv.Set2D(kernelimg,y+kernel_size/2,x+kernel_size/2,cv.Scalar(kernel_val/2+0.5))
	cv.Filter2D(src_f, dest,kernel,(-1,-1))
	# cv.Resize(kernelimg,big_kernelimg)
	cv.Pow(dest,dest_mag,2)


	# return (dest_mag, big_kernelimg, dest)
	return (dest_mag, dest)
	# cv.ShowImage("Mag",dest_mag)
	# cv.ShowImage("Kernel",big_kernelimg)
	# cv.ShowImage("Process window",dest)