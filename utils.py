import cv2.cv as cv
import svm
import svmutil
import gabor
import glob
import face
import operator
import os.path
import numpy
from sklearn.lda import LDA
from sklearn.decomposition import RandomizedPCA as PCA

# python ~/repo/libsvm-3.11/tools/grid.py -log2c -5,5,1 -svmtrain "C:\Users\Yasser\repo\libsvm-3.11\windows\svm-train.exe" -gnuplot "C:\Users\Yasser\repo\gnuplot\bin\gnuplot.exe" -v 10 data/anger.data
global pca
pca = {}
global lda
lda = {}
global subdir
subdir = 'data/'

def fit_pca_and_lda(img_kind):
	print img_kind
	global pca
	global lda
	global subdir

	subdir = "data/train/"
	classes = []
	data = []

	the_ones = glob.glob(subdir + "f_" + img_kind + "*.jpg")
	all_of_them = glob.glob(subdir + "f_*_*.jpg")
	the_others = []

	for x in all_of_them:
		if the_ones.count(x) < 1:
			the_others.append(x)

	for x in the_ones:
		classes.append(1)
		data.append(get_image_features(cv.LoadImageM(x)))

	for x in the_others:
		classes.append(-1)
		data.append(get_image_features(cv.LoadImageM(x)))

	# c_pca = PCA(n_components=30)
	# print 'fiting-pca'
	# c_pca.fit(data)

	c_lda = LDA(n_components=2)
	print 'fiting-lda'
	c_lda.fit(data, classes)
	print 'finish'

	# pca[img_kind] = c_pca
	lda[img_kind] = c_lda

def main():
	print 'Starting Main'
	
	img_kinds = ["happy", "anger", "neutral", "surprise"]
	happy_cv_svm_params = "-t 2 -c 0.03125 -g 0.0078125"
	surprise_cv_svm_params = "-t 2 -c 0.03125 -g 0.0078125"
	anger_cv_svm_params = "-t 2 -c 0.03125 -g 0.0078125"
	neutral_cv_svm_params = "-t 2 -c 0.03125 -g 0.0078125"

	svm_params = "-t 0 -c 3"
	print 'fiting all PCAs and LDAs'
	for img_kind in img_kinds:
		fit_pca_and_lda(img_kind)
		print '------------------------'
	
	print '++++++++++++++++++++++++++++'

	train_test()

	# for img_kind in img_kinds:
	#  	print '________' + img_kind + '________'
	#  	print '_________________________________'
	 	
	# 	data_gen(img_kind)
	# 	test_model(img_kind)
	
	#img_kind = "happy"

	#example_make_model(img_kind, happy_cv_svm_params)

	#img_kind = "surprise"
	#example_make_model(img_kind, surprise_cv_svm_params)

	#img_kind = "anger"
	#example_make_model(img_kind, anger_cv_svm_params)

	#img_kind = "neutral"
	#example_make_model(img_kind, neutral_cv_svm_params)
	
	test_model(img_kind)

	live_test()

	#test_model(img_kind)

	#pca_test(img_kinds[0])

	#lda_test(img_kinds[0])

def data_gen(img_kind):
	subdir = "data/"
	extension = '.data'
	file_path = subdir + img_kind + extension
	output_file = open(file_path, 'w+')

	the_ones = glob.glob(subdir + "f_" + img_kind + "*.jpg")
	all_of_them = glob.glob(subdir + "f_*_*.jpg")
	the_others = []

	for x in all_of_them:
		if the_ones.count(x) < 1:
			the_others.append(x)
	
	for x in the_ones:
		img_features = get_image_features(cv.LoadImageM(x), True, img_kind)
		class_label = 1
		#write label in a new line
		output_file.write(str(class_label))
		#write features one by one and increment the index
		for i in xrange(1,len(img_features)):
			output_file.write(' ' + str(i) + ':' + str(img_features[i-1]))
		#write newline
		output_file.write("\n")
		print x
	
	for x in the_others:
		img_features = get_image_features(cv.LoadImageM(x), True, img_kind)
		class_label = -1
		#write label in a new line
		output_file.write(str(class_label))
		#write features one by one and increment the index
		for i in xrange(1,len(img_features)):
			output_file.write(' ' + str(i) + ':' + str(img_features[i-1]))
		#write newline
		output_file.write("\n")
		print x
	
	output_file.close()

def pca_test(img_kind):
	import pylab as pl
	from mpl_toolkits.mplot3d import Axes3D

	subdir = "data/"

	classes = []
	data = []

	the_ones = glob.glob(subdir + "f_" + img_kind + "*.jpg")
	all_of_them = glob.glob(subdir + "f_*_*.jpg")
	the_others = []

	for x in all_of_them:
		if the_ones.count(x) < 1:
			the_others.append(x)
	
	for x in the_ones:
		classes.append(1)
		data.append(get_image_features(cv.LoadImageM(x)))
	
	for x in the_others:
		classes.append(-1)
		data.append(get_image_features(cv.LoadImageM(x)))
	
	pca = PCA(46, whiten=True)
	print 'fiting'
	pca.fit(data)
	print 'transforming'
	X_r = pca.transform(data)
	print '----'

	print X_r.shape

	x0 = [x[0] for x in X_r]
	x1 = [x[1] for x in X_r]

	pl.figure()

	for i in xrange(0,len(x0)):
		if classes[i] == 1:
			pl.scatter(x0[i], x1[i], c = 'r')
		else:
			pl.scatter(x0[i], x1[i], c = 'b')
	

	
	# for c, i, target_name in zip("rg", [1, -1], target_names):
	#     pl.scatter(X_r[classes == i, 0], X_r[classes == i, 1], c=c, label=target_name)
	pl.legend()
	pl.title('PCA of dataset')

	pl.show()

def lda_test(img_kind):
	import pylab as pl
	

	subdir = "data/"

	classes = []
	data = []

	the_ones = glob.glob(subdir + "f_" + img_kind + "*.jpg")
	all_of_them = glob.glob(subdir + "f_*_*.jpg")
	the_others = []

	for x in all_of_them:
		if the_ones.count(x) < 1:
			the_others.append(x)
	
	for x in the_ones:
		classes.append(1)
		data.append(get_image_features(cv.LoadImageM(x)))
	
	for x in the_others:
		classes.append(-1)
		data.append(get_image_features(cv.LoadImageM(x)))
	
	lda = LDA(n_components=2)
	print 'fiting'
	lda.fit(data, classes)
	print 'transforming'
	X_r = lda.transform(data)
	print '----'

	print X_r.shape

	x0 = [x[0] for x in X_r]
	x1 = [x[1] for x in X_r]

	pl.figure()
	for i in xrange(0,len(x0)):
		if classes[i] == 1:
			pl.scatter(x0[i], x1[i], c = 'r')
		else:
			pl.scatter(x0[i], x1[i], c = 'b')
	

	
	# for c, i, target_name in zip("rg", [1, -1], target_names):
	#     pl.scatter(X_r[classes == i, 0], X_r[classes == i, 1], c=c, label=target_name)
	pl.legend()
	pl.title('LDA of dataset')

	pl.show()

def live_test():
	subdir = 'data/'
	img_kinds = ["happy", "anger", "neutral", "surprise"]
	models = {}
	# load all the models
	print "Loading Models"
	for img_kind in img_kinds:
		print 'loading for: ' + img_kind
		models[img_kind] = svmutil.svm_load_model(subdir + img_kind + '.model')
	print "---------------------"

	print "Loading cascade"
	face_cascade = "haarcascades/haarcascade_frontalface_alt.xml"
	hc = cv.Load(face_cascade)
	print "---------------------"

	capture = cv.CaptureFromCAM(0)
	while True:
		img = cv.QueryFrame(capture)
		cv.ShowImage("camera",img)
		key_pressed = cv.WaitKey(50)
		if key_pressed == 27:
			break
		elif key_pressed == 32:
			print '~> KEY PRESSED <~'
			# do face detection
			print 'detecting face'
			returned = face.handel_camera_image(img, hc)
			if returned == None:
				print "No face || more than one face"
				pass
			else:
				(img_o, img_face) = returned
				cv.ShowImage("face",img_face)
				# get features from the face
				results = {}
				for img_kind in img_kinds:
					test_data = get_image_features(img_face, True, img_kind)
					predict_input_data = []
					predict_input_data.append(test_data)

					# do svm query
					(val, val_2, label) = svmutil.svm_predict([1] ,predict_input_data, models[img_kind])
					results[img_kind] = label[0][0]
					print img_kind + str(results[img_kind])

				sorted_results = sorted(results.iteritems(), key=operator.itemgetter(1))
				print sorted_results[len(sorted_results)-1][0]

				print "---------------------"

def test_model(img_kind):
	subdir = "data/"
	model = svmutil.svm_load_model(subdir + img_kind + '.model')
	print "Finished Loading Model"

	total_count = 0
	correct_count = 0
	wrong_count = 0

	
	the_ones = glob.glob(subdir + "f_" + img_kind + "*.jpg")
	all_of_them = glob.glob(subdir + "f_*_*.jpg")
	the_others = []

	for x in all_of_them:
		total_count += 1
		if the_ones.count(x) < 1:
			the_others.append(x)
	
	for x in the_ones:
		img = cv.LoadImageM(x)
		cv.ShowImage("img", img)
		cv.WaitKey(10)
		img_features = get_image_features(img, True, img_kind)
		predict_input_data = []
		predict_input_data.append(img_features)
		(val, val_2, val_3) = svmutil.svm_predict([1], predict_input_data, model)
		if int(val[0]) == 1:
			print 'correct'
			correct_count += 1
		else:
			wrong_count += 1

	for x in the_others:
		img = cv.LoadImageM(x)
		cv.ShowImage("img", img)
		cv.WaitKey(10)
		img_features = get_image_features(img, True, img_kind)
		predict_input_data = []
		predict_input_data.append(img_features)
		(val, val_2, val_3) = svmutil.svm_predict([1], predict_input_data, model)
		if int(val[0]) == -1:
			correct_count += 1
		else:
			wrong_count += 1
	
	print "Total Pictures: " + str(total_count)
	print "Correct: " + str(correct_count)
	print "Wrong: " + str(wrong_count)
	print "Accuracy: " + str(correct_count/float(total_count) * 100) + '%'

# This function trains some models with train set
# The tries to test them with test set
def train_test():
	train_subdir = "data/train/"
	test_subdir = "data/test/"
	img_kinds = ["happy", "anger", "neutral", "surprise"]
	models = {}
	params = "-t 0 -c 3"
	svm_params = {	"happy": params,
					"anger": params,
					"neutral": params,
					"surprise": params}

	#train the models
	print 'BUILDING TRAIN MODELS'
	for img_kind in img_kinds:
		print "\t" + img_kind
		problem = build_problem(img_kind, train_subdir)
		param = svm.svm_parameter(svm_params[img_kind])
		models[img_kind] = svmutil.svm_train(problem, param)
		example_make_model(img_kind, svm_params[img_kind])
	print '================================'

	#for each image in test set let's see what is the answe
	total_count = 0
	correct_count = 0
	wrong_count = 0

	print 'TESTING MODELS'
	for img_kind in img_kinds:
		images = glob.glob(test_subdir + "f_" + img_kind + "*.jpg")
		for image in images:
			print "\t" + image
			image_data = cv.LoadImage(image)
			
			# Let's see what are the results from the models
			results = {}
			for kind in img_kinds:
				test_data = get_image_features(image_data, True, kind)
				predict_input_data = []
				predict_input_data.append(test_data)

				# do svm query
				(val, val_2, label) = svmutil.svm_predict([1] ,predict_input_data, models[kind])
				results[kind] = label[0][0]
			
			sorted_results = sorted(results.iteritems(), key=operator.itemgetter(1))
			result = sorted_results[len(sorted_results)-1][0]

			total_count += 1
			if result == img_kind:
				print 'YES :' + result
				correct_count += 1
			else:
				print 'NO  :' + result
				print sorted_results
				wrong_count += 1
			print '-----------------------'
	print '================================'
	print "Total Pictures: " + str(total_count)
	print "Correct: " + str(correct_count)
	print "Wrong: " + str(wrong_count)
	print "Accuracy: " + str(correct_count/float(total_count) * 100)
	
# img_kind = "happy"
# svm_params = "-t 0 -c 10"
def example_make_model(img_kind, svm_params):
	subdir = "data/"
	problem = build_problem(img_kind)
	print "Prob built"

	param = svm.svm_parameter(svm_params)
	print "Params Set"

	problem_model = svmutil.svm_train(problem, param)
	print "Model built"

	svmutil.svm_save_model(subdir + img_kind + '.model', problem_model)
	print "Done"

# gets an opencv image
# returns all elements of that image as a list
def get_features(img):
	features = [] # The list of features

	for x in xrange(0,img.height):
		for y in xrange(0,img.width):
			features.append(img[x,y])
	
	return features

# gets an opencv image
# computes all of its gabor filters and returns them in a list
def get_image_features(img, reduceP=False, img_kind = None):
	features = []

	kernel_var = 50
	gabor_psi = 90

	gabor_pulsation = 2
	gabor_phase = 0
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 4
	gabor_phase = 0
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 6
	gabor_phase = 0
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))
	
	gabor_pulsation = 2
	gabor_phase = 30
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 4
	gabor_phase = 30
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 6
	gabor_phase = 30
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 2
	gabor_phase = 60
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 4
	gabor_phase = 60
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 6
	gabor_phase = 60
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 2
	gabor_phase = 90
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 4
	gabor_phase = 90
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 6
	gabor_phase = 90
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 2
	gabor_phase = 120
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 4
	gabor_phase = 120
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 6
	gabor_phase = 120
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 2
	gabor_phase = 150
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 4
	gabor_phase = 150
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	gabor_pulsation = 6
	gabor_phase = 150
	(t_img_mag, t_img) = gabor.Process(img, kernel_var, gabor_pulsation, gabor_phase, gabor_psi)
	features.extend(get_features(t_img_mag))

	if not reduceP:
		return features
	if reduceP:
		global pca
		global lda
		data = [features]

		# transformed_pca = pca[img_kind].transform(data)
		transformed_lda = lda[img_kind].transform(data)

		# pca_list = transformed_pca.tolist()[0]
		lda_list = transformed_lda.tolist()[0]
		# pca_list.extend(lda_list)

		return lda_list

def build_problem(img_kind, subdir = "data/"):
	subdir = "data/"

	classes = []
	data = []

	the_ones = glob.glob(subdir + "f_" + img_kind + "*.jpg")
	all_of_them = glob.glob(subdir + "f_*_*.jpg")
	the_others = []

	for x in all_of_them:
		if the_ones.count(x) < 1:
			the_others.append(x)
	
	for x in the_ones:
		classes.append(1)
		data.append(get_image_features(cv.LoadImageM(x), True, img_kind))
	
	for x in the_others:
		classes.append(-1)
		data.append(get_image_features(cv.LoadImageM(x), True, img_kind))

	prob = svm.svm_problem(classes, data)

	return prob

if __name__ == '__main__':
	main()