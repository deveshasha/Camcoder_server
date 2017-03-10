import os
import cv2
import numpy as np
import json
from os.path import dirname, join
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create your views here.


def home(request):
	return HttpResponse("HOME")

@csrf_exempt
def handle_upload(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage(os.path.join(BASE_DIR,'media/images'),'/media/images/')
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        # print fs
        # print filename
        
        srcfile_url = BASE_DIR + uploaded_file_url
        destfile_url = BASE_DIR + '/media/output/'
        filename = preprocess(srcfile_url, destfile_url, filename)
        output_path = os.path.join(BASE_DIR,'media/output/')

        global FILENAME
        FILENAME = filename.replace(".tiff","")
        #print FILENAME

        output_filename = 'code_'
        output_filename += filename.replace(".tiff","")
        output_path += output_filename

        command = 'tesseract '+destfile_url+'preprocessed_'+ filename +' '+ output_path + ' -l eng'
        #print command
        os.system(command)
        
        output_path += '.txt'
        fs2 = FileSystemStorage()
        text_file = fs2.open(output_path)
        text = text_file.read()

        return HttpResponse(text)
    else:
    	return HttpResponse("NOT A POST REQUEST")

def preprocess(srcfile_url, destfile_url, filename):
	#Read image
	image = cv2.imread(srcfile_url)

	#Resize accdn to calculated ratio
	ratio = 720.0 / image.shape[0]
	dim = (int(image.shape[1] * ratio) , 720)

	# perform the actual resizing of the image
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
	image_height = image.shape[0]
	image_width = image.shape[1]

	#convert to grayscale
	img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.bitwise_not(img_gray)
	#cv2.imwrite('1_img_gray.jpg',img_gray)

	#complementing image 2 diff methods
	#im_complement = (255-img_gray)
	#cv2.imwrite('cmpl.jpg',im_complement)

	#calculate background image
	se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
	img_bg = cv2.erode(img_gray,se)
	#cv2.imwrite('2_img_bg.jpg',img_bg)

	#Subtract background from image
	img_fg = img_gray - img_bg
	#cv2.imwrite('3_img_fg.jpg',img_fg)

	#binarize image
	(thresh, img_thresh) = cv2.threshold(img_fg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#cv2.imwrite('4_img_thresh.jpg',img_thresh)

	#Filter blobs by area
	connectivity = 8
	output = cv2.connectedComponentsWithStats(img_thresh, connectivity, cv2.CV_32S)
	num_labels = output[0]
	# labels = output[1]
	stats = output[2]

	for i in range(1,num_labels):
		if(stats[i,cv2.CC_STAT_AREA] < 10):
			left = stats[i,cv2.CC_STAT_LEFT]
			width = stats[i,cv2.CC_STAT_WIDTH]
			top = stats[i,cv2.CC_STAT_TOP]
			height = stats[i,cv2.CC_STAT_HEIGHT]

			for x in range(top,top+height):
				for y in range(left,left+width):
					img_thresh[x][y] = 0

	#cv2.imwrite('5_blob_removed.jpg',img_thresh)

	#Find houghlines in image
	lines = cv2.HoughLines(img_thresh,1,np.pi/180,275)

	######################################################
	# for line in lines:								
	# 	rho, theta = line[0]								
	# 	a = np.cos(theta)									
	# 	b = np.sin(theta)
	# 	x0 = a*rho
	# 	y0 = b*rho
	# 	x1 = int(x0 + 1200*(-b))
	# 	y1 = int(y0 + 1200*(a))
	# 	x2 = int(x0 - 1200*(-b))
	# 	y2 = int(y0 - 1200*(a))
	# 	cv2.line(img_thresh,(x1,y1),(x2,y2),(255,255,0),2)

	# cv2.imwrite('houghlines.jpg',img_thresh)
	######################################################

	if lines is not None:
		angles_sum = 0
		for line in lines:
			rho, theta = line[0]
			angles_sum += theta

		avg_angle = angles_sum/len(lines)
		avg_angle = avg_angle*180/np.pi
		angle = 90 - avg_angle
	else:
		angle = 0
	##############  CREDITS : Adrian Rosebrock from pyimagesearch.com  ######################

	(h, w) = img_thresh.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	img_thresh = cv2.warpAffine(img_thresh, M, (nW, nH))

	#########################################################################################

	#Resize again after rotation
	ratio = 720.0 / img_thresh.shape[0]
	dim = (int(img_thresh.shape[1] * ratio) , 720)
	# perform the actual resizing of the image
	img_thresh = cv2.resize(img_thresh, dim, interpolation = cv2.INTER_CUBIC)

	#cv2.imwrite('6_final_inverted.jpg',img_thresh)
	img_thresh = cv2.bitwise_not(img_thresh)
	filename = filename.replace("jpg","tiff")
	cv2.imwrite(destfile_url + 'preprocessed_'+ filename , img_thresh)
	return filename

@csrf_exempt
def run_code(request):
	if request.method == 'POST':
		media_exe_path = os.path.join(BASE_DIR,'media/exe/')

		c_file = open(media_exe_path + FILENAME + '.c','w+')
		code = request.POST['code']
		c_file.write(code)
		c_file.close()
		#print media_exe_path
		#print FILENAME

		output_text_file = media_exe_path + FILENAME + '.txt'
        #NOT WORKING
		gcc_compile_command = 'gcc -o ' + media_exe_path + FILENAME + ' ' + media_exe_path + FILENAME + '.c'
		#print gcc_compile_command
		os.system(gcc_compile_command)

		gcc_run_command = media_exe_path + FILENAME + ' > ' + output_text_file
		#print gcc_run_command
		os.system(gcc_run_command)

		fs2 = FileSystemStorage()
        text_file = fs2.open(output_text_file)
        text = text_file.read()

	return HttpResponse(text)

def handle_uploaded_file(request):
	return HttpResponse("handle_iplaooodde file")