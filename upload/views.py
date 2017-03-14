import os
import sys
import re
import cv2
import numpy as np
import math
import json
import subprocess
from difflib import SequenceMatcher
import string
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
        print filename
        
        srcfile_url = BASE_DIR + uploaded_file_url
        destfile_url = BASE_DIR + '/media/output/'
        filename = preprocess_image(srcfile_url, destfile_url, filename)
        output_path = os.path.join(BASE_DIR,'media/output/')
        #output_path_after_spellcheck = os.path.join(BASE_DIR,'media/output/')
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

        #output_name_after_spellcheck = 'After_spellcheck_' + filename.replace(".tiff","")
        #output_path_after_spellcheck += output_name_after_spellcheck + '.txt'
        #print output_path_after_spellcheck
        #spellcheck(output_path,output_path_after_spellcheck)
        fs2 = FileSystemStorage()
        text_file = fs2.open(output_path)
        #text = text_file.read().splitlines()
        #text = [line for line in text if line.strip() != '']
        source = text_file.read().splitlines()
        final_upload = preprocess_text(source)
        return HttpResponse(final_upload)
    else:
    	return HttpResponse("NOT A POST REQUEST")




def preprocess_text(source):
	#print source
	p = re.compile( """
					.*?									# Consume text
					^(?!for|if|while)		# Does not contain for, if, while
					.*?									# Consume text
					[^;>{}]							# Does not end in semicolon or brackets
					\s*$								# Ends with optional whitespace
					""", re.VERBOSE)

	#fix missing semicolons

	sequences = ["#include <stdio.h>", "#include <string.h>", "#include <stdlib.h>"]
	
	# Given a line, find best possible match, given a list of valid sequences
	

	for line in xrange(len(source)):
		if re.search(p, source[line]) is not None:
			source[line] = source[line] + ";"

		sequences_ratios = []

		for seq_num in xrange(len(sequences)):
			s = SequenceMatcher(None, source[line], sequences[seq_num])
			sequences_ratios.append(s.ratio())
		# Replace a line, if threshold is reached
		if sequences_ratios:
			max_ratio = max(sequences_ratios)
			if max_ratio > 0.6:
				source[line] = sequences[sequences_ratios.index(max_ratio)]


	text = '\n'.join(source)
	return text
	

def preprocess_image(srcfile_url, destfile_url, filename):
	
	####### canny2.py starts #######
	#Read image
	img = cv2.imread(srcfile_url)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(thresh, img_thresh) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# blur = cv2.GaussianBlur(img_thresh,(15,15),0)
	im2, contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Find largest contour
	index = 0
	cnt = contours[0]
	max_perimeter = cv2.arcLength(cnt,True)

	for i in range(1,len(contours)):
		cnt = contours[i]
		perimeter = cv2.arcLength(cnt,True)
		if perimeter > max_perimeter:
			max_perimeter = perimeter
			index = i

	# cont_img = cv2.drawContours(img, contours, index, (0,255,0), 2)
	# cv2.imwrite('cont_img.jpg',cont_img)

	# Minimum bounding rect
	cnt = contours[index]
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)

	# Non-minimum bounding rect
	x,y,w,h = cv2.boundingRect(cnt)
	# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	# sides of Non-min bounding rect
	xleft = x
	yleft = y + h

	xright = x + w
	yright = y + h

	# points are 1234 clockwise from bottom most of minboundrect
	x1 = box[0][0]
	y1 = box[0][1]

	x2 = box[1][0]
	y2 = box[1][1]

	x3 = box[2][0]
	y3 = box[2][1]

	x4 = box[3][0]
	y4 = box[3][1]

	# Find dist of minrect from boundingrect
	ldist = math.sqrt((xleft - x1)**2 + (yleft - y1)**2)
	rdist = math.sqrt((xright - x1)**2 + (yright - y1)**2)  

	"""

	Closer to left -> clockwise
	Closer to right -> anticlock

	"""

	# Point closer to left
	if ldist < rdist:
		angle = abs(np.arctan2(y4 - y1, x4 - x1) * 180.0 / np.pi)
		clockwise = True
	else:
		angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
		clockwise = False
		angle = 180 - angle

	# length and breadth(width) of minboundrect
	rectL = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
	rectW = math.sqrt((x1 - x4)**2 + (y1 - y4)**2)

	if rectW > rectL:
		rectL,rectW = rectW,rectL

	# img_box = cv2.drawContours(img,[box],0,(0,0,255),2)
	# cv2.imwrite('img_box.jpg',img_box)

	if angle != 0:
		# Rotating image accdn to angle
		(ht, wd) = img_thresh.shape[:2]
		(cX, cY) = ((x1 + x3) / 2 , (y1 + y3) / 2)
		if clockwise:
			M = cv2.getRotationMatrix2D((cX, cY), -angle , 1.0)
		else:
			M = cv2.getRotationMatrix2D((cX, cY), angle , 1.0)

		#img_rot = cv2.warpAffine(img, M, (wd, ht),borderValue=(255,255,255))
		img_rot = cv2.warpAffine(img, M, (wd, ht))

		# cv2.imwrite('rotated.jpg',img_rot)


		# Cropping rotated image by numpy slicing
		rowstart = cY - (int)(rectL / 2)
		rowstart += 5

		rowend = cY + (int)(rectL / 2)
		rowend -= 5

		colstart = cX - (int)(rectW / 2)
		colstart += 5

		colend = cX + (int)(rectW / 2)
		colend -= 5

		img_rot = img_rot[rowstart:rowend , colstart:colend]
	else:
		img_rot = img

	image = img_rot
	
	#cv2.imwrite('rotated_cropped.jpg',img_rot)

	####### canny2.py ends #######

	####### hello.py starts #######

	#Resize accdn to calculated ratio
	# ratio = 720.0 / image.shape[0]
	# dim = (int(image.shape[1] * ratio) , 720)

	# # perform the actual resizing of the image
	# resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
	# image_height = image.shape[0]
	# image_width = image.shape[1]

	#convert to grayscale
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
	#lines = cv2.HoughLines(img_thresh,1,np.pi/180,275)

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

	# if lines is not None:
	# 	angles_sum = 0
	# 	for line in lines:
	# 		rho, theta = line[0]
	# 		angles_sum += theta

	# 	avg_angle = angles_sum/len(lines)
	# 	avg_angle = avg_angle*180/np.pi
	# 	angle = 90 - avg_angle
	# else:
	# 	angle = 0
	##############  CREDITS : Adrian Rosebrock from pyimagesearch.com  ######################

	# (h, w) = img_thresh.shape[:2]
	# (cX, cY) = (w // 2, h // 2)

	# # grab the rotation matrix (applying the negative of the
	# # angle to rotate clockwise), then grab the sine and cosine
	# # (i.e., the rotation components of the matrix)
	# M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	# cos = np.abs(M[0, 0])
	# sin = np.abs(M[0, 1])

	# # compute the new bounding dimensions of the image
	# nW = int((h * sin) + (w * cos))
	# nH = int((h * cos) + (w * sin))

	# # adjust the rotation matrix to take into account translation
	# M[0, 2] += (nW / 2) - cX
	# M[1, 2] += (nH / 2) - cY

	# # perform the actual rotation and return the image
	# img_thresh = cv2.warpAffine(img_thresh, M, (nW, nH))

	#########################################################################################

	# #Resize again after rotation
	# ratio = 720.0 / img_thresh.shape[0]
	# dim = (int(img_thresh.shape[1] * ratio) , 720)
	# # perform the actual resizing of the image
	# img_thresh = cv2.resize(img_thresh, dim, interpolation = cv2.INTER_CUBIC)

	#cv2.imwrite('6_final_inverted.jpg',img_thresh)
	img_thresh = cv2.bitwise_not(img_thresh)
	filename = filename.replace("jpg","tiff")
	cv2.imwrite(destfile_url + 'preprocessed_'+ filename , img_thresh)
	return filename

@csrf_exempt
def run_code(request):
	if request.method == 'POST':
		media_exe_path = os.path.join(BASE_DIR,'media/exe/')
		program_filename = media_exe_path + FILENAME + '.c'
		program_filename_after_compiler_feedback = media_exe_path + FILENAME + 'feed.c'
		c_file = open(program_filename,'w+')
		code = request.POST['code']
		c_file.write(code)
		c_file.close()
		#print media_exe_path
		#print FILENAME

		output_text_file = media_exe_path + FILENAME + '.txt'
        #NOT WORKING
		#gcc_compile_command = 'gcc -o ' + media_exe_path + FILENAME + ' ' + media_exe_path + FILENAME + '.c'
		#print gcc_compile_command
		#os.system(gcc_compile_command)
		try:
			output = subprocess.check_output(["gcc", program_filename], stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			print "Compiler errors. Fixing errors..."
			print "----------------------------"
			error = e.output
			compiler_feedback(error,program_filename,program_filename_after_compiler_feedback)

			#Compile your program

			gcc_compile_command = 'gcc -o ' + media_exe_path + FILENAME + 'feed ' + media_exe_path + FILENAME + 'feed.c'
			#print gcc_compile_command
			os.system(gcc_compile_command)

			#Run your program
			gcc_run_command = media_exe_path + FILENAME + 'feed' + ' > ' + output_text_file
			print gcc_run_command
			os.system(gcc_run_command)


		else:
			print "Compiled Successfully."
			print "----------------------------"
			#Compile your program

			gcc_compile_command = 'gcc -o ' + media_exe_path + FILENAME + ' ' + media_exe_path + FILENAME + '.c'
			os.system(gcc_compile_command)

			#Run your program
			gcc_run_command = media_exe_path + FILENAME + ' > ' + output_text_file
			print gcc_run_command
			os.system(gcc_run_command)

		fs2 = FileSystemStorage()
        text_file = fs2.open(output_text_file)
        text = text_file.read()

	return HttpResponse(text)


# def spellcheck(input_path,output_path):
# 	print 'inside spellcheck'
# 	frequent_words = ['#include', '<stdio.h>', 'int', 'main()', 'char', 'char*' ,'return',\
# 				  'printf',  'string', 'main'];

# 	delimiters = ',|;| |\n'
# 	print input_path
# 	print output_path
# 	before = open(input_path)
# 	after = open(output_path, 'w+')
	
# 	for line in before.readlines():
# 		line = line.replace('\r', '')
# 		line = line.replace('\x1c', '\"')
# 		words = re.split(delimiters, line)
# 		words = filter(None, words);

# 		for word in words:
# 			if (len(word) < 11):
# 				if (len(word) < 7):
# 					thresh = 1
# 				else:
# 					thresh = 2
# 				for candidate in frequent_words:	
# 					ed = levenshtein(word, candidate)
# 					if (ed <= thresh and ed > 0):
# 						line = line.replace(word, candidate)
# 						print line
# 		after.write(line)

# def levenshtein(s1, s2):
# 	print 'inside levenshtein'
# 	if len(s1) < len(s2):
# 		return levenshtein(s2, s1)
 
#     # len(s1) >= len(s2)
# 	if len(s2) == 0:
# 		return len(s1)
 
# 	previous_row = xrange(len(s2) + 1)
# 	for i, c1 in enumerate(s1):
# 		current_row = [i + 1]
#         for j, c2 in enumerate(s2):
#             insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
#             deletions = current_row[j] + 1       # than s2
#             substitutions = previous_row[j] + (c1 != c2)
#             current_row.append(min(insertions, deletions, substitutions))
#         previous_row = current_row
 
# 	return previous_row[-1]

def compiler_feedback(error,program_source,output_source):
	print 'compiler_feedback'

	valid_errors = ["expected declaration or statement at end of input",
									"undeclared (first use in this function)",
									"expected '}' before 'else'"]

	fixed = [False, False, False]

	source = open(program_source, 'r',).read().splitlines()
	#print type(source)
	#p = re.compile("^(.*?):(.*?):(.*?):[ ](.*?)$")


	gcc_output = error.splitlines()
	#print gcc_output
	#print gcc_output[1].splitlines()
	#print source
	#Given a gcc output line, identify if it contains an error message

	for line_num in xrange(len(gcc_output)):

		gcc_line = re.search("error: ",gcc_output[line_num])
		if gcc_line is not None:

		# Determine what the error message is
			error_message = gcc_output[line_num][gcc_line.end():]
	        #error_line = int(gcc_line.group(2)) - 1; # gcc lines are 1-indexed
			# Case 0: Missing } after function or loop. Solution: Add }
			get_line_array = gcc_output[line_num].split(":")
			error_line = int(get_line_array[2]) - 1
			if valid_errors[0] in error_message and fixed[0] is False:
				source.insert(error_line, "}")
				fixed[0] = True;
			# Case 1: Undeclared variable. Solution: make it an int
			# Move semicolon fix code here?
			if valid_errors[1] in error_message and fixed[1] is False:
				source[error_line] = "int " + source[error_line]
				fixed[1] = True;
			# Case 2: Missing } before else. Solution: Add one. TODO: Merge w/ Case 1
			if valid_errors[2] in error_message and fixed[2] is False:
				source.insert(error_line, "}");
				fixed[2] = True;
					
		
		source.append("\n")
    
    
	final_answer = '\n'.join(source)

	#Write fixed source file
	with open(output_source, 'wb+') as f:
		f.write(final_answer);


def handle_uploaded_file(request):
	return HttpResponse("handle_iplaooodde file")