# Camcoder_server
Django app to handle backend processing for [Camcoder](https://github.com/deveshasha/Camcoder)

## Execution stages
The application follows client-server architecture.
</br></br>
![Flow](https://i.imgur.com/bPPumT8.jpg)

## Pre-processing 

### 1. Border Detection / Finding largest contour
The first step in preprocessing is the detection of the page/ board border. The user may
click an image which includes backgrounds such as surfaces of tables, which may introduce
unnecessary noise and thus hamper overall efficiency of the system. Border detection helps to
extract the region of interest. The algorithm finds all the continuous contours in the image. The
largest contour from this is chosen, which is the page border. As all the required text will be
written inside the page we can be sure that the largest continuous contour is our page border.</br>
<img src="https://i.imgur.com/WmUnFZq.jpg"  alt = "Border Detection" width = "600"/>

### 2. Rotation for inclined contours
The extracted contour may not be a perfect rectangle. So a minimum bounding rectangle is
fitted by finding the 4 corner points on the contour. The side of this contour-bounding rectangle
may be at an angle with the horizontal. Using such an image for text recognition will give
ambiguous results. Thus if there is an inclination in the detected contour the image is straightened
before further processing. The angle of the contour with the horizontal is calculated and then the
image is rotated accordingly in the opposite direction.</br>
<img src="https://i.imgur.com/Luvjfgc.jpg"  alt = "Minimum bounding rectangle" width = "600"/>
<img src="https://i.imgur.com/sbxy0Hv.jpg"  alt = "Rotated minimum bounding rectangle" width = "600"/>

### 3. Extracting region of interest after rotation
After the image is rotated we now want to extract only that region of the image which has
the page inside it. Thus the rotated image is cropped according to the minimum bounding rectangle
found in the previous step. All the text will definitely be written inside the page thus we can be
sure that the region we get after cropping contains the written text and it is our required region of
interest.</br>
<img src="https://i.imgur.com/Erxrl7K.jpg"  alt = "Cropped image after rotation" width = "600"/>

### 4. Binarization and Noise Removal
The output of the previous step now has the image which is properly oriented with all the
noisy background removed. The standard image processing techniques are now applied to this
image to make it ready for text recognition. These techniques include Binarization (using Otsu‟s
method) and dust and noise removal using erosion, smallest connected component method, etc.</br>
Deskewing text within the page:
The orientation of the page is corrected by applying all the above mentioned techniques.
But it may still be possible that the text within the page is written in a slant way. Thus as a final
step of pre-processing. the inclination of the text block with respect to the page is found. Hough
transform is used to find the orientation of the text block. If the text is written at an angle the image
is rotated accordingly in the opposite direction.</br>
<img src="https://i.imgur.com/seDW757.jpg"  width = "600"/>

Code for above steps is available [here](https://github.com/deveshasha/Camcoder_server/blob/master/upload/views.py) under 
```preprocess_image``` function.

## Processing
Processing of the images was carried out by using [Tesseract OCR](https://github.com/tesseract-ocr/tesseract). Images of handwritten code were used for training Tesseract. The accuracy of Tesseract can be improved by training it to recognize handwritten text. For training, Tesseract requires what is called a “box file”, which is essentially a text file that lists the characters in the training images in order, one per line, with the coordinates of that character's bounding box. Creation and correction of this box file was carried out using [jTessBoxEditor](https://github.com/nguyenq/jTessBoxEditor) (shown below).</br>
<img src="https://i.imgur.com/YyPWJab.jpg"   width = "600"/>

The corrected box files are then sent to Tesseract for training. You can find more about Tesseract training [here](https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract).

## Post-processing
In order to improve the overall accuracy of the system, certain post-processing techniques are applied. The text returned by Tesseract contains mistakes in most case. Before passing it to manual adjustment, we apply some heuristic text-level processing on it to avoid as much manual modification as possible.</br>
The post-processing engine searches for common OCR errors and replaces them with the correct
string. For example, bad printfs such as `prnztf` is replaced with `printf(` and `{nt` is replaced ith `int`. The simple find-and-replace algorithm is effective for resolving many common OCR rrors, but is severely limited because it has a rigid list of problems that it is looking for and does not consider context.</br>
There are too many ways for words to be misinterpreted by Tesseract for find-and-replace to be a comprehensive solution. There are certain strings like `#include <string.h>` that are commonly included in C programs that must be spelled correctly. We go through the OCR output line by line and compute the edit distance between the line and a list of common strings, like `#include <string.h>`. The closest match is then substituted in for the misspelled string. We use the Sequence Matcher library available in python to perform this operation. If the misspelled word is at least 60% similar to the correct word, then it is replaced.</br>
We use regular expressions to find lines that are missing semicolons. In C, not all lines end in a semicolon. We want to skip lines that start with `(for|while|if)` and lines that end with `(;|>|{|})`. The regular expression `“.*?^(?!for|if|while).*? [^;>{}]\s*$”` expresses the above idea. Regular expressions are useful in identifying patterns, but are mathematically limited in their power. For example, there is no regular expression that will tell us if a line has balanced nested parenthesis. Even so, we can still repair some common non-nested parenthesis errors using regular expressions.</br>
After the common syntactical errors are fixed, the code is passed to a GNU compiler which is responsible for compiling and running the code and returning the output. The compiler contains a full parser which is able to determine if parenthesis, brackets, or braces are balanced. If the program does not compile after the above stages of post processing, the system runs the postprocessed code through a post-compilation processing engine. There are many cases where fixing compiler errors require no human intervention. Currently, the
post-compilation processing engine handles two types of errors: expected declaration or statement at end of input and undeclared variables. For the case of the missing bracket, the system simply inserts the missing bracket into the position the compiler complains at. The line number at which the missing bracket is to be inserted is extracted from the error message given by the compiler. For
variables missing a type, the system declares the variable as in integer. A potential problem is that one syntax error in the code often causes cascading errors messages in the compiler output. For this reason, the system only fixes one type of error at a time before compiling again.</br></br>
Below images illustrate the output of post-processing the stage:</br>
<img src="https://i.imgur.com/gmMoa6Z.jpg"   width = "600"/>

## Screenshots from the app
<img src="https://i.imgur.com/mnECFtV.jpg"   width = "600"/></br>
<img src="https://i.imgur.com/sLp0gil.jpg"   width = "600"/>
