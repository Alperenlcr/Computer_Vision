import cv2
import numpy
from matplotlib import pyplot
# # # # # How to Read, Write, Show Images in OpenCV
# OPENCV BGR format
# Syntax: cv2.imread(path, flag)
# Parameters:
# path: A string representing the path of the image to be read.
# flag: It specifies the way in which image should be read. It’s default value is cv2.IMREAD_COLOR
# Return Value: This method returns an image that is loaded from the specified file.
#
# cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected.
# It is the default flag. Alternatively, we can pass integer value 1 for this flag.
# cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode.
# Alternatively, we can pass integer value 0 for this flag.
# cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel.
# Alternatively, we can pass integer value -1 for this flag.
"""
print("Flag = 0")
img0 = cv2.imread("lena.jpg", 0)
print(type(img0))
print(img0)
cv2.imshow("image0", img0)
cv2.waitKey(5555)

print("Flag = 1")
img1 = cv2.imread("lena.jpg", 1)
print(type(img1))
print(img1)
cv2.imshow("image1", img1)
cv2.waitKey(5555)

print("Flag = -1")
imgminus1 = cv2.imread("lena.jpg", -1)
print(type(imgminus1))
print(imgminus1)
cv2.imshow("image-1", imgminus1)
cv2.waitKey(5555)

print("Not exits")
img2 = cv2.imread("not_here.jpg", 0)
print(type(img2))
print(img2)

cv2.destroyAllWindows()

cv2.imwrite("lena_flag:0_copy.jpg", img0)
"""
"""
img = cv2.imread("lena.jpg", 1)
cv2.imshow("image", img)
k = cv2.waitKey(5000)
if k == ord("s"):
    cv2.imwrite("lena_copy.jpg", img)
cv2.destroyAllWindows()
"""
# changing one pixels value
# img[x, y] = [123, 123, 123]
# changing pixels value
# img[x1:x2, y1:y2, channel] = 123
# # # # # How to Read, Write, Show Videos from Camera in OpenCV
"""
cap = cv2.VideoCapture(0)
# 0 means default camera of computer
# we can get data from videos by replacing 0 with video.mp4 or video.avi

# codec --> compressor(sikistirmak), decompressor(sikistirilani acmak)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out_video.mp4", fourcc, 10.0, (640, 480))
# 10.0 --> fps // height and width --> (h, w)

# VideoCaptureProperties
# https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
# like height, weight, fps...
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():  # gives True or False
    ret, frame = cap.read()
    cv2.imshow("frame_normal", frame)

    out.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame_gray", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
"""
# # # # # Draw geometric shapes on images using Python OpenCV
"""
# img = cv2.imread("lena.jpg", 1)
img = numpy.zeros([500, 500, 3], numpy.uint8)
img = cv2.line(img, (100, 0), (0, 100), (255, 255, 0), 5)
img = cv2.arrowedLine(img, (100, 100), (200, 200), (0, 255, 0), 10)
img = cv2.rectangle(img, (0, 400), (500, 500), (0, 255, 255), -1)
img = cv2.rectangle(img, (0, 400), (500, 500), (0, 0, 0), 10)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, "LENA", (150, 450), font, 3, (255, 255, 255), 5, cv2.LINE_8)
img = cv2.circle(img, (267, 269), 20, (255, 0, 0), 3)

cv2.imshow("Geometry", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
"""
# # # # # Setting Camera Parameters in OpenCV Python
"""
cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

print(cap.get(3))
print(cap.get(4))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("video", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
"""
# # # # # Show Date and Time on Videos using OpenCV Python
"""
cap = cv2.VideoCapture(0)
text = "Width : {}    Height : {}".format(cap.get(3), cap.get(4))
textname = "ALPEREN OLCER"

from datetime import date
today = date.today()
textdate = "Today's date:" + str(today)
font = cv2.FONT_HERSHEY_SIMPLEX
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.putText(frame, text, (80, 30), font, 1, (255, 0, 0), 4)
    frame = cv2.putText(frame, textname, (220, 420), font, 1, (255, 255, 255), 4)
    frame = cv2.putText(frame, textdate, (140, 450), font, 1, (0, 255, 0), 4)

    cv2.imshow("texted", frame)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
"""
# # # # # Handle Mouse Events in OpenCV
"""
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        text = "{}, {}".format(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x, y), font, 0.5, (255, 255, 255), 2)
        cv2.imshow("window", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        text = "({}, {}, {})".format(img[x, y, 0], img[x, y, 1], img[x, y, 2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow("window", img)

# img = numpy.zeros([500, 500, 3], numpy.uint8)
img = cv2.imread("lena.jpg")
cv2.imshow("window", img)
cv2.setMouseCallback("window", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
        points.append((x, y))
        if len(points) > 1:
            cv2.line(img, points[-1], points[-2], (0, 255, 0), 2)
        cv2.imshow("window", img)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        blue = img[x, y, 0]
        green = img[x, y, 1]
        red = img[x, y, 2]
        colour = numpy.zeros((100, 100, 3), numpy.uint8)
        colour[:] = [blue, green, red]
        cv2.imshow("colour", colour)
        cv2.waitKey(0)

img = cv2.imread("lena.jpg")
cv2.imshow("window", img)
points = []
cv2.setMouseCallback("window", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# # # # # cv.split, cv.merge, cv.resize, cv.add, cv.addWeighted, ROI
# ROI --> region of interest
# fotografın belirli kismi ile calismak
"""

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
        kare.append((x, y))
        cv2.imshow("window", img)
    elif event == cv2.EVENT_MOUSEWHEEL:
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        kare.append((x, y))
        cv2.imshow("window", img)


def get_square(img):
    cv2.imshow("window", img)
    cv2.setMouseCallback("window", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("messi5.jpg")
print(img.shape)    # satir, sutun, channel
print(img.size)     # pixels count
print(img.dtype)    # data type
print(342*548*3)

kare = []
get_square(img)
x1, y1, x2, y2 = kare[0][0], kare[0][1], kare[1][0], kare[1][1]
print(x1, y1, x2, y2)
# x1, y1, x2, y2 = 180, 240, 230, 290
cv2.destroyAllWindows()
chosen = img[y1:y2, x1:x2]
i = 0
while True:
    i += 60
    cv2.imshow("chosen", img)
    cv2.waitKey(1000)
    # 6 5 4 3 2 1
    try:
        img[y1:y2, x1-i:x2-i] = chosen
    except ValueError:
        break

cv2.imshow("last", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("messi5.jpg")
img2 = cv2.imread("lena.jpg")
img = cv2.resize(img, (500,500))
img2 = cv2.resize(img2, (500,500))
img_added = cv2.add(img, img2)
cv2.imshow("added", img_added)
cv2.waitKey(5000)
cv2.destroyAllWindows()

img_added = cv2.addWeighted(img, .7, img2, .3, 0)
cv2.imshow("added", img_added)
cv2.waitKey(5000)
cv2.destroyAllWindows()
"""
# # # # # Bitwise Operations (bitwise AND, OR, NOT and XOR)
"""
img1 = numpy.zeros((250, 500, 3), numpy.uint8)
img1 = cv2.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)
img2 = numpy.full((250, 500, 3), 255, numpy.uint8)
img2 = cv2.rectangle(img2, (0, 0), (250, 250), (0, 0, 0), -1)
bitAnd = cv2.bitwise_and(img2, img1)
bitOr = cv2.bitwise_or(img2, img1)
bitXor = cv2.bitwise_xor(img1, img2)
bitNot1 = cv2.bitwise_not(img1)
bitNot2 = cv2.bitwise_not(img2)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow('bitAnd', bitAnd)
cv2.imshow('bitOr', bitOr)
cv2.imshow('bitXor', bitXor)
cv2.imshow('bitNot1', bitNot1)
cv2.imshow('bitNot2', bitNot2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# # # # # How to Bind Trackbar To OpenCV
"""
def yazdir(x):
    print(x)

img = numpy.zeros((300, 512, 3), numpy.uint8)
cv2.namedWindow("window")

cv2.createTrackbar("B", "window", 0, 255, yazdir)
cv2.createTrackbar("G", "window", 0, 255, yazdir)
cv2.createTrackbar("R", "window", 0, 255, yazdir)
cv2.createTrackbar("OFF(0)---ON(1)", "window", 0, 1, yazdir)


while True:
    cv2.imshow("window", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:     # 27 is <ESC>
        break
    b = cv2.getTrackbarPos("B", "window")
    g = cv2.getTrackbarPos("G", "window")
    r = cv2.getTrackbarPos("R", "window")
    s = cv2.getTrackbarPos("OFF(0)---ON(1)", "window")

    if s == 1:
        img[:] = [b, g, r]

cv2.destroyAllWindows()

# Create a black image, a window
cv2.namedWindow('image')

cv2.createTrackbar('CP', 'image', 10, 400, yazdir)

switch = 'color/gray'
cv2.createTrackbar(switch, 'image', 0, 1, yazdir)

while True:
    img = cv2.imread('lena.jpg')
    pos = cv2.getTrackbarPos('CP', 'image')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(pos), (50, 150), font, 6, (0, 255, 0), 10)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    s = cv2.getTrackbarPos(switch, 'image')

    if s == 1:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imshow('image', img)

cv2.destroyAllWindows()
"""
# # # # # Object Detection and Object Tracking Using HSV Color Space
# hue renk
# value parliklik degeri
# saturation renk derinligi
"""
def no(x):
    pass

cv2.namedWindow("RENK SEC")
cv2.createTrackbar("l_h", "RENK SEC", 0, 255, no)
cv2.createTrackbar("l_s", "RENK SEC", 0, 255, no)
cv2.createTrackbar("l_v", "RENK SEC", 0, 255, no)

cv2.createTrackbar("u_h", "RENK SEC", 0, 255, no)
cv2.createTrackbar("u_s", "RENK SEC", 0, 255, no)
cv2.createTrackbar("u_v", "RENK SEC", 0, 255, no)

while True:
    frame = cv2.imread("smarties.png")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("l_h", "RENK SEC")
    l_s = cv2.getTrackbarPos("l_s", "RENK SEC")
    l_v = cv2.getTrackbarPos("l_v", "RENK SEC")

    u_h = cv2.getTrackbarPos("u_h", "RENK SEC")
    u_s = cv2.getTrackbarPos("u_s", "RENK SEC")
    u_v = cv2.getTrackbarPos("u_v", "RENK SEC")

    l_b = numpy.array([l_h, l_s, l_v])
    u_b = numpy.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow("l", l_b)
    # cv2.imshow("u", u_b)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
"""
"""
def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("US", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 0, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = numpy.array([l_h, l_s, l_v])
    u_b = numpy.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""
# # # # # Simple Image Thresholding
"""
img = cv2.imread('messi5.jpg', 0)
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, th3 = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow("Image", img)
cv2.imshow("th1", th1)
cv2.imshow("th2", th2)
cv2.imshow("th3", th3)
cv2.imshow("th4", th4)
cv2.imshow("th5", th5)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# # # # # Adaptive Thresholding
# Adaptive Thresholding algorithm
# (i) Divide image into strips
# (ii) Apply global threshold method to each strip.
"""
img = cv2.imread('sudoku.png', 0)
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

cv2.imshow("Image", img)
cv2.imshow("THRESH_BINARY", th1)
cv2.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# # # # # matplotlib with OpenCV
# opencv reads BGR
# matplotlib reads RBG
"""
img = cv2.imread("lena.jpg")

pyplot.imshow(img)
pyplot.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pyplot.imshow(img)
pyplot.show()
"""
"""
img = cv2.imread('gradient.png', 0)
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, th3 = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

images = [img, th1, th2, th3, th4, th5]
titles = ["original", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]

for i in range(6):
    pyplot.subplot(2, 3, i+1), pyplot.imshow(images[i], "gray")
    pyplot.title(titles[i])
    pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
"""
# # # # # Morphological Transformations
# Morphological transformations are some simple operations based on the image shape.
# It is normally performed on binary images.
"""
img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel = numpy.ones((5, 5), numpy.uint8)

dilation = cv2.dilate(mask, kernel, iterations=2)
# 1 olanlari kernel ile carpar genisletir
erosion = cv2.erode(mask, kernel, iterations=1)
# ufaltma
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# Opening is just another name of erosion followed by dilation. It is useful in removing noise.
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# Dilation followed by Erosion.
# It is useful in closing small holes inside the foreground objects, or small black points on the object.
Morphological_Gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
# It is the difference between dilation and erosion of an image.
# The result will look like the outline of the object.
Top_Hat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
# It is the difference between input image and Opening of the image.
Black_Hat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
# It is the difference between the closing of the input image and input image.
titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'Morph_Gradient', 'Top_Hat', "Black_Hat"]
images = [img, mask, dilation, erosion, opening, closing, Morphological_Gradient, Top_Hat, Black_Hat]

for i in range(len(images)):
    pyplot.subplot(3, 3, i+1), pyplot.imshow(images[i], 'gray')
    pyplot.title(titles[i])
    pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
"""
# # # # # Smoothing Images | Blurring Images OpenCV
# As in one-dimensional signals, images also can be filtered with various low-pass filters (LPF),
# high-pass filters (HPF), etc. LPF helps in removing noise, blurring images, etc.
# HPF filters help in finding edges in images.
"""
foto = ["gradient.png", "lena.jpg", "messi5.jpg", "sudoku.png", "smarties.png"]
for k in range(len(foto)):

    img = cv2.imread(foto[k])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernel = numpy.ones((5, 5), numpy.float32)/25
    dst = cv2.filter2D(img, -1, kernel)
    blur = cv2.blur(img, (5, 5))
    GaussianBlur = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
# takes the median of all the pixels under the kernel area and the central element is replaced with this median value.
    # This is highly effective against salt-and-pepper noise in an image.
    bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)
    # highly effective in noise removal while keeping edges sharp.

    titles = ['image', '2D Convolution', 'blur', 'GaussianBlur', 'median', 'bilateralFilter']
    images = [img, dst, blur, GaussianBlur, median, bilateralFilter]

    for i in range(6):
        pyplot.subplot(2, 3, i+1), pyplot.imshow(images[i], 'gray')
        pyplot.title(titles[i])
        pyplot.xticks([]), pyplot.yticks([])

    pyplot.show()
"""
# # # # # Image Gradients and Edge Detection
# https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
"""
img = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

# olusturma
Laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# edges hepsini gosteriyor
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
# edges belirtilen yonlerde gosteriyor

# donusturme datatype
Laplacian = numpy.uint8(numpy.absolute(Laplacian))
sobelX = numpy.uint8(numpy.absolute(sobelX))
sobelY = numpy.uint8(numpy.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined']
images = [img, Laplacian, sobelX, sobelY, sobelCombined]
for i in range(5):
    pyplot.subplot(2, 3, i+1), pyplot.imshow(images[i], 'gray')
    pyplot.title(titles[i])
    pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
"""
# # # # # Canny Edge Detection in OpenCV
# algorithm composed of these
# Noise reduction;
# Gradient calculation;
# Non-maximum suppression;
# Double threshold;
# Edge Tracking by Hysteresis.
"""
img = cv2.imread("lena.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

canny = cv2.Canny(img, 100, 200)

titles = ['image', 'canny']
images = [img, canny]
for i in range(2):
    pyplot.subplot(1, 2, i+1), pyplot.imshow(images[i], 'gray')
    pyplot.title(titles[i])
    pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
"""
# # # # # Image Pyramids with Python and OpenCV
# There are two kinds of Image Pyramids. 1) Gaussian Pyramid and 2) Laplacian Pyramids
"""
img = cv2.imread("lena.jpg")
downsized1 = cv2.pyrDown(img)
downsized2 = cv2.pyrDown(downsized1)
alfa = cv2.pyrUp(downsized2)
beta = cv2.pyrUp(img)

cv2.imshow("original", img)
cv2.imshow("downsized1", downsized1)
cv2.imshow("downsized2", downsized2)
cv2.imshow("alfa", alfa)
cv2.imshow("beta", beta)



cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# https://docs.opencv.org/3.4/dc/dff/tutorial_py_pyramids.html
# Laplacian pyramid images are like edge images only.
"""
img = cv2.imread("lena.jpg")
layer = img.copy()
gaussian_pyramid_list = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gaussian_pyramid_list.append(layer)
    cv2.imshow(str(i), layer)

layer = gaussian_pyramid_list[5]
cv2.imshow('upper level Gaussian Pyramid', layer)
laplacian_pyramid_list = [layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gaussian_pyramid_list[i])
    laplacian = cv2.subtract(gaussian_pyramid_list[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# # # # # Image Blending using Pyramids in OpenCV
# sasirtici bir sekilde guzel sonuc veriyor
"""
apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')
print(apple.shape)
print(orange.shape)
apple_orange = numpy.hstack((apple[:, :256], orange[:, 256:]))

# generate Gaussian pyramid for apple and orange
# laplacian da kullanmak icin
apple_copy, orange_copy = apple.copy(), orange.copy()
gp_apple, gp_orange = [apple_copy], [orange_copy]
for i in range(6):
    apple_copy, orange_copy = cv2.pyrDown(apple_copy), cv2.pyrDown(orange_copy)
    gp_apple.append(apple_copy)
    gp_orange.append(orange_copy)

# generate Laplacian Pyramid for apple, orange
# 5 kere kucultulmus laplacian iki foto icin
apple_copy, orange_copy = gp_apple[5], gp_orange[5]
lp_apple, lp_orange = [apple_copy], [orange_copy]
for i in range(5, 0, -1):
    gaussian_expanded, gaussian_expanded2 = cv2.pyrUp(gp_apple[i]), cv2.pyrUp(gp_orange[i])
    laplacian, laplacian2 = cv2.subtract(gp_apple[i-1], gaussian_expanded), cv2.subtract(gp_orange[i - 1], gaussian_expanded2)
    lp_apple.append(laplacian)
    lp_orange.append(laplacian2)

# Now add left and right halves of images in each level
# laplacianlari yari yari ekleme
apple_orange_pyramid = []
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    cols, rows, ch = apple_lap.shape
    laplacian = numpy.hstack((apple_lap[:, 0:cols//2], orange_lap[:, cols//2:]))
    apple_orange_pyramid.append(laplacian)

# now reconstruct
apple_orange_reconstruct = apple_orange_pyramid[0]
for i in range(1, 6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)
    # daha iyi goruntusu olanla kotu olani ekleme

easyWay = cv2.pyrDown(apple_orange)
easyWay = cv2.pyrDown(easyWay)
easyWay = cv2.pyrUp(easyWay)
easyWay = cv2.pyrUp(easyWay)

cv2.imshow("apple", apple)
cv2.imshow("orange", orange)
cv2.imshow("mine", easyWay)
cv2.imshow("apple_orange", apple_orange)
cv2.imshow("apple_orange_reconstruct", apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

