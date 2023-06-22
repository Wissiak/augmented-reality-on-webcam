'''
Code base from https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ 
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


fname = 'images/book1-reference'
ext = '.png'
image = cv2.imread(fname + ext)
if fname == 'images/book1-reference':
	pts = np.array([
		(152, 23), # top left
		(740, 25), # top right
		(775, 863), # bottom right
		(144, 861) # bottom left
	])

def do_warp(points):
	warped = four_point_transform(image, points)
	plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
	plt.title("Warped image")
	plt.show()

	cv2.imwrite(f'{fname}-cut{ext}', warped)

try:
    pts
except NameError:
	id = 111
	fig = plt.figure()
	ax = fig.add_subplot(id)
	
	ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.title("Initial Image (right click on 4 edges)")

	coords = np.zeros((4, 2))

	no_points = 0
	def onclick(event):
		ix, iy = event.xdata, event.ydata

		global coords
		global no_points

		coords[no_points] = (ix, iy)
		no_points += 1
		
		if no_points == 4:
			print(coords)
			fig.canvas.mpl_disconnect(cid)
			plt.close()
			do_warp(coords)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
else:
	do_warp(pts)

