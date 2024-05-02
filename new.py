# Python program to read image and display image using OpenCV

# # importing OpenCV(cv2) module
# import cv2

# # Save image in set directory
# # Read RGB image
# img = cv2.imread('images/img1.jpg',cv2.IMREAD_COLOR) 

# # Output img with window name as 'image'
# cv2.imshow('image', img) 

# # Maintain output window until
# # user presses a key
# cv2.waitKey(0)	 

# # Destroying present windows on screen
# cv2.destroyAllWindows() 

# ******************************************************************


# *****************************************************************

# Addition of two images

# Python program to illustrate 
# arithmetic operation of 
# addition of two images 
	
# organizing imports 
# import cv2 
# import numpy as np 
	
# # path to input images are specified and 
# # images are loaded with imread command 
# image1 = cv2.imread('images/img1.jpg') 
# image2 = cv2.imread('images/img2.jpg') 

# # cv2.addWeighted is applied over the 
# # image inputs with applied parameters 
# weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0) 

# # the window showing output image 
# # with the weighted sum 
# cv2.imshow('Weighted Image', weightedSum) 

# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 

# ******************************************************************

# ******************************************************************

# Subtraction of two images

# Python program to illustrate 
# arithmetic operation of 
# subtraction of pixels of two images 

# organizing imports 
# import cv2 
# import numpy as np 
	
# # path to input images are specified and 
# # images are loaded with imread command 
# image1 = cv2.imread('images/sub1.jpg') 
# image2 = cv2.imread('images/sub2.jpg') 

# # cv2.subtract is applied over the 
# # image inputs with applied parameters 
# sub = cv2.subtract(image1, image2) 

# # the window showing output image 
# # with the subtracted image 
# cv2.imshow('Subtracted Image', sub) 

# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 


# ************************************************************

# **************************************************************


# Bitwise AND

# Python program to illustrate 
# arithmetic operation of 
# bitwise AND of two images 
	
# organizing imports 
# import cv2 
# import numpy as np 
	
# # path to input images are specified and 
# # images are loaded with imread command 
# img1 = cv2.imread('images/bit1.png') 
# img2 = cv2.imread('images/bit2.png') 

# # cv2.bitwise_and is applied over the 
# # image inputs with applied parameters 
# dest_and = cv2.bitwise_and(img2, img1, mask = None) 

# # the window showing output image 
# # with the Bitwise AND operation 
# # on the input images 
# cv2.imshow('Bitwise And', dest_and) 

# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 



# *****************************************************************

# *****************************************************************

# Bitwise OR


# Python program to illustrate 
# arithmetic operation of 
# bitwise OR of two images 
	
# organizing imports 
# import cv2 
# import numpy as np 
	
# # path to input images are specified and 
# # images are loaded with imread command 
# img1 = cv2.imread('images/bit1.png') 
# img2 = cv2.imread('images/bit2.png') 

# # cv2.bitwise_or is applied over the 
# # image inputs with applied parameters 
# dest_or = cv2.bitwise_or(img2, img1, mask = None) 

# # the window showing output image 
# # with the Bitwise OR operation 
# # on the input images 
# cv2.imshow('Bitwise OR', dest_or) 

# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 

# *****************************************************************

# *****************************************************************


# Bitwise XOR 

# Python program to illustrate 
# arithmetic operation of 
# bitwise XOR of two images 
	
# organizing imports 
# import cv2 
# import numpy as np 
	
# # path to input images are specified and 
# # images are loaded with imread command 
# img1 = cv2.imread('images/bit1.png') 
# img2 = cv2.imread('images/bit2.png') 

# # cv2.bitwise_xor is applied over the 
# # image inputs with applied parameters 
# dest_xor = cv2.bitwise_xor(img1, img2, mask = None) 

# # the window showing output image 
# # with the Bitwise XOR operation 
# # on the input images 
# cv2.imshow('Bitwise XOR', dest_xor) 

# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 


# *****************************************************************

# *****************************************************************

# Bitwise NOT

# Python program to illustrate 
# arithmetic operation of 
# bitwise NOT on input image 
	
# organizing imports 
# import cv2 
# import numpy as np 
	
# # path to input images are specified and 
# #images are loaded with imread command 
# img1 = cv2.imread('images/bit1.png') 
# img2 = cv2.imread('images/bit2.png') 

# # cv2.bitwise_not is applied over the 
# # image input with applied parameters 
# dest_not1 = cv2.bitwise_not(img1, mask = None) 
# dest_not2 = cv2.bitwise_not(img2, mask = None) 

# # the windows showing output image 
# # with the Bitwise NOT operation 
# # on the 1st and 2nd input image 
# cv2.imshow('Bitwise NOT on image 1', dest_not1) 
# cv2.imshow('Bitwise NOT on image 2', dest_not2) 

# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 


# *****************************************************************

# *****************************************************************

# Resizing an image


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# image = cv2.imread(r"images/img1.jpg", 1)
# # Loading the image

# half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
# bigger = cv2.resize(image, (1050, 1610))

# stretch_near = cv2.resize(image, (780, 540), 
# 			interpolation = cv2.INTER_LINEAR)


# Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
# images =[image, half, bigger, stretch_near]
# count = 4

# for i in range(count):
# 	plt.subplot(2, 2, i + 1)
# 	plt.title(Titles[i])
# 	plt.imshow(images[i])

# plt.show()

# ******************************************************************


# ******************************************************************

# Converting an image to greyscale


# # import opencv 
# import cv2 

# # Load the input image 
# image = cv2.imread('images/img1.jpg') 
# cv2.imshow('Original', image) 
# cv2.waitKey(0) 

# # Use the cvtColor() function to grayscale the image 
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# cv2.imshow('Grayscale', gray_image) 
# cv2.waitKey(0) 

# # Window shown waits for any key pressing event 
# cv2.destroyAllWindows()






# *****************************************************************


# *****************************************************************








# Blurring an image[for side option]




# importing libraries 
# import cv2 
# import numpy as np 

# image = cv2.imread('images/img1.jpg') 

# cv2.imshow('Original Image', image) 
# cv2.waitKey(0) 

# # Gaussian Blur 
# Gaussian = cv2.GaussianBlur(image, (7, 7), 0) 
# cv2.imshow('Gaussian Blurring', Gaussian) 
# cv2.waitKey(0) 

# # Median Blur 
# median = cv2.medianBlur(image, 5) 
# cv2.imshow('Median Blurring', median) 
# cv2.waitKey(0) 


# # Bilateral Blur 
# bilateral = cv2.bilateralFilter(image, 9, 75, 75) 
# cv2.imshow('Bilateral Blurring', bilateral) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 


# ************************************************************


# ************************************************************

# To add border in the image[side]

# Python program to explain cv2.copyMakeBorder() method 

# # importing cv2 
# import cv2 

# # path 
# path = r'images/img1.jpg'

# # Reading an image in default mode 
# image = cv2.imread(path) 

# # Window name in which image is displayed 
# window_name = 'Image'

# # Using cv2.copyMakeBorder() method 
# image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0) 

# # Displaying the image 
# cv2.imshow(window_name, image) 

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# *****************************************************************



# *****************************************************************


# To rotate an image[side]


# Import the necessary Libraries
# import cv2
# import matplotlib.pyplot as plt

# # Read image from disk.
# img = cv2.imread('images/img1.jpg')

# # Convert BGR image to RGB
# image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Image rotation parameter
# center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
# angle = 30
# scale = 1

# # getRotationMatrix2D creates a matrix needed for transformation.
# rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# # We want matrix for rotation w.r.t center to 30 degree without scaling.
# rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (img.shape[1], img.shape[0]))

# # Create subplots
# fig, axs = plt.subplots(1, 2, figsize=(7, 4))

# # Plot the original image
# axs[0].imshow(image_rgb)
# axs[0].set_title('Original Image')

# # Plot the Rotated image
# axs[1].imshow(rotated_image)
# axs[1].set_title('Image Rotation')

# # Remove ticks from the subplots
# for ax in axs:
#     ax.set_xticks([])
#     ax.set_yticks([])

# # Display the subplots
# plt.tight_layout()
# plt.show()
