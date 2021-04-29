#!/usr/local/bin/python
import argparse
import cv2
import numpy as np
import pdb

def closest(dataset,K):   #to find the closest value from the data set
     dataset = np.asarray(dataset) 
     idx = (np.abs(dataset - K)).argmin() 
     return dataset[idx]

def gray_gaussian_SobelFilter(image): #convert to grayscale, gaussian filter and then sobel filter

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #grayscale
    image = cv2.GaussianBlur(image, (3, 3), 0) #gaussian blur

    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(1, size[0] - 1):   #sobel filter calculation
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], y))
    
    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180 #angles
    convolved = convolved.astype('uint8')
    return convolved, angles


def non_maximum_suppression(image, angles): #non max supression
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1): #defining the angles for supression
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                compare = max(image[i - 1, j], image[i + 1, j])
            else:
                compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

    
def double_threshold_hysteresis(image, low, high): #double threshold hysteresis
    weak = 50 #weak pixels
    strong = 250 #strong pixels
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, 0,1,-2, 0, 2, -1, 0, 1))
    dy = np.array((-1,-2,-1,0,0,0,1,2,1))
    size = image.shape
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result

def Canny(image, low, high): #overall canny edge detection function
    image, angles = gray_gaussian_SobelFilter(image)
    image = non_maximum_suppression(image, angles)
    image = double_threshold_hysteresis(image, low, high)
    return image

def checkforcountry(image): #check the currency's country
    x1 = int(len(image)*0.285) #croping the image 
    x2 = int(len(image)*0.5714)
    y1 = int(len(image[0])*0.75)
    y2 = int(len(image[0])*0.86)
    display  = []
    c0 = 0
    temp = image[x1:x2]

    for i in range(len(temp)):
	    temp1 = list(temp[i])
	    temp2 = temp1[y1:y2]

	    for i in range(len(temp2)):  #check conditions
	        if temp2[i] > 100 :
	            c0 +=1
	    temp3 = np.array(temp2)
	    display.append(temp3)
    display = np.array(display)
    final_out = cv2.resize(display,(0,0),fx= 2.75,fy=2.75)

    return c0,final_out

def yenvaluedetection(image):  #yen value detection out of 1000,2000,5000,10000
	dataset_for_yen = [6.828,7.027,6.8936,5.4565] #dataset 

	cannied = Canny(image,20,80) #canny 
	x = cannied
	x1 = 0
	x2 = int(len(x)*0.2) #cropping the required information

	y1 = int(len(x[0])*0.8336)
	y2 = int(len(x[0])*0.98)

	out  = []
	xx = x[x1:x2]

	c0 = 0
	c1 = 0
	for i in range(len(xx)):
	    temp = list(xx[i])
	    temp1 = temp[y1:y2]


	    for i in range(len(temp1)):
	        if temp1[i] == 0 :
	            c0 +=1
	        else:
	            c1+=1
	    temp2 = np.array(temp1)
	    out.append(temp2)
	out = np.array(out)
	final_outt = cv2.resize(out,(0,0),fx= 2.75,fy=2.75)

	check = closest(dataset_for_yen,c0/c1) #comparing weak and dark pixels 


	tempy = dataset_for_yen.index(check)

	return tempy,final_outt,cannied


def rupeevaluedetection(image):
	dataset_for_rupee = [8.946,8.242,5.36,9.2,7.398,7.436] #rupee value detection out of 10,50,100,200,500,2000

	cannied = cv2.Canny(image,0,180)
	x = cannied
	x1 = int(len(x)*0.6164)#cropping the required information

	x2 = int(len(x)*0.822)

	y1 = int(len(x[0])*0.74)
	y2 = int(len(x[0])*0.91)
	out  = []
	xx = x[x1:x2]

	c0 = 0
	c1 = 0
	for i in range(len(xx)):
	    temp = list(xx[i])
	    temp1 = temp[y1:y2]


	    for i in range(len(temp1)):
	        if temp1[i] == 0 :
	            c0 +=1
	        else:
	            c1+=1
	    temp2 = np.array(temp1)
	    out.append(temp2)
	out = np.array(out)
	final_outt = cv2.resize(out,(0,0),fx= 2.75,fy=2.75)

	check = closest(dataset_for_rupee,c0/c1) #comparing weak and dark pixels 
	tempy = dataset_for_rupee.index(check)

	return tempy,final_outt,cannied


def show_yen(index):  #display of the detected currency
	val1 = cv2.imread("display1000yen.jpg")
	val2 = cv2.imread("display2000yen.jpg")
	val3= cv2.imread("display5000yen.jpg")
	val4 = cv2.imread("display10000yen.jpg")
	if index == 0:
		imvalue = val1
	elif index ==1:
		imvalue = val2
	elif index == 2:
		imvalue = val3
	elif index ==3:
		imvalue = val4

	return imvalue


def show_rupee(index): #display of the detected currency
	val1 = cv2.imread("display10rupee.jpg")
	val2 = cv2.imread("display50rupee.jpg")
	val3= cv2.imread("display100rupee.jpg")
	val4 = cv2.imread("display200rupee.jpg")
	val5 = cv2.imread("display500rupee.jpg")
	val6 = cv2.imread("display2000rupee.jpg")

	if index == 0:
		imvalue = val1
	elif index ==1:
		imvalue = val2
	elif index == 2:
		imvalue = val3
	elif index ==3:
		imvalue = val4
	elif index ==4:
		imvalue = val5
	else:
		imvalue = val6
	return imvalue

if __name__ == "__main__":
	# Create the parser
	my_parser = argparse.ArgumentParser(description=' ')

	# Add the arguments
	my_parser.add_argument('--input', action='store', type=str, required=True)

	# Execute the parse_args() method
	args = my_parser.parse_args()

	input_path = args.input
	image = cv2.imread(input_path) #read image
	image = cv2.resize(image,(600,int(600*0.45))) #resize
	#print(image)
	detectcountry = cv2.Canny(image,0,180) 
	#print(detectcountry)


	check,data_image= checkforcountry(detectcountry) #check the country


	if check == 0:
		indexx, extracted_image,canniedd = rupeevaluedetection(image)
		detected_value = show_rupee(indexx)
		data_image = cv2.resize(data_image,(250,270))
		extracted_image =  cv2.resize(extracted_image,(250,270))
		detected_value = cv2.resize(detected_value,(500,270))

		one = np.concatenate((image,detected_value),axis = 1)
		two = np.concatenate((data_image,canniedd,extracted_image),axis = 1)	

		#cv2.imshow("Table 1",two)  #diplay output
		cv2.imwrite("two.jpg", two)

		#cv2.imshow("Table 2",one)
		cv2.imwrite("one.jpg", one)

		cv2.waitKey(0)


	else:
		indexx, extracted_image,canniedd = yenvaluedetection(image)
		detected_value = show_yen(indexx)
		data_image = cv2.resize(data_image,(250,270))
		extracted_image =  cv2.resize(extracted_image,(250,270))
		detected_value = cv2.resize(detected_value,(500,270))

		one = np.concatenate((image,detected_value),axis = 1) #display output
		two = np.concatenate((data_image,canniedd,extracted_image),axis = 1)

		#cv2.imshow("Table 1",two)
		cv2.imwrite("two.jpg", two)
		cv2.imwrite("one.jpg", one)
		#cv2.imshow("Table 2",one)

		cv2.waitKey(0)