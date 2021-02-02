# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 00:03:51 2020

@author: Gurkaran
"""


# Generate_Training_Data.py

import sys
import numpy as np
import cv2
import os

AREA_OF_CONTOUR = 100
COLOR_RED = (0, 0, 255)
WIDTH = 20
HEIGHT = 30

def main():
    Training_Chars = cv2.imread("training_alphanumerics.png")            # read in training alphanumerics image

    if Training_Chars is None:                          # if image is not read successfully
        print("Not able to read image from the file \n\n")       
        os.system("pause")                                  
        return                                              # exit the program
    # end if

    gray_Image = cv2.cvtColor(Training_Chars, cv2.COLOR_BGR2GRAY)          # grayscale image
    blurred_Image = cv2.GaussianBlur(gray_Image, (5,5), 0)                        # blurred image

                                                        # filtering grayscale image to black and white
    threshold_Image = cv2.adaptiveThreshold(blurred_Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,  11, 2)                                    

    cv2.imshow("Threshold Image", threshold_Image)      # outputting threshold image

    threshold_Image_Copy = threshold_Image.copy()        # getting a copy of the thresh image so that findContours donot modifies the image

    Contours, Hierarchy = cv2.findContours(threshold_Image_Copy,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    Respnse_Images =  np.empty((0, WIDTH * HEIGHT))

    classifications = []         # declaring classifications list, so that after classifying chars, it will be written to file in the end

                                    # possible  chars and numbers from 0 to 9
    possible_Characters = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for Contours in Contours:                          # for each contour
        if cv2.contourArea(Contours) > AREA_OF_CONTOUR:          # if contour is greater than the minimium value
            [Position_X, Position_Y, W, H] = cv2.boundingRect(Contours)         # taking and breaking bounding rect

                                                # drawing rectangle around each contour
            cv2.rectangle(Training_Chars,(Position_X, Position_Y),(Position_X+W,Position_Y+H),COLOR_RED, 2) 

            ROI_Image = threshold_Image[Position_Y:Position_Y+H, Position_X:Position_X+W]            # cropping character out of threshold image
            ROI_Image_Resized = cv2.resize(ROI_Image, (WIDTH, HEIGHT))     # resizing the image

            cv2.imshow("ROI Image", ROI_Image)                  
            cv2.imshow("ROI Image Resized", ROI_Image_Resized)      
            cv2.imshow("training_chars.png", Training_Chars)      

            Key_Pressed = cv2.waitKey(0)                     # get the keyboard key pressed

            if Key_Pressed == 27:                   # if esc key is pressed
                sys.exit()                    
            elif Key_Pressed in possible_Characters:      
                classifications.append(Key_Pressed)                                            

                Response_Image = ROI_Image_Resized.reshape((1, WIDTH * HEIGHT))  # flattening the image to 1d numpy array in order to write the file later
                Respnse_Images = np.append(Respnse_Images, Response_Image, 0)                    # add current response image numpy array to list of response image numpy arrays

    classify_images = np.array(classifications, np.float32)                  

    classify_images_resized = classify_images.reshape((classify_images.size, 1))   

    print("\n\nTraining  has been been completed successfully !!\n")

    np.savetxt("classify_images.txt", classify_images_resized)           # write classified or grouped images to file
    np.savetxt("response_images.txt", Respnse_Images)          #

    cv2.destroyAllWindows()           

    return

if __name__ == "__main__":
    main()





