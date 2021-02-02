# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:03:51 2020

@author: Gurkaran
"""

import cv2
import numpy as np
import os
import ProbableChar
import math
import ProbablePlate

COLOR_YELLOW = (0.0, 255.0, 255.0)
COLOR_GREEN = (0.0, 255.0, 0.0)
MINIMUM_WIDTH = 2
MINIMUM_HEIGHT = 8
ASPECT_RATIO_MINIMUM = 0.25
ASPECT_RATIO_MAXIMUM = 1.0
MINIMUM_AREA = 80
MINIMUM_MATCHING_CHARACTERS = 3
MAXIMUM_DIAGONAL_SIZE = 5.0
MAXIMUM_AREA_CHANGE = 0.5
MAXIMUM_WIDTH_CHANGE = 0.8
MAXIMUM_HEIGHT_CHANGE = 0.2
MAXIMUM_ANGLE_BETWEEN_CHARS = 12.0
PADDING_FACTOR_FOR_PLATE_WIDTH = 1.3
PADDING_FACTOR_FOR_PLATE_HEIGHT = 1.5
IMAGE_WIDTH_RESIZED = 20
IMAGE_HEIGHT_RESIZED = 30
MINIMUM_DIAGONAL_SIZE = 0.3

def main():
    ### KNN Algorithm
    knn = cv2.ml.KNearest_create()
    try:
        classify_images = np.loadtxt("classify_images.txt", np.float32)                  # reading trained classifications
        response_images = np.loadtxt("response_images.txt", np.float32)                 # reading trained images
    except:                                                                                 #  error message
        print("Not able to open classify_images.txt or response_images.txt, exiting program\n") 
        os.system("pause")
        return
    
    classify_images = classify_images.reshape((classify_images.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
    knn.setDefaultK(1)                                                             # setting the value of K to 1
    knn.train(response_images, cv2.ml.ROW_SAMPLE, classify_images)           # train KNN object
    
    Original_image  = cv2.imread("LicensePlates/A9.png")              #Can load png and jpg file # opening image of car with license plate
    
    if Original_image is None:                            # if image was not read successfully
        print("\nLicense Plate image not read from file \n\n")  
        os.system("pause")                                 
        return                                              # in case of no license plate image present exit the program
    
    list_Probable_License_Plates = [] 
    height_orig, width_orig, num_Channels = Original_image.shape
    grayscale_Image = np.zeros((height_orig, width_orig, 1), np.uint8)    #Making array for grayscale image
    thresh_Image = np.zeros((height_orig, width_orig, 1), np.uint8)
    cv2.destroyAllWindows()
     
    ### Preprocessing the image starts
    
    HSV_Image = np.zeros((height_orig, width_orig, 3), np.uint8)
    HSV_Image = cv2.cvtColor(Original_image, cv2.COLOR_BGR2HSV)
    Hue_Image, saturation_Image, grayscale_Image = cv2.split(HSV_Image)
    
    height, width = grayscale_Image.shape
    topHat_Image = np.zeros((height, width, 1), np.uint8)
    blackHat_Image = np.zeros((height, width, 1), np.uint8)
    
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat_Image = cv2.morphologyEx(grayscale_Image, cv2.MORPH_TOPHAT, structuringElement)
    blackHat_Image = cv2.morphologyEx(grayscale_Image, cv2.MORPH_BLACKHAT, structuringElement)
    
    grayscale_Image_Plus_TopHat = cv2.add(grayscale_Image, topHat_Image)
    grayscale_Image_Plus_TopHat_Minus_BlackHat = cv2.subtract(grayscale_Image_Plus_TopHat, blackHat_Image)
    
    Blurr_Image = np.zeros((height, width, 1), np.uint8)
    Blurr_Image = cv2.GaussianBlur(grayscale_Image_Plus_TopHat_Minus_BlackHat, (5, 5), 0)
    thresh_Image = cv2.adaptiveThreshold(Blurr_Image, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    ### Preprocessing the image ends
    
    listOfProbableChars = []
    intCountOfProbableChars = 0
    contours = []
    list_Of_ListsOf_ComparableChars = []
    Thresh_image_Copy = thresh_Image.copy()
    contours, hierarchy = cv2.findContours(Thresh_image_Copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # finding all the contours
    height, width = thresh_Image.shape
    
    for i in range(0, len(contours)):                       # for each contour
        
        probableChar = ProbableChar.ProbableChar(contours[i])
        if (probableChar.intboundingRect_cv2_Area > MINIMUM_AREA and
        probableChar.intBoundingRect_cv2_Width > MINIMUM_WIDTH and probableChar.intBoundingRect_cv2_Height > MINIMUM_HEIGHT and
        ASPECT_RATIO_MINIMUM < probableChar.aspect_Ratio and probableChar.aspect_Ratio < ASPECT_RATIO_MAXIMUM):
            intCountOfProbableChars  = intCountOfProbableChars + 1           # increment count of probable chars
            listOfProbableChars.append(probableChar)   
    
    list_Of_ListsOf_ComparableChars = findListOfListsOfComparableChars(listOfProbableChars)  
    
    for ListOfComparableChars in list_Of_ListsOf_ComparableChars:
        probablePlate = ProbablePlate.ProbablePlate()
        ListOfComparableChars.sort(key = lambda ProbableChar: ProbableChar.center_X)        # sort chars from left to right based on x position
        LicPlateCenter_X = (ListOfComparableChars[0].center_X + ListOfComparableChars[len(ListOfComparableChars) - 1].center_X) / 2.0
        LicPlateCenter_Y = (ListOfComparableChars[0].center_Y + ListOfComparableChars[len(ListOfComparableChars) - 1].center_Y) / 2.0
        LicPlateCenter = LicPlateCenter_X, LicPlateCenter_Y

            # calculating plate width and height
        LicPlateWidth = int((ListOfComparableChars[len(ListOfComparableChars) - 1].intBoundingRect_cv2_X + ListOfComparableChars[len(ListOfComparableChars) - 1].intBoundingRect_cv2_Width - ListOfComparableChars[0].intBoundingRect_cv2_X) * PADDING_FACTOR_FOR_PLATE_WIDTH)
        CharHeightsTotal = 0
        for probableChar in ListOfComparableChars:
            CharHeightsTotal = CharHeightsTotal + probableChar.intBoundingRect_cv2_Height
        
        AvgCharHeight = CharHeightsTotal / len(ListOfComparableChars)
        LicPlateHeight = int(AvgCharHeight * PADDING_FACTOR_FOR_PLATE_HEIGHT)
            # calculate correction angle of plate region
        Opposite = ListOfComparableChars[len(ListOfComparableChars) - 1].center_Y - ListOfComparableChars[0].center_Y
        
        position_X = abs(ListOfComparableChars[0].center_X - ListOfComparableChars[len(ListOfComparableChars) - 1].center_X)
        position_Y = abs(ListOfComparableChars[0].center_Y - ListOfComparableChars[len(ListOfComparableChars) - 1].center_Y)
        Hypotenuse = math.sqrt((position_X ** 2) + (position_Y ** 2))
        Angle_In_Rad = math.asin(Opposite / Hypotenuse)
        Angle_In_Deg = Angle_In_Rad * (180.0 / math.pi)    
        
        # packing plate center point, width and height, and angle into rect member variable of plate
        probablePlate.plate_Location = ( tuple(LicPlateCenter), (LicPlateWidth, LicPlateHeight), Angle_In_Deg )
        rotation_Matrix = cv2.getRotationMatrix2D(tuple(LicPlateCenter), Angle_In_Deg, 1.0)
        rotated_image = cv2.warpAffine(Original_image, rotation_Matrix, (width_orig, height_orig)) 
        cropped_image = cv2.getRectSubPix(rotated_image, (LicPlateWidth, LicPlateHeight), tuple(LicPlateCenter))
        probablePlate.plate_Image = cropped_image
        if probablePlate.plate_Image is not None:                     # plate was found
            list_Probable_License_Plates.append(probablePlate)
            
    print("\n" + str(len(list_Probable_License_Plates)) + " probable plates found")
    if len(list_Probable_License_Plates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")   
    else:          
        for probablePlate in list_Probable_License_Plates:
            height, width, num_Channels = (probablePlate.plate_Image).shape
            
            ### Preprocessing the image starts
            
            HSV_Image = np.zeros((height, width, 3), np.uint8)
            HSV_Image = cv2.cvtColor((probablePlate.plate_Image), cv2.COLOR_BGR2HSV)
            Hue_Image, saturation_Image, grayscale_Image = cv2.split(HSV_Image)

            height_grey, width_grey = grayscale_Image.shape
            topHat_Image = np.zeros((height_grey, width_grey, 1), np.uint8)
            blackHat_Image = np.zeros((height_grey, width_grey, 1), np.uint8)

            structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            topHat_Image = cv2.morphologyEx(grayscale_Image, cv2.MORPH_TOPHAT, structuringElement)
            blackHat_Image = cv2.morphologyEx(grayscale_Image, cv2.MORPH_BLACKHAT, structuringElement)

            grayscale_Image_Plus_TopHat = cv2.add(grayscale_Image, topHat_Image)
            grayscale_Image_Plus_TopHat_Minus_BlackHat = cv2.subtract(grayscale_Image_Plus_TopHat, blackHat_Image)

            Blurr_Image = np.zeros((height_grey, width_grey, 1), np.uint8)
            Blurr_Image = cv2.GaussianBlur(grayscale_Image_Plus_TopHat_Minus_BlackHat, (5, 5), 0)
            thresh_Image = cv2.adaptiveThreshold(Blurr_Image, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
            
            ### Preprocessing the image ends
            
            probablePlate.grayscale_Image = grayscale_Image
            probablePlate.thresh_Image = thresh_Image
            
            probablePlate.thresh_Image = cv2.resize(probablePlate.thresh_Image, (0, 0), fx = 1.6, fy = 1.6)
            threshold_Value, probablePlate.thresh_Image = cv2.threshold(probablePlate.thresh_Image, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            listOfProbableCharsInLicensePlate = []
            contours = []
            
            Thresh_image_Copy = (probablePlate.thresh_Image).copy()
            contours, hierarchy = cv2.findContours(Thresh_image_Copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # finding all the contours

            for contour in contours:                        # for each contour
                probableChar = ProbableChar.ProbableChar(contour)
            
                if (probableChar.intboundingRect_cv2_Area > MINIMUM_AREA and probableChar.intBoundingRect_cv2_Width > MINIMUM_WIDTH and probableChar.intBoundingRect_cv2_Height > MINIMUM_HEIGHT and
                ASPECT_RATIO_MINIMUM < probableChar.aspect_Ratio and probableChar.aspect_Ratio < ASPECT_RATIO_MAXIMUM):
                    listOfProbableCharsInLicensePlate.append(probableChar)
            
            list_Of_ListsOf_ComparableCharsInPlate = findListOfListsOfComparableChars(listOfProbableCharsInLicensePlate)
            
            if (len(list_Of_ListsOf_ComparableCharsInPlate) == 0):			# if no groups of matching chars were found in the plate
                probablePlate.character_String = ""
                continue
            for i in range(0, len(list_Of_ListsOf_ComparableCharsInPlate)):                              # within each list of matching chars
                list_Of_ListsOf_ComparableCharsInPlate[i].sort(key = lambda ProbableChar: ProbableChar.center_X)        # sorting chars from left to right  
                
                ListOfComparableCharsWithInnerCharsRemoved = list( list_Of_ListsOf_ComparableCharsInPlate[i])
                for currentChar in list_Of_ListsOf_ComparableCharsInPlate[i]:
                    for nextChar in list_Of_ListsOf_ComparableCharsInPlate[i]:
                        if currentChar != nextChar:        # if current char and next char are not the same char
                                                                                # if current char and next char have center points at aproximately the same location
                            position_X = abs(currentChar.center_X - nextChar.center_X)
                            position_Y = abs(currentChar.center_Y - nextChar.center_Y)
                            
                            if (math.sqrt((position_X ** 2) + (position_Y ** 2))) < (currentChar.diagonal_Size * MINIMUM_DIAGONAL_SIZE):
                                if currentChar.intboundingRect_cv2_Area < nextChar.intboundingRect_cv2_Area:         # if current char is smaller than next char
                                    if currentChar in ListOfComparableCharsWithInnerCharsRemoved:              # if current char was not already removed on a previous iteration
                                        ListOfComparableCharsWithInnerCharsRemoved.remove(currentChar)
                                    else:                                                                       # else if next char is smaller than current char
                                        if nextChar in ListOfComparableCharsWithInnerCharsRemoved:                # if next char was not already removed on a previous iteration
                                            ListOfComparableCharsWithInnerCharsRemoved.remove(nextChar)
                
                list_Of_ListsOf_ComparableCharsInPlate[i] = ListOfComparableCharsWithInnerCharsRemoved
            
            length_Longest_List_Chars = 0
            index_Longest_List_Chars = 0
            for i in range(0, len(list_Of_ListsOf_ComparableCharsInPlate)):
                if len(list_Of_ListsOf_ComparableCharsInPlate[i]) > length_Longest_List_Chars:
                    length_Longest_List_Chars = len(list_Of_ListsOf_ComparableCharsInPlate[i])
                    index_Longest_List_Chars = i

                # suppose that the longest list of matching chars within the plate is the actual list of chars
            longestListOfComparableCharsInPlate = list_Of_ListsOf_ComparableCharsInPlate[index_Longest_List_Chars]
            
            height, width = (probablePlate.thresh_Image).shape
            character_String = ""
            imgThreshColor = np.zeros((height, width, 3), np.uint8)
            longestListOfComparableCharsInPlate.sort(key = lambda ProbableChar: ProbableChar.center_X) 
            cv2.cvtColor((probablePlate.thresh_Image), cv2.COLOR_GRAY2BGR, imgThreshColor)
            for currentChar in longestListOfComparableCharsInPlate:
                 point1 = (currentChar.intBoundingRect_cv2_X, currentChar.intBoundingRect_cv2_Y)
                 point2 = ((currentChar.intBoundingRect_cv2_X + currentChar.intBoundingRect_cv2_Width), (currentChar.intBoundingRect_cv2_Y + currentChar.intBoundingRect_cv2_Height))
                 cv2.rectangle(imgThreshColor, point1, point2, COLOR_GREEN, 2) 
                 ROI_Image = (probablePlate.thresh_Image)[currentChar.intBoundingRect_cv2_Y : currentChar.intBoundingRect_cv2_Y + currentChar.intBoundingRect_cv2_Height,
                                    currentChar.intBoundingRect_cv2_X : currentChar.intBoundingRect_cv2_X + currentChar.intBoundingRect_cv2_Width]
                 try:
                     ROI_Image_Resized = cv2.resize(ROI_Image, (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED))
                 except Exception as e:
                     print(str(e))    
                 modifyROIResized = ROI_Image_Resized.reshape((1, IMAGE_WIDTH_RESIZED * IMAGE_HEIGHT_RESIZED))        # flattening image into 1d numpy array
                 modifyROIResized = np.float32(modifyROIResized)               # converting 1d numpy array of ints to 1d numpy array of floats
                 ret_val, results, neighresp, dist = knn.findNearest(modifyROIResized, k = 1)
                 str_Char = str(chr(int(results[0][0])))            # get character from results
                 character_String = character_String + str_Char                        # append current char to full string
                 
            probablePlate.character_String = character_String
        list_Probable_License_Plates.sort(key = lambda probablePlate: len(probablePlate.character_String), reverse = True)
        license_Plate = list_Probable_License_Plates[0]
        cv2.imshow("image_Plate", license_Plate.plate_Image)
        cv2.imshow("image_Threshold", license_Plate.thresh_Image)
        if len(license_Plate.character_String) == 0:                     # if no chars were found in the plate
            print("\nNo characters are detected\n\n")            
            return
        Rect_Points = cv2.boxPoints(license_Plate.plate_Location)
        cv2.line(Original_image, tuple(Rect_Points[0]), tuple(Rect_Points[1]), COLOR_YELLOW, 2)         # draw 4 yellow lines
        cv2.line(Original_image, tuple(Rect_Points[1]), tuple(Rect_Points[2]), COLOR_YELLOW, 2)
        cv2.line(Original_image, tuple(Rect_Points[2]), tuple(Rect_Points[3]), COLOR_YELLOW, 2)
        cv2.line(Original_image, tuple(Rect_Points[3]), tuple(Rect_Points[0]), COLOR_YELLOW, 2)
        print("\nlicense plate Number read from image = " + license_Plate.character_String + "\n")
        print("----------------------------------------")
        writingInterpretedPlateCharsOnOriginalImage(Original_image, license_Plate)           # write license plate text on the image

        cv2.imshow("Original_image", Original_image)                # re-show scene image

        cv2.imwrite("Original_image.png", Original_image)
        
    cv2.waitKey(0)
    return       
            

def findListOfListsOfComparableChars(listOfProbableChars):
            # with this function, we start off with all the probable chars in one big list
            # the purpose of this function is to re-arrange this list of chars into a list of lists of matching chars,
    list_Of_ListsOf_ComparableChars = []                  

    for probableChar in listOfProbableChars:                       
        ListOfComparableChars = []
        for probablyComparableChar in listOfProbableChars:
            if probablyComparableChar == probableChar:  #To eliminate doubling the character
                continue
            position_X = abs(probableChar.center_X - probablyComparableChar.center_X)
            position_Y = abs(probableChar.center_Y - probablyComparableChar.center_Y)
            distance_Between_Chars = math.sqrt((position_X ** 2) + (position_Y ** 2))
            adjacent = float(abs(probableChar.center_X - probablyComparableChar.center_X))
            opposite = float(abs(probableChar.center_Y - probablyComparableChar.center_Y))
            if adjacent != 0.0:                           # check so that we do not divide by zero 
                Angle_In_Rad = math.atan(opposite / adjacent)      
            else:
                Angle_In_Rad = 1.5708
            Angle_Between_Chars = Angle_In_Rad * (180.0 / math.pi)       # calculating angle in degrees                
            Change_In_Area = float(abs(probablyComparableChar.intboundingRect_cv2_Area - probableChar.intboundingRect_cv2_Area)) / float(probableChar.intboundingRect_cv2_Area)

            Width_Change = float(abs(probablyComparableChar.intBoundingRect_cv2_Width - probableChar.intBoundingRect_cv2_Width)) / float(probableChar.intBoundingRect_cv2_Width)
            Height_Change = float(abs(probablyComparableChar.intBoundingRect_cv2_Height - probableChar.intBoundingRect_cv2_Height)) / float(probableChar.intBoundingRect_cv2_Height)
            
            if (distance_Between_Chars < (probableChar.diagonal_Size * MAXIMUM_DIAGONAL_SIZE) and
            Angle_Between_Chars < MAXIMUM_ANGLE_BETWEEN_CHARS and Change_In_Area < MAXIMUM_AREA_CHANGE and
            Width_Change < MAXIMUM_WIDTH_CHANGE and Height_Change < MAXIMUM_HEIGHT_CHANGE):

                ListOfComparableChars.append(probablyComparableChar)        # if the chars are a match, add the current char to list of comparable chars

        ListOfComparableChars.append(probableChar)                # also add the current char to current list of comparable chars

        if len(ListOfComparableChars) < MINIMUM_MATCHING_CHARACTERS:    
            continue
                                        
        list_Of_ListsOf_ComparableChars.append(ListOfComparableChars)      # so add to our list of lists of matching chars
        listOfProbableCharsWithCurrentComparisonsRemoved = []
                                                # removing the current list of comparable chars to avoid adding using same chars twice,
        listOfProbableCharsWithCurrentComparisonsRemoved = list(set(listOfProbableChars) - set(ListOfComparableChars))

        iterativeListOfListsOfComparableChars = findListOfListsOfComparableChars(listOfProbableCharsWithCurrentComparisonsRemoved)      # recursive call

        for iterativeListOfComparableChars in iterativeListOfListsOfComparableChars:        # for each list of matching chars found by recursive call
            list_Of_ListsOf_ComparableChars.append(iterativeListOfComparableChars)             # add to our original list of lists of matching chars
        break       # exit for

    # end for

    return list_Of_ListsOf_ComparableChars
                     
            
def writingInterpretedPlateCharsOnOriginalImage(Original_image, license_Plate):
    Center_TextArea_X = 0                             # this will be the center of the area the text that will be written
    Center_TextArea_Y = 0

    Text_Lower_Left_X = 0                          # this will be the bottom left of the area that the text that will be written
    Text_Lower_Left_Y = 0

    scene_Height, scene_Width, scene_NumChannels = Original_image.shape
    Licplate_Height, plate_Width, plate_NumChannels = license_Plate.plate_Image.shape

    FontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain font
    FontScale = float(Licplate_Height) / 50.0                    # base font scale on height of plate area
    font_Thickness = int(round(FontScale * 1.5))           # font thickness on font scale

    SizeOfText, baseline = cv2.getTextSize(license_Plate.character_String, FontFace, FontScale, font_Thickness)        # calling cv2.getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (LicPlateCenter_X, LicPlateCenter_Y), (LicPlateWidth, LicPlateHeight), Angle_In_Deg ) = license_Plate.plate_Location

    LicPlateCenter_X = int(LicPlateCenter_X)              # So that center is an integer
    LicPlateCenter_Y = int(LicPlateCenter_Y)

    Center_TextArea_X = int(LicPlateCenter_X)         #So that the horizontal location of the text area is the same as the plate

    if LicPlateCenter_Y < (scene_Height * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        Center_TextArea_Y = int(round(LicPlateCenter_Y)) + int(round(Licplate_Height * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        Center_TextArea_Y = int(round(LicPlateCenter_Y)) - int(round(Licplate_Height * 1.6))      # write the chars in above the plate
    # end if

    WidthOfTextSize, HeightofTextSize = SizeOfText                # unpacking text size width and height

    Text_Lower_Left_X = int(Center_TextArea_X - (WidthOfTextSize / 2))           # calculate the lower left origin of the text area
    Text_Lower_Left_Y = int(Center_TextArea_Y + (HeightofTextSize / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(Original_image, license_Plate.character_String, (Text_Lower_Left_X, Text_Lower_Left_Y), FontFace, FontScale, COLOR_YELLOW, font_Thickness)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    