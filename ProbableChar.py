# ProbableChar.py

import cv2
import numpy as np
import math

###################################################################################################
class ProbableChar:

    # constructor #################################################################################
    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect_cv2= cv2.boundingRect(self.contour)

        [X, Y, Width, Height] = self.boundingRect_cv2

        self.intBoundingRect_cv2_X = X
        self.intBoundingRect_cv2_Y = Y
        self.intBoundingRect_cv2_Width = Width
        self.intBoundingRect_cv2_Height = Height

        self.intboundingRect_cv2_Area = self.intBoundingRect_cv2_Width * self.intBoundingRect_cv2_Height

        self.center_X = (self.intBoundingRect_cv2_X + self.intBoundingRect_cv2_X + self.intBoundingRect_cv2_Width) / 2
        self.center_Y = (self.intBoundingRect_cv2_Y + self.intBoundingRect_cv2_Y + self.intBoundingRect_cv2_Height) / 2

        self.diagonal_Size = math.sqrt((self.intBoundingRect_cv2_Width ** 2) + (self.intBoundingRect_cv2_Height ** 2))

        self.aspect_Ratio = float(self.intBoundingRect_cv2_Width) / float(self.intBoundingRect_cv2_Height)
    # end constructor

# end class








