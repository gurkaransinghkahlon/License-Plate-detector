# ProbablePlate.py

import cv2
import numpy as np

###################################################################################################
class ProbablePlate:

    # constructor #################################################################################
    def __init__(self):
        self.plate_Image = None
        self.grayscale_Image = None
        self.thresh_Image = None

        self.plate_Location = None

        self.character_String = ""
    # end constructor

# end class




