# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import pydicom as dicom
import os
import numpy as np
from matplotlib import pyplot as plt, cm

#%%
# fetch the path to the test data
PathDicom = "./med_data/Melanom/Melanom_MRCT/Melanom_SelPat/003/Texturanalyse/003 MR Perfusion/19030515"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))

#%%

# read the file
dataset = dicom.read_file("/home/d1304/med_data/Melanom/Melanom_MRCT/Melanom_SelPat/012/Texturanalyse/012 MR/19022015/57120000/47235034")
    
# Normal mode:
print()
print("Storage type.....:", dataset.SOPClassUID)
print()

pat_name = dataset.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print("Patient's name...:", display_name)
print("Patient id.......:", dataset.PatientID)
print("Modality.........:", dataset.Modality)
print("Study Date.......:", dataset.StudyDate)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

# use .get() if not sure the item exists, and want a default value if missing
print("Slice location...:", dataset.get('SliceLocation', "(missing)"))
print(dataset.pixel_array.shape)
plt.imshow(dataset.pixel_array)

