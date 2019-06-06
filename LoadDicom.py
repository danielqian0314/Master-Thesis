%matplotlib inline
#%% Required Library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#%% Get ids of patients
INPUT_FOLDER = './data/'
#INPUT_FOLDER = "./med_data/Melanom/"
patients = os.listdir(INPUT_FOLDER)
patients.sort()
df=pd.read_excel(INPUT_FOLDER+"Auswertung_Melanomstudie _CNN.xlsx", index_col=0,header=1)

#%% get path of dicom files in a given dir
def findDicomfile(path):
    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM
#%% load all dicom slices of a given patient and image type
def load_scan(patient_id, image_type):
    lstFilesDCM=findDicomfile(INPUT_FOLDER + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type)
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    print("number of slices:", len(slices))

    return slices
#%% load the compressed dicom mask of a given patient and image type
def load_scan_mask(patient_id, image_type):
    paths = findDicomfile(INPUT_FOLDER + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type+ " Läsion")
    print(paths)
    image =dicom.read_file(paths[0])

    return image


#%% get a 3d pixel array from all slices
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    print("size of pixel array:", image.shape)

    return np.array(image, dtype=np.int16)


#%% show certain slice
def showSlice(patient, patient_pixels, slice,axis):
    pat_name = patient[slice].PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", patient[slice].PatientID)
    print("Modality.........:", patient[slice].Modality)
    print("Study Date.......:", patient[slice].StudyDate)
    print("Slice Number.....:", slice)
    print("slice direction..:", axis)
    
    if axis==0 :
        plt.imshow(patient_pixels[slice,:,:], cmap=plt.cm.gray)
    elif axis==1:
        plt.imshow(patient_pixels[:,slice,:], cmap=plt.cm.gray)
    elif axis==2:
        plt.imshow(patient_pixels[:,:,slice], cmap=plt.cm.gray)
    else:
        print("axis out of range")

    plt.show()

#%% show mask
def showMaskInfo(patient_mask):
    pat_name = patient_mask.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", patient_mask.PatientID)
    print("Modality.........:", patient_mask.Modality)
    print("Study Date.......:", patient_mask.StudyDate)
    print("Mask size........:", patient_mask.pixel_array.shape)

#%% 
first_patient_CT_mask = load_scan_mask(patients[0], "CT")
plt.imshow(first_patient_CT_mask.pixel_array[200,:,:],cmap=plt.cm.gray)
#%% 

#%%
showSlice(first_patient_CT,first_patient_CT_pixel,250,1)

#%%
def showPatientInfo(patient_id):

    return df.loc[int(patient_id)]
#%%
showPatientInfo(patients[0])
first_patient_CT = load_scan(patients[0], "CT")
first_patient_CT_pixel = get_pixels_hu(first_patient_CT)


















#%%

