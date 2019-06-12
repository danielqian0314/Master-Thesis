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
#INPUT_FOLDER = './data/'
INPUT_FOLDER = "./med_data/Melanom/Melanom_MRCT/Melanom_SelPat/"
patients = os.listdir(INPUT_FOLDER)
patients.remove('PETs')
patients.sort()
#df=pd.read_excel("Auswertung_Melanomstudie _CNN.xlsx", index_col=0,header=1)
print(patients)

#%%
def showPatientInfo(patient_id):

    return df.loc[int(patient_id)]

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
    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    # except:
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    # for s in slices:
    #     s.SliceThickness = slice_thickness

    print("patient:",patient_id, "number of slices:", len(slices))

    return slices

#%%
def load_scan_PET(patient_id, image_type):
    lstFilesDCM=findDicomfile(INPUT_FOLDER + "PETs/PETs/tue004_"+patient_id+"PET"+image_type)
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    print("patient:",patient_id, "number of slices:", len(slices))

    return slices
#%% load the compressed dicom mask of a given patient and image type
def load_scan_mask(patient_id, image_type):
    paths = findDicomfile(INPUT_FOLDER + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type+ " Läsion")
    print(paths)
    image =dicom.read_file(paths[0])

    return image
#%%
def load_scan_perfusion(patient_id):
    paths = findDicomfile(INPUT_FOLDER + patient_id+"/Texturanalyse/"+patient_id+" "+ "MR"+ " Perfusion")
    print(paths)
    image =dicom.read_file(paths[0])

    return image


#%% get a 3d pixel array from all slices
def get_pixels(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

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
first_patient_CT_mask.p

#%% 
first_patient_PETMR=load_scan_PET(patients[0],"MR")
#%%
first_patient_PETMR_pixels=get_pixels(first_patient_PETMR)
#%%
showSlice(first_patient_PETCT,first_patient_PETCT_pixels,200,0)


#%% load all CT images
i=0
patient_CT = load_scan(patients[i], "CT") 
patient_CT_pixels = get_pixels(patient_CT)
np.save("patient_CT_pixels"+patients[i], patient_CT_pixels)
print(patient_CT_pixels.shape)
#%%
patient_CT_mask = load_scan_mask(patients[i], "CT")
patient_CT_mask_pixels = patient_CT_mask.pixel_array
np.save("patient_CT_mask_pixels"+patients[i], patient_CT_mask_pixels)
print(patient_CT_mask_pixels.shape)
#%%
patient_MR = load_scan(patient, "MR") for patient in patients]
patient_MR_pixels = np.asarray([get_pixels(mr) for mr in patient_MR])
np.save("patient_MR_pixels", patient_MR_pixels)

#%%
patient_MR_mask = [load_scan_mask(patient, "MR") for patient in patients]
patient_MR_mask_pixels = np.asarray([mr_mask.pixel_array for mr_mask in patient_MR_mask])
np.save("patient_MR_mask_pixels", patient_MR_mask_pixels)
patient_MR_mask_pixels.shape

#%%
patient_PETCT = [load_scan_PET(patient, "CT") for patient in patients]
patient_PETCT_pixels = np.asarray([get_pixels(petct) for petct in patient_PETCT])
np.save("patient_PETCT_pixels", patient_PETCT_pixels)

#%%
patient_PETMR = [load_scan_PET(patient, "MR") for patient in patients]
patient_PETMR_pixels = np.asarray([get_pixels(petmr) for petmr in patient_PETMR])
np.save("patient_PETMR_pixels", patient_PETMR_pixels)

#%%
i=patient_MR_mask_pixels[0].nonzero()[0][0]
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.imshow(patient_MR_pixels[0,i,:,:])
ax.autoscale(False)
ax2 = fig.add_subplot(2, 1, 2, sharex=ax, sharey=ax)
ax2.imshow(patient_MR_pixels[0,i,:,:]*patient_MR_mask_pixels[0,i,:,:])
ax2.autoscale(False)
plt.show()

#%%
i=patient_CT_mask_pixels[0].nonzero()[0][0]
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.imshow(patient_CT_pixels[0,i+1,:,:])
ax.autoscale(False)
ax2 = fig.add_subplot(2, 1, 2, sharex=ax, sharey=ax)
ax2.imshow(patient_CT_pixels[0,i+1,:,:]*patient_CT_mask_pixels[0,i,:,:])
ax2.autoscale(False)
plt.show()




















#%%

