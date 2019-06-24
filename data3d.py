#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave

from skimage.io import imread

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#%% Get ids of patients
data_path = "./data/"
#INPUT_FOLDER = "./med_data/Melanom/Melanom_MRCT/Melanom_SelPat/"
patients = os.listdir(data_path)
patients.remove('PETs')
patients.sort()
#df=pd.read_excel("Auswertung_Melanomstudie _CNN.xlsx", index_col=0,header=1)
print(patients)

#%%
def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


def findDicomfile(path):
    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM

#%% Load train data
print('-'*30)
print('Loading DICOM...')
print('-'*30)
image_type="CT"
image_rows = int(512)
image_cols = int(512)
image_depth = 16 
images=[]
total=0
for patient_id in patients:
    lstFilesDCM=findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type)
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    images.append(slices)
    total+=len(slices)
    print('Patient {0}: {1} {2} slices'.format(patient_id,len(slices),image_type))
total=total//(image_depth//2)+1

print('-'*30)
print('Creating training images...')
print('-'*30)
imgs = np.ndarray((total, image_depth,image_rows, image_cols), dtype=np.uint16)
imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint16)
imgs_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint16)
imgs_mask_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint16)

i = 0
j = 0
for slices in images:
    for image in slices:
        img = image.pixel_array
        img = img.astype(np.uint16)
        imgs_temp[i,j] = img
        j += 1
        if j % (image_depth//2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d images'.format(i, total))

for x in range(0, imgs_temp.shape[0]-1):
    imgs[x]=np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

print('Loading of train data done.')
print(imgs.shape)
#%% Load mask
for patient_id in patients:    
    masks=[]
    lstFilesDCM = findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type+ " Läsion")
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    masks.extend(slices)
i = 0 
j = 0  
for mask in masks:
    img_mask = mask.pixel_array
    img_mask = img_mask.astype(np.uint16)
    for k in range(0,img_mask.shape[0]):
        imgs_mask_temp[i,j]= img_mask[k]
        j += 1
        if j % (image_depth//2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d masks'.format(i, total))

for x in range(0, imgs_mask_temp.shape[0]-1):
    imgs_mask[x]=np.append(imgs_mask_temp[x], imgs_mask_temp[x+1], axis=0)

print('Loading of masks done.')

#%% preprocessing

imgs_mask = preprocess(imgs_mask)
print('Preprocessing of masks done.')
imgs = preprocess(imgs)
print('Preprocessing of images done.')

np.save('CT_3d_train.npy', imgs)
np.save('CT_3d_mask_train.npy', imgs_mask)


print('Saving to .npy files done.')
#%%
def load_train_data(image_type):
    imgs_train = np.load(image_type+'_3d_train.npy')
    imgs_mask_train = np.load(image_type+'_3d_mask_train.npy')

    return imgs_train, imgs_mask_train


train,mask_train=load_train_data("CT")

#%%
