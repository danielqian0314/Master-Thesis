#%%


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os

import matplotlib.pyplot as plt



#%% Get ids of patients
#data_path = './data/'
data_path= "./med_data/Melanom/Melanom_Lung/"
patients = os.listdir(data_path)
patients.sort()
#df=pd.read_excel("Auswertung_Melanomstudie _CNN.xlsx", index_col=0,header=1)

patients=["001"]
print(patients)
#%%
def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=3)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


def findDicomfile(path):
    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM

#%% Load train data

#%% Load slice
print('-'*30)
print('Loading DICOM...')
print('-'*30)
image_type="CT"
images=[]
imagesPET=[]
masks=[]
total=0
for patient_id in patients:
    lstFilesDCM=findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+ image_type)
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    images.append(slices)
    total+=len(slices)
    print('Patient {0}: {1} {2} slices'.format(patient_id,len(slices),image_type))

# for patient_id in patients:
#     lstFilesDCM=findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+ "PET"+image_type)
#     slices = [dicom.read_file(s) for s in lstFilesDCM]
#     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
#     imagesPET.append(slices)
#     total+=len(slices)
#     print('Patient {0}: {1} PET{2} slices'.format(patient_id,len(slices),image_type))

# Load mask
for patient_id in patients:    
    lstFilesDCM = findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+ image_type+ "Lesion")
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    masks.extend(slices)
    print('Patient {0}: {1} {2} masks'.format(patient_id,slices[0].pixel_array.shape[0],image_type))

#%%
#%%
print(masks[0].ImagePositionPatient)
#print(masks[0].SliceThickness)
#print(imagesPET[0][0].ImagePositionPatient)
print(images[0][800].ImagePositionPatient)
print(images[0][800].SliceThickness)
#%%
x_max=0
y_max=0
z_max=0
for mask in masks:
    mask=mask.pixel_array
    where = np.array(np.where(mask))
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    print("{0}*{1}*{2}".format((x2-x1+1),(y2-y1+1),(z2-z1+1)))
    if x2-x1+1>x_max:
        x_max=x2-x1+1
    if y2-y1+1>y_max:
        y_max=y2-y1+1
    if z2-z1+1>z_max:
        z_max=z2-z1+1
print(x_max,y_max,z_max)
#%%
croped_masks=[]
croped_images=[]
i=0
for mask in masks:
    mask=mask.pixel_array
    where = np.array(np.where(mask))
    x1, y1, z1 = np.amin(where, axis=1)
    croped_mask=mask[x1:x1+x_max,y1+y_max,z1:z1+z_max]
    croped_masks.extend(croped_mask)
    for k in range(x1,x1+x_max):
      croped_images.extend(images[i][k].pixel_array[y1:y1+y_max,z1:z1+z_max])
    
#%%
plt.imshow(images[0][x1].pixel_array[y1:y2,z1:z2],cmap=plt.cm.gray)
#%%
plt.imshow(croped_mask[0]*images[0][x1].pixel_array[y1:y2,z1:z2],cmap=plt.cm.gray)



















#%%

#%%
print('-'*30)
print('Creating training images...')
print('-'*30)
imgs_temp = np.ndarray((total, image_rows, image_cols), dtype=np.uint16)
imgs_mask_temp = np.ndarray((total, image_rows, image_cols), dtype=np.uint16)

i = 0
for slices in images:
    for image in slices:
        img = image.pixel_array
        img = img.astype(np.uint16)
        imgs_temp[i] = img
        i += 1
    print('Done Patient {0}: {1}/{2} 2d images'.format(patient_id,i,total))

imgs = imgs_temp

print('Loading of train data done.')
# Load mask
for patient_id in patients:    
    masks=[]
    lstFilesDCM = findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+" "+ image_type+ " Läsion")
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    masks.extend(slices)
i = 0   
for mask in masks:
    img_mask = mask.pixel_array
    img_mask = img_mask.astype(np.uint16)
    imgs_mask_temp[i:img_mask.shape[0]]= img_mask
    i += img_mask.shape[0]
    print('Done Patient {0}: {1}/{2} 2d masks'.format(patient_id,i,total))

imgs_mask = imgs_mask_temp

print('Loading of masks done.')

# preprocessing

imgs_mask = preprocess(imgs_mask)
print('Preprocessing of masks done.')
imgs = preprocess(imgs)
print('Preprocessing of images done.')

np.save('CT_train.npy', imgs)
np.save('CT_mask_train.npy', imgs_mask)


print('Saving to .npy files done.')
#%%
def load_train_data(image_type):
    imgs_train = np.load(image_type+'_train.npy')
    imgs_mask_train = np.load(image_type+'_mask_train.npy')

    return imgs_train, imgs_mask_train


train,mask_train=load_train_data("CT")
print(mask_train.nonzero())
#%%print(mask_train.nonzero())
plt.imshow(preprocess_squeeze(train)[257]*preprocess_squeeze(mask_train)[257],cmap=plt.cm.gray)
#%%
plt.imshow(preprocess_squeeze(train)[257],cmap=plt.cm.gray)

#%%

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    return imgs_test



if __name__ == '__main__':
    create_train_data()
    create_test_data()
