#%%


import numpy as np # linear algebra
import pydicom as dicom
import os

import matplotlib.pyplot as plt



#%% Get ids of patients
data_path= "./med_data/Melanom/Melanom_Lung/"
patients = os.listdir(data_path)
patients.sort()
print(np.asarray(patients).astype(int))

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


#%% Load slice
def load_dicom(image_type):
    print('-'*30)
    print('Loading DICOM...')
    print('-'*30)
    images=[]
    masks=[]
    pets=[]
    for patient_id in patients:
        lstFilesDCM=findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+ image_type)
        slices = [dicom.read_file(s) for s in lstFilesDCM]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        if slices!=[]:
            images.append(slices)
        
        print('Patient {0}: {1} {2} slices'.format(patient_id,len(slices),image_type))
        lstFilesDCM = findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+ image_type+ "Lesion")
        mask = [dicom.read_file(s) for s in lstFilesDCM]
        if mask !=[]:
            masks.append(mask[0])
            print('Patient {0}: {1} {2} masks'.format(patient_id,mask[0].pixel_array.shape[0],image_type))
            
        
        lstFilesDCM = findDicomfile(data_path + patient_id+"/Texturanalyse/"+patient_id+ 'PET'+image_type)
        pet = [dicom.read_file(s) for s in lstFilesDCM]
        for s in pet:
            if hasattr(s,'ImagePositionPatient')==0:
                pet.remove(s)
        pet.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        if pet !=[]:
            pets.append(pet)
            print('Patient {0}: {1}  PET{2}s'.format(patient_id,len(pet),image_type)) 
            
    return images,masks,pets


#%%

imagesCT,masksCT,petsCT=load_dicom('CT')
#%%
patient_id='052'
lstFilesDCM=findDicomfile('/home/d1304/med_data/Melanom/Melanom_Lung/052/Texturanalyse/052CT/19071210')
slices = [dicom.read_file(s) for s in lstFilesDCM]
slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
print('Patient {0}: {1} {2} slices'.format(patient_id,len(slices),'CT'))
imagesCT[31]=slices
#%% determing the croped size
def largest_lesion_size(masks):
    shape=np.zeros(3)
    for mask in masks:
        mask=mask.pixel_array
        where = np.array(np.where(mask))
        x1, y1, z1 = np.amin(where, axis=1)
        x2, y2, z2 = np.amax(where, axis=1)
        print("{0}*{1}*{2}".format((x2-x1+1),(y2-y1+1),(z2-z1+1)))
        if x2-x1+1>shape[0]:
            shape[0]=x2-x1+1
        if y2-y1+1>shape[1]:
            shape[1]=y2-y1+1
        if z2-z1+1>shape[2]:
            shape[2]=z2-z1+1
    print("largest lesion size:",shape)
    
    return shape


shapeCT = np.array((42,65,86))





#%%
def get_aligned_croped_pet(origin,shape,position,spacing):
    croped=np.zeros(shape)
    for m in range(shape[0]):
        for n in range(shape[1]):
            position_x=position[0]+n*spacing
            position_y=position[1]+m*spacing
            x=(position_x-origin.ImagePositionPatient[0])/origin.PixelSpacing[0]
            y=(position_y-origin.ImagePositionPatient[1])/origin.PixelSpacing[1]
            croped[m][n]=origin.pixel_array[int(y)][int(x)]
            
    return croped
            
   

   

#%% croping and alignment
croped_postions_ct=[]
croped_masks_ct=[]
croped_images_ct=[]
croped_PET_ct=[]
patientsCT=[]

for i in range(len(masksCT)):
    mask=masksCT[i].pixel_array
    where = np.array(np.where(mask))
    x1, y1, z1 = np.amin(where, axis=1)
    print(x1,y1,z1)
    if imagesCT[i][x1].ImagePositionPatient[2]>petsCT[i][-1].ImagePositionPatient[2] or imagesCT[i][x1].ImagePositionPatient[2]<petsCT[i][0].ImagePositionPatient[2]:
        print('{0}: unable to find allign PET'.format(patients[i]))
    else:
        croped_position=np.array([imagesCT[i][x1].ImagePositionPatient[0]+z1*imagesCT[i][x1].PixelSpacing[0], imagesCT[i][x1].ImagePositionPatient[1]+y1*imagesCT[i][x1].PixelSpacing[1],imagesCT[i][x1].ImagePositionPatient[2]])
        print('{0}:position for lesion {1}'.format(patients[i],croped_position))
        croped_postions_ct.append(croped_position)
        
        croped_mask=mask[x1:x1+shapeCT[0],y1:y1+shapeCT[1],z1:z1+shapeCT[2]]
        croped_masks_ct.append(croped_mask)
        print('{0}: mask croped '.format(patients[i]))
        
        croped_image=[]
        for k in range(x1,x1+shapeCT[0]):
          croped_slice=imagesCT[i][k].pixel_array[y1:y1+shapeCT[1],z1:z1+shapeCT[2]]
          croped_image.append(croped_slice)    
        croped_images_ct.append(croped_image)   
        print('{0}: CT croped '.format(patients[i]))
        
               
        position=croped_position
        croped_pet=[]
        for t in range(x1,x1+shapeCT[0]):        
            pet_num= (imagesCT[i][t].ImagePositionPatient[2]-petsCT[i][0].ImagePositionPatient[2])//3  
            pet_num=int(pet_num)
            croped_slice_pet=get_aligned_croped_pet(petsCT[i][pet_num],shapeCT[1:3],position,imagesCT[i][t].PixelSpacing[0])
            croped_pet.append(croped_slice_pet)
        print('{0}: PETCT croped'.format(patients[i]))
        croped_PET_ct.append(croped_pet)
        patientsCT.append(patients[i])
        

#%%
print('-'*30)
print('Creating training datas 2d (CT)...')
print('-'*30)
total=42*34
image_rows=65
image_cols=86
cts2d = np.ndarray((total, image_rows, image_cols))
petcts2d = np.ndarray((total, image_rows, image_cols))
masks2d = np.ndarray((total, image_rows, image_cols))


i = 0
for croped_image in croped_images_ct:
    for img in croped_image:       
        cts2d[i] = img
        i += 1
    
print('Loading of ct data done.')

i = 0
for croped_pet in croped_PET_ct:
    for img in croped_pet:       
        petcts2d[i] = img
        i += 1
    
print('Loading of petct data done.')



# Load mask
i = 0
for croped_mask in croped_masks_ct:       
        masks2d[i:i+42] = croped_mask
        i += 42
        
print('Loading of masks done.')


#%%
np.save('/no_backup/d1304/CT2d.npy', cts2d*masks2d)
np.save('/no_backup/d1304/PETCT2d.npy', petcts2d*masks2d)
np.save('/no_backup/d1304/positionCTLesion.npy', np.asarray(croped_postions_ct))
np.save('/no_backup/d1304/patientsCT.npy', np.asarray(patientsCT))

print('Saving to .npy files done.')


#%%
print('-'*30)
print('Creating training datas 3d (CT)...')
print('-'*30)
total=34
image_depths=42
image_rows=65
image_cols=86
cts3d = np.ndarray((total,image_depths,image_rows, image_cols))
petcts3d = np.ndarray((total,image_depths,image_rows, image_cols))
masks3d = np.ndarray((total,image_depths,image_rows, image_cols))


i = 0
for croped_image in croped_images_ct:
    j = 0
    for img in croped_image:       
        cts3d[i][j] = img
        j += 1
    i += 1
    
print('Loading of ct data done.')

i = 0
for croped_pet in croped_PET_ct:
    j=0
    for img in croped_pet:       
        petcts3d[i][j] = img
        j+=1
    i += 1
    
print('Loading of petct data done.')



# Load mask
i = 0
for croped_mask in croped_masks_ct:       
        masks3d[i] = croped_mask
        i += 1
        
print('Loading of masks done.')

#%%

np.save('/no_backup/d1304/CT3d.npy', cts3d*masks3d)
np.save('/no_backup/d1304/PETCT3d.npy', petcts3d*masks3d)

print('Saving to .npy files done.')