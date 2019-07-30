#%%


import numpy as np # linear algebra
import pydicom as dicom
import os

import matplotlib.pyplot as plt



#%% Get ids of patients
data_path= "./med_data/Melanom/Melanom_Lung/"
patients = os.listdir(data_path)
patients.sort()

patients.remove("029")
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
imagesMR,masksMR,petsMR=load_dicom('MR')

#%%
masksMR[9]=dicom.read_file('/home/d1304/med_data/Melanom/Melanom_Lung/017/Texturanalyse/017MRLesion/Tue004_017MR/Tue004_017MR.SEG..OBJ1.OBJ1.2019.07.12.12.12.17.dcm')
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
#%%
shapeMR=np.array((64,68,118))

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
            
#%%
croped_postions_mr=[]
croped_masks_mr=[]
croped_images_mr=[]
croped_PET_mr=[]
patientsMR=[]

#%%
for i in range(9,len(masksMR)):
    mask=masksMR[i].pixel_array
    where = np.array(np.where(mask))
    x1, y1, z1 = np.amin(where, axis=1)
    print(x1,y1,z1)
    if imagesMR[i][x1].ImagePositionPatient[2]>petsMR[i][-1].ImagePositionPatient[2] or imagesMR[i][x1].ImagePositionPatient[2]<petsMR[i][0].ImagePositionPatient[2]:
        print('{0}: unable to find allign PET'.format(patients[i]))
    else:
        croped_position=np.array([imagesMR[i][x1].ImagePositionPatient[0]+z1*imagesMR[i][x1].PixelSpacing[0], imagesMR[i][x1].ImagePositionPatient[1]+y1*imagesMR[i][x1].PixelSpacing[1],imagesMR[i][x1].ImagePositionPatient[2]])
        print('{0}:position for lesion {1}'.format(patients[i],croped_position))
        croped_postions_mr.append(croped_position)
        
        croped_mask=mask[x1:x1+shapeMR[0],y1:y1+shapeMR[1],z1:z1+shapeMR[2]]
        croped_masks_mr.append(croped_mask)
        print('{0}: mask croped {1}'.format(patients[i],croped_mask.shape))
        
        croped_image=[]
        for k in range(x1,x1+shapeMR[0]):
          if k>=len(imagesMR[i]):
              croped_slice=imagesMR[i][k-len(imagesMR[i])].pixel_array[y1:y1+shapeMR[1],z1:z1+shapeMR[2]]
          else:
              croped_slice=imagesMR[i][k].pixel_array[y1:y1+shapeMR[1],z1:z1+shapeMR[2]]
          croped_image.append(croped_slice)    
        croped_images_mr.append(croped_image)   
        print('{0}: MR croped {1} {2}'.format(patients[i],len(croped_image),croped_image[0].shape))
        
               
        position=croped_position
        croped_pet=[]
        for t in range(x1,x1+shapeMR[0]):
            if t>=len(imagesMR[i]):
                k=t-len(imagesMR[i])  
            else:
                k=t
            pet_num=0
            for n in range(len(petsMR[i])):
                if imagesMR[i][k].ImagePositionPatient[2]<=petsMR[i][n].ImagePositionPatient[2]:
                    croped_slice_pet=get_aligned_croped_pet(petsMR[i][n],shapeMR[1:3],position,imagesMR[i][k].PixelSpacing[0])
                    croped_pet.append(croped_slice_pet)
                    break
            
        print('{0}: PETMR croped {1} {2}'.format(patients[i], len(croped_pet), croped_pet[0].shape))
        croped_PET_mr.append(croped_pet)
        patientsMR.append(patients[i])
 
#%%
patientsMR=[patientsMR[i] for i in range(36) if not i==15]
 #%%
print('-'*30)
print('Creating training datas 2d (MR)...')
print('-'*30)
total=64*35
image_rows=68
image_cols=118
mrs2d = np.ndarray((total, image_rows, image_cols))
petmrs2d = np.ndarray((total, image_rows, image_cols))
masks2d = np.ndarray((total, image_rows, image_cols))


i = 0
for croped_image in croped_images_mr:
    for img in croped_image:       
        input=np.zeros((68,118))
        input[:img.shape[0],:img.shape[1]]=img
        mrs2d[i] = input
        i += 1
    
print('Loading of mr data done.')

i = 0
for croped_pet in croped_PET_mr:
    for img in croped_pet:       
        petmrs2d[i] = img
        i += 1
    
print('Loading of petmr data done.')



# Load mask
i = 0
for croped_mask in croped_masks_mr:       
        mask=np.zeros((64,68,118))
        mask[:croped_mask.shape[0],:croped_mask.shape[1],:croped_mask.shape[2]]=croped_mask
        masks2d[i:i+64] = mask
        i += 64
        
print('Loading of masks done.')


np.save('/no_backup/d1304/MR2d.npy', mrs2d*masks2d)
np.save('/no_backup/d1304/PETMR2d.npy', petmrs2d*masks2d)


print('Saving to .npy files done.')

#%%
print('-'*30)
print('Creating training datas 3d (MR)...')
print('-'*30)
total=35
image_depths=64
image_rows=68
image_cols=118
mrs3d = np.ndarray((total, image_depths, image_rows, image_cols))
petmrs3d = np.ndarray((total, image_depths, image_rows, image_cols))
masks3d = np.ndarray((total, image_depths, image_rows, image_cols))


i = 0

for croped_image in croped_images_mr:
    j=0
    for img in croped_image:       
        input=np.zeros((68,118))
        input[:img.shape[0],:img.shape[1]]=img
        mrs3d[i][j] = input
        j+=1
    i += 1
    
print('Loading of mr data done.')

i = 0
for croped_pet in croped_PET_mr:
    j=0
    for img in croped_pet:       
        petmrs3d[i][j] = img
        j+=1
    i += 1
    
print('Loading of petmr data done.')



# Load mask
i = 0
for croped_mask in croped_masks_mr:       
        mask=np.zeros((64,68,118))
        mask[:croped_mask.shape[0],:croped_mask.shape[1],:croped_mask.shape[2]]=croped_mask
        masks3d[i] = mask
        i += 1
        
print('Loading of masks done.')



np.save('/no_backup/d1304/MR3d.npy', mrs3d*masks3d)
np.save('/no_backup/d1304/PETMR3d.npy', petmrs3d*masks3d)


print('Saving to .npy files done.')

#%%
np.save('/no_backup/d1304/patientsMR.npy', np.asarray(patientsMR))
np.save('/no_backup/d1304/positionMRLesion.npy', np.asarray(croped_postions_mr))


