#%%
import numpy as np
import treatmentresponse.label as label
import tensorflow.keras
from matplotlib import pyplot as plt

data_path='/home/d1304/no_backup/d1304/'
def get_data_MR_2D(patientIDs):
    patientsMR=np.load(data_path+'patientsMR.npy').astype(int)
    MR2d=np.load(data_path+'MR2d.npy')
    PETMR2d=np.load(data_path+'PETMR2d.npy')
    input_MR2d=[]
    input_PETMR2d=[]
    for k in range(len(patientIDs)):
        for i in range(len(patientsMR)):            
            if patientsMR[i]==patientIDs[k]:
                input_MR2d.extend(MR2d[i*64:(i+1)*64])
                input_PETMR2d.extend(PETMR2d[i*64:(i+1)*64])
                break
    
    input_MR2d=np.asarray(input_MR2d)
    input_PETMR2d=np.asarray(input_PETMR2d)

    input_image=np.zeros((input_MR2d.shape[0],input_MR2d.shape[1],input_MR2d.shape[2],2))
    input_image[:,:,:,0]=input_MR2d
    input_image[:,:,:,1]=input_PETMR2d

    input=[input_image]    
    for meta in label.getMetadata(patientIDs):
        meta=np.repeat(meta,64,axis=0)
        input.append(meta)
   
    
    output=[]    
    for labels in label.getLabels(patientIDs):
        labels=np.repeat(labels,64,axis=0)
        output.append(labels)

    return input,output


def get_data_CT_2D(patientIDs):
    patientsCT=np.load(data_path+'patientsCT.npy').astype(int)
    CT3d=np.load(data_path+'CT3d.npy')
    PETCT3d=np.load(data_path+'PETCT3d.npy')
    input_CT2d=[]
    input_PETCT2d=[]
    num_slices_patient=[]
    for k in range(len(patientIDs)):
        for i in range(len(patientsCT)):            
            if patientsCT[i]==patientIDs[k]:
                where = np.array(np.where(CT3d[i]))
                x2, y2, z2 = np.amax(where, axis=1)            
                input_CT2d.extend(CT3d[i][0:x2+1])
                input_PETCT2d.extend(PETCT3d[i][0:x2+1])              
                num_slices_patient.append(x2+1)
                break
    
    input_CT2d=np.asarray(input_CT2d)
    input_PETCT2d=np.asarray(input_PETCT2d)

    input_image=np.zeros((input_CT2d.shape[0],input_CT2d.shape[1],input_CT2d.shape[2],2))
    input_image[:,:,:,0]=input_CT2d
    input_image[:,:,:,1]=input_PETCT2d

    input=[input_image]    
    for meta in label.getMetadata(patientIDs):
        meta=np.repeat(meta,num_slices_patient,axis=0)
        input.append(meta)
   
    
    output=[]    
    for labels in label.getLabels_Class(patientIDs):
        labels=np.repeat(labels,num_slices_patient,axis=0)
        output.append(labels)

    return input,output

def get_data_CT_3axis_2D(patientIDs):
    patientsCT=np.load(data_path+'patientsCT.npy').astype(int)
    CT3d=np.load(data_path+'CT3d.npy')
    PETCT3d=np.load(data_path+'PETCT3d.npy')
    input_CT2d=[]
    input_PETCT2d=[]
    num_slices_patient=[]
    for k in range(len(patientIDs)):
        for i in range(len(patientsCT)):            
            if patientsCT[i]==patientIDs[k]:
                where = np.array(np.where(CT3d[i]))
                x1, y1, z1 = np.amax(where, axis=1)
                print(x1,y1,z1)
                slice2d_y=np.zeros((y1+1,65,86))
                slicepet2d_y=np.zeros((y1+1,65,86))
                slice2d_z=np.zeros((z1+1,65,86))
                slicepet2d_z=np.zeros((z1+1,65,86))
                for x in range(x1+1):               
                    input_CT2d.append(CT3d[i][x])
                    input_PETCT2d.append(PETCT3d[i][x])
                for y in range(y1+1):
                    slice2d_y[y,:42,:86]=CT3d[i][:,y,:]                   
                    input_CT2d.append(slice2d_y[y])
                    slicepet2d_y[y,:42,:86]=PETCT3d[i][:,y,:]
                    input_PETCT2d.append(slicepet2d_y[y])
                for z in range(z1+1):
                    slice2d_z[z,:42,:65]=CT3d[i][:,:,z]
                    input_CT2d.append(slice2d_z[z])
                    slicepet2d_z[z,:42,:65]=PETCT3d[i][:,:,z]
                    input_PETCT2d.append(slice2d_z[z])
                    
                
                num_slices_patient.append(x1+1+y1+1+z1+1)
                break
    
    input_CT2d=np.asarray(input_CT2d)
    input_PETCT2d=np.asarray(input_PETCT2d)

    input_image=np.zeros((input_CT2d.shape[0],input_CT2d.shape[1],input_CT2d.shape[2],2))
    input_image[:,:,:,0]=input_CT2d
    input_image[:,:,:,1]=input_PETCT2d

    input=[input_image]    
    for meta in label.getMetadata(patientIDs):
        meta=np.repeat(meta,num_slices_patient,axis=0)
        input.append(meta)
   
    
    output=[]    
    for labels in label.getLabels_Class(patientIDs):
        labels=np.repeat(labels,num_slices_patient,axis=0)
        output.append(labels)

    return input,output

#%%
input,output=get_data_CT_3axis_2D([1])



