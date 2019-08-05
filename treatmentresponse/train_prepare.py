#%%
import numpy as np
import treatmentresponse.label as label
import tensorflow.keras
#%%
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

def get_data_central_MR_2D(patientIDs):
    patientsMR=np.load(data_path+'patientsMR.npy').astype(int)
    MR2d=np.load(data_path+'MR2d.npy')
    PETMR2d=np.load(data_path+'PETMR2d.npy')
    input_MR2d=[]
    input_PETMR2d=[]
    for k in range(len(patientIDs)):        
        for i in range(len(patientsMR)):            
            if patientsMR[i]==patientIDs[k]:
                 where = np.array(np.where(MR2d[i*64:(i+1)*64]))
                 x2, y2, z2 = np.amax(where, axis=1)
                central_num=int(x2//2)
                input_MR2d.extend(MR2d[i*64+central_num-1:i*64+central_num+1])
                input_PETMR2d.extend(PETMR2d[i*64+central_num-1:i*64+central_num+1])
                break
    
    input_MR2d=np.asarray(input_MR2d)
    input_PETMR2d=np.asarray(input_PETMR2d)

    input_image=np.zeros((input_MR2d.shape[0],input_MR2d.shape[1],input_MR2d.shape[2],2))
    input_image[:,:,:,0]=input_MR2d
    input_image[:,:,:,1]=input_PETMR2d

    input=[input_image]    
    for meta in label.getMetadata(patientIDs):
        meta=np.repeat(meta,3,axis=0)
        input.append(meta)
   
    
    output=[]    
    for labels in label.getLabels(patientIDs):
        labels=np.repeat(labels,3,axis=0)
        output.append(labels)

    return input,output