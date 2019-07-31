#%% Required Library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os


#%% Get ids of patients
df=pd.read_excel("Auswertung_Melanomstudie _CNN.xlsx", index_col=0,header=1)

#%%
df
#%%

def getMetadata(patientIDs):
    #%%
    ages=[df[df.index== id]['Age'].values[0] for id in  patientIDs]
    ages=np.asarray(ages)
    #%%
    genders=[0 if df[df.index== id]['Sex'].values[0]=='m' else 1 for id in  patientIDs]
    genders=np.asarray(genders)
    #%%
    weights=[df[df.index== id]['Weight [kg]'].values[0] for id in  patientIDs]
    weights=np.asarray(weights)
    #%%
    hights=[df[df.index== id]['Hight [m]'].values[0] for id in  patientIDs]
    hights=np.asarray(hights)
    #%%
    BMIs=[df[df.index== id]['BMI [kg/m2]'].values[0] for id in  patientIDs]
    BMIs=np.asarray(BMIs)

    #%%
    VOI_str=[df[df.index== id]['VOI (HL)'].values[0] for id in  patientIDs]
    VOI=[np.nan if voi_str!=voi_str else int(voi_str.split('L',1)[1]) for voi_str in VOI_str]
    VOI=np.asarray(VOI)
    #%%
    SUV=[df[df.index==id]['SUV (peak)'].values[0] for id in  patientIDs]
    SUV=np.asarray(SUV)
    
    #%%
    def ortClass(ort):
        if ort =='Knochen':
            return 0
        elif ort =='Leber':
            return 1
        elif ort =='LN':
            return 2
        elif ort == 'Lunge':
            return 3
        elif ort =='Milz':
            return 4
        elif ort =='Pleura':
            return 5
        elif ort =='Visceral':
            return 6
        elif ort == 'Weichteil':
            return 7
        else:
            return np.nan


    Orts=[ortClass(df[df.index==id][' Ort (HL)'].values[0]) for id in  patientIDs]
    Orts=np.asarray(Orts)


    
    return [ages,genders,weights,hights,BMIs,VOI,SUV,Orts]

#%%
def getLabels(patientIDs):
    PFS=[1246 if df[df.index==id]['PFS'].isna().values[0] or df[df.index==id]['PFS'].values[0]<-40000 else df[df.index==id]['PFS'].values[0] for id in  patientIDs]
    OAS=[1266 if df[df.index==id]['OAS'].isna().values[0] or df[df.index==id]['OAS'].values[0]<-40000 else df[df.index==id]['OAS'].values[0] for id in  patientIDs]
    Responses=[df[df.index== id]['Response'].values[0] for id in  patientIDs]
    responsesClass=[]
    for response in Responses:
        if response=='Progress':
            responsesClass.append(0)
        elif response=='Complete Response' or response=='Complete response':
            responsesClass.append(1)
        elif response=='Partial Response':
            responsesClass.append(2)
        elif response=='Stable disease':
            responsesClass.append(3)
        else:
            responsesClass.append(np.nan)
    
    responsesClass=np.asarray(responsesClass)
    return [responsesClass,PFS,OAS]

