#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:16:44 2019

@author: d1304
"""
#%%
import pandas as pd
data = pd.read_csv('/home/d1304/treatmentresponse/model/axis_lr0.001_class.csv')
fig = plt.figure(figsize=(18, 15))
plt.plot(data['loss'], marker='*', label='loss')
plt.plot(data['Response_Classification_loss'], marker='*', label='Response_Classification_loss')
plt.plot(data['Survival_Rate_loss'], marker='*', label='Survival_Rate_loss')
plt.plot(data['Treatment_Regress_loss'], marker='*', label='Treatment_Regress_loss')
 
plt.xticks(arange(151))
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.15)

plt.show()
plt.savefig('/home/d1304/treatmentresponse/model/loss.png')
print('save successful') 
#%%
import pandas as pd
data = pd.read_csv('/home/d1304/treatmentresponse/model/axis_lr0.001_class.csv')
fig = plt.figure(figsize=(18, 15))
plt.plot(data['Response_Classification_acc'], marker='*', label='Response_Classification_acc')
plt.plot(data['Survival_Rate_acc'], marker='*', label='Survival_Rate_acc')
plt.plot(data['Treatment_Regress_acc'], marker='*', label='Treatment_Regress_acc')
 
plt.xticks(arange(151))
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.15)

plt.show()
plt.savefig('/home/d1304/treatmentresponse/model/accuracy.png')
print('save successful') 