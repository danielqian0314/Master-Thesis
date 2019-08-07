#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:16:44 2019

@author: d1304
"""

import pandas as pd
df = pd.read_csv('/home/d1304/treatmentresponse/model/3axis_lr0.01_class.csv')
fig = plt.figure(figsize=(18, 15))
    plt.plot(data['regression_predicted_survival_rate'], marker='*', label='predicted_survival_rate')
    plt.plot(data['actual_survival_rate'], marker='x', label='actual_survival_rate', )
     
    plt.xticks(arange(len(patientsID)), data['patient_id'], rotation=60)
    plt.xlabel("Patient Number")
    plt.ylabel("Survival Rate")
    plt.legend()
    plt.grid()
    plt.gcf().subplots_adjust(bottom=0.15)
    std = np.std(np.array(data['regression_predicted_survival_rate']) - np.array(data['actual_survival_rate']))
    mae = np.mean(np.abs(np.array(data['regression_predicted_survival_rate']) - np.array(data['actual_survival_rate'])))
    plt.title("Predicted survival rate vs. Actual survival rate, Std:{:0.2f}, MAE:{:0.2f}".format(std, mae))
    plt.show()
    save_file = logging_file + "survival_rate_plot.png"
    plt.savefig(save_file)
    print('save successful') 