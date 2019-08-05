#%%
import os
import sys
import argparse
import shutil
import yaml
import matplotlib.pyplot as plt
import statistics
import math
import pandas as pd
import numpy as np
from pylab import *

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import callbacks as cb
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint
)
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.layers import *


# Import own scripts
import treatmentresponse.training_prepare as training_prepare
import treatmentresponse.TreatmentResponseNet as TreatmentResponseNet





#%%
def train(cf):
    # parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['Training']['gpu_num'])
    train_eval_ratio = cf['Data']['train_val_split']
    batch_size = cf['Training']['batch_size']
    epoch = cf['Training']['num_epochs']
    slice_size = cf['Training']['slice_size']
    num_slice_per_group = cf['Training']['num_slice_per_group']
    patientsID=cf['Data']['patientsID_train']
    # get train data 
    
    input,output=training_prepare.get_data_MR_2D(patientsID)
    output[0]=tensorflow.keras.utils.to_categorical(output[0],4)
    output[1]=tensorflow.keras.utils.to_categorical(output[0],3)
    output[2]=tensorflow.keras.utils.to_categorical(output[0],3)

    # take n groups samples for each patient, calculate the number of total samples
    count_train= int(input[0].shape[0]*(1-train_eval_ratio))
    count_validation = input[0].shape[0]-count_train

    steps_per_epoch = math.ceil(count_train / batch_size)
    validata_steps = math.ceil(count_validation / batch_size)

    print('expected step per epoch: ', steps_per_epoch)
    print('expected validata_steps: ', validata_steps)
    print('expected step per batch_size: ', batch_size)

    print('-' * 75)
    print(' Model\n')

    if cf['Pretrained_Model']['path'] is not None:
        print(' Load pretrained model')
        model = tensorflow.keras.models.load_model(filepath=cf['Pretrained_Model']['path'])
    else:
        print(' Load model')
        model, _ = TreatmentResponseNet.createModel(slice_size, num_slice_per_group)

    learning_rate = cf['Training']['learning_rate']
    adm = optimizers.Adam(lr=learning_rate)
    #model.compile(loss=['categorical_crossentropy','mse','mse'], optimizer='adam', metrics={'Response_Classification':'accuracy', 'Survival_Rate':'mae','Treatment_Regress':'mae'})
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
    model.summary()
    print(' Model compiled!')

    def get_callbacks(model_file, logging_file):
        callbacks = list()
        # save the model
        callbacks.append(cb.ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, mode='auto'))
        # save log file
        callbacks.append(CSVLogger(logging_file, append=True))
        return callbacks

    print('-' * 75)
    print(' Training...')

    

    path_w = cf['Paths']['model'] + "treatment_net" + ".hdf5"
    logging_file = cf['Paths']['model'] + "treatment_net" + ".txt"

    res = model.fit(
              x=input,y=output,batch_size=batch_size,
              epochs = epoch,
              validation_split=train_eval_ratio,
              callbacks=get_callbacks(model_file=path_w, logging_file=logging_file))

    print('Network Training Finished!')

    #evaluation_plot.plot(logging_file)



#%%
def test(cf):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['Training']['gpu_num'])
    patientsID=cf['Data']['patientsID_test']
    logging_file = cf['Paths']['model']

    if cf['Pretrained_Model']['path'] is not None:
        print(' Load pretrained model')
        model = tensorflow.keras.models.load_model(filepath=cf['Pretrained_Model']['path'])
        print(' Summary of the model:')
        model.summary()
    else:
        print('no pretrained  model')

    # get test Patient
    test_input, test_output = training_prepare.get_data_MR_2D(patientsID)
    test_output[0]=tensorflow.keras.utils.to_categorical(test_output[0],4)
    print('Evaluate on patients', patientsID)  
    #treatment_evaluation = model.evaluate(test_input, test_output[0], verbose=1)

    #print('Response loss:',treatment_evaluation)
    #print('Response loss:',treatment_evaluation[1])
    #print('Response loss:',treatment_evaluation[2])
    treatment_predict = model.predict(test_input)
    
    for i in range(len(patientsID)):
        print(patientsID[i])
        k=i*64
        print(treatment_predict[0][k])
        if treatment_predict[0][k][0]==1:
            print('Progress')
        elif treatment_predict[0][k][1]==1:
            print('Complete Response')
        elif treatment_predict[0][k][2]==1:
            print('Partial Response')
        elif treatment_predict[0][k][3]==1:
            print('Stable disease')
        print(treatment_predict[1][k])
        print(treatment_predict[2][k])

     # plot survival rate
    predicted_survivalRate=[]
    actual_survivalRate=[]
    for i in range(len(patientsID)):
        predicted_survivalRate.append(treatment_predict[1][i*64])
        actual_survivalRate.append(test_output[1][i*64])
     
     
    print(len(predicted_survivalRate))
    print(len(actual_survivalRate))
    print(len(patientsID))
     # use dataframe sort actual age and predicted age
    data = pd.DataFrame({'actual_survival_rate': actual_survivalRate,
              'regression_predicted_survival_rate': predicted_survivalRate,
              'patient_id': patientsID,
              })
     
    data.sort_values(by=['actual_survival_rate'], inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
     
     # plot
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
 

#%%
""" def data_preprocess(cf):
    if not os.path.exists(cf['Paths']['save']):
        os.makedirs(cf['Paths']['save'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(
                cf['Paths']['save']))
            if stop == 'n':
                return

    if not os.path.exists(cf['Paths']['model']):
        os.makedirs(cf['Paths']['model'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(
                cf['Paths']['model']))
            if stop == 'n':
                return

    if not os.path.exists(cf['Paths']['histories']):
        os.makedirs(cf['Paths']['histories'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(
                cf['Paths']['histories']))
            if stop == 'n':
                return

    print('-' * 75)
    print(' Config\n')
    print(' Local saving directory : ' + cf['Paths']['save'])

    # Copy train script and configuration file (make experiment reproducible)
    shutil.copy(os.path.basename(sys.argv[0]), os.path.join(cf['Paths']['save'], 'train.py'))

    shutil.copy(cf['Paths']['config'], os.path.join(cf['Paths']['save'], 'config_Age.yml'))

    shutil.copy('./util/generator.py', os.path.join(cf['Paths']['save'], 'generator.py'))
    shutil.copy('./get_train_eval_files.py', os.path.join(cf['Paths']['save'], 'get_train_eval_files.py'))
    shutil.copy('./network/ageNet.py', os.path.join(cf['Paths']['save'], 'network.py'))

    # Extend the configuration file with new entries
    with open(os.path.join(cf['Paths']['save'], 'config_Age.yml'), "w") as ymlfile:
        yaml.dump(cf, ymlfile) """

#%%
#if __name__ == '__main__':
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
parser = argparse.ArgumentParser(description='TreatmentNet training')

parser.add_argument('-c', '--config_path',
                    type=str,
                    default='treatmentresponse/config.yml',
                    help='Configuration file')
parser.add_argument('-e', '--exp_name',
                    type=str,
                    default='treatmentresponse',
                    help='Name of experiment')

arguments = parser.parse_args()

arguments.config_path = "treatmentresponse/config.yml"

assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                            ' -c pathname in the command line.'
assert arguments.exp_name is not None, 'Please provide a name for the experiment' \
                                        ' -e name in the command line'

# Parse the configuration file
with open(arguments.config_path, 'r') as ymlfile:
    cf = yaml.load(ymlfile)

# Set paths
cf['Paths']['save'] = arguments.exp_name
cf['Paths']['model'] = os.path.join(cf['Paths']['save'], 'model/')
cf['Paths']['histories'] = os.path.join(cf['Paths']['save'], 'histories/')
cf['Paths']['config'] = arguments.config_path

# create folder to store training results

if cf['Case'] == "train":
    #data_preprocess(cf)
    train(cf)
else:
    test(cf)
    
    

