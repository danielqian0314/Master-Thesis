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
from pylab import *

import tensorflow.keras
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import callbacks as cb
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint
)

from tensorflow.keras.layers import *

# Import own scripts
import training_prepare
import TreatmentResponseNet
import segmentationNet





#%%
def train_seg(cf):
    # parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['Training']['gpu_num'])
    train_eval_ratio = cf['Data']['train_val_split']
    batch_size = cf['Training']['batch_size']
    epoch = cf['Training']['num_epochs']
    num_slice_per_group = cf['Training']['num_slice_per_group']
    patientsID=np.load(data_path+'patientsCT.npy').astype(int)
    # get train data 
    
    input,output=training_prepare.get_image_CT_2D(patientsID)

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
        model, _ = segmentationNet.get_unet()

    
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

    

    path_w = cf['Paths']['model'] + "segmentation_net" + ".hdf5"
    logging_file = cf['Paths']['model'] + "segmentation_net" + ".txt"

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
     
    print('Evaluate on patients', patientsID)
    treatment_evaluation = model.evaluate(test_input, test_output, verbose=1)

    print('Response loss:',treatment_evaluation[0])
    print('Response loss:',treatment_evaluation[1])
    print('Response loss:',treatment_evaluation[2])

#%%
   """  # get patient name for plot


    # convert predicted age to float
    predicted_survivalRate = []
    for i in range(len(age_prediction)):
        total_predicted_age_regresion.append(float(age_prediction[i]))
    print(total_predicted_age_regresion)

    residual = np.abs(np.array(predicted_survivalRate) - np.array(test_age))

    # use dataframe sort actual age and predicted age
    data = pd.DataFrame({'actual_age': test_age,
                         'regression_predicted_age': total_predicted_age_regresion,
                         'patient_id': total_patient_name,
                         'resudial': residual
                         })

    data.sort_values(by=['actual_age'], inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)

    # plot
    fig = plt.figure(figsize=(18, 15))
    plt.plot(data['regression_predicted_age'], marker='*', label='predicted age')
    plt.plot(data['actual_age'], marker='x', label='actual_age', )

    plt.xticks(arange(len(total_patient_name)), data['patient_id'], rotation=60)
    plt.xlabel("Patient Number")
    plt.ylabel("Age")
    plt.legend()
    plt.grid()
    plt.gcf().subplots_adjust(bottom=0.15)
    std = np.std(np.array(data['regression_predicted_age']) - np.array(data['actual_age']))
    mae = np.mean(np.abs(np.array(data['regression_predicted_age']) - np.array(data['actual_age'])))
    plt.title("Predicted age vs. Actual age, Std:{:0.2f}, MAE:{:0.2f}".format(std, mae))
    plt.show()
    save_file = logging_file + "predictedage.png"
    plt.savefig(save_file)
    print('save successful') """

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
parser = argparse.ArgumentParser(description='TreatmentNet training')

parser.add_argument('-c', '--config_path',
                    type=str,
                    default='config/config.yml',
                    help='Configuration file')

parser.add_argument('-e', '--exp_name',
                    type=str,
                    default=None,
                    help='Name of experiment')

arguments = parser.parse_args()

arguments.config_path = "config/config.yml"

assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                            ' -c pathname in the command line.'
assert arguments.exp_name is not None, 'Please provide a name for the experiment' \
                                        ' -e name in the command line'

# Parse the configuration file
with open(arguments.config_path, 'r') as ymlfile:
    cf = yaml.load(ymlfile)

# Set paths
cf['Paths']['save'] = 'exp/' + arguments.exp_name
cf['Paths']['model'] = os.path.join(cf['Paths']['save'], 'model/')
cf['Paths']['histories'] = os.path.join(cf['Paths']['save'], 'histories/')
cf['Paths']['config'] = arguments.config_path

# create folder to store training results

if cf['Case'] == "train":
    #data_preprocess(cf)
    train(cf)
elif cf['Case']=="train_seg":
    train_seg(cf)
else:
    test(cf)