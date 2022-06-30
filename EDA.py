#!/usr/bin/python3

"""
This script read a train data set specified in a configuration file and train a CNN.

"""
#Example to run the program


#nohup python3 Train_CNN.py WCD_Signals_w100_nu10GeV1TeV ./config/config_Train_nu_and_horizontal_80deg_w100.yml &



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('~/.matplotlib/matplotlibrc.bin')

mpl.use('agg')




import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

import random


def _get_available_gpus():  
    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

#Input files
if tf.__version__ != '1.5.0':
    #Use this if we are using our own computer
    tfback._get_available_gpus = _get_available_gpus


"""if (len(sys.argv) < 2):
    print("ERROR. No arguments were given to identify the run", file = sys.stderr)
    print("Please indicate the ID for the experiment and the yaml config file")
    sys.exit(1)


with open(sys.argv[2], 'r') as ymlfile:   
    cfg = yaml.safe_load(ymlfile)"""


#exp_ID = sys.argv[1]


#print("Experiment ID: ", exp_ID)
#print("Exp. Config.: ", cfg)


random_st=0 
seed = random_st
np.random.seed(seed)
random.seed(seed)
#tf.random.set_seed(seed)

PMTNUMBER = 1


#################################################################


#Define python functions

#Normalisation 

def NormalizeAllData(x):
    pmt = PMTNUMBER
    for j in range(x.shape[0]):
        x[j,:] = x[j,:]/np.sum(x[j,:])
    return x


def plot_simple_histogram(data1,plotname,name1,xlabel_title,n_bins=60,pos="right",density_option=True,log_option=False,plot_gaussian=False):
    #Set limits for histogram
    lim_inf = data1.min()-0.5*data1.min()
    lim_sup = data1.max()+0.05*data1.max()
    kwargs = dict(range=([lim_inf,lim_sup]), bins = n_bins, density=density_option)
    n1,x,_ = plt.hist(data1, **kwargs)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)
    plt.step(x[:-1],n1,color="black",where="post",label="Data counts")
    plt.autoscale(enable=True)
    plt.xlabel(xlabel_title)
    if density_option==True:
        plt.ylabel('Normalised counts')
    else:     
        plt.ylabel('Counts')
    #ax.set_xscale('log')
    if log_option==True: 
        ax.set_yscale('log')
    #Box with statistics
    textstr = '\n'.join((
    name1,
    'Entries = %2d' % (len(data1), ),
    'Mean = %4.2f' % (np.nanmean(data1), ),
    'Median = %4.2f' % (np.nanmedian(data1), ),
    'Std Dev = %4.2f' % (np.nanstd(data1), )))
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    if pos == "right":
        ax.text(x=0.74, y=0.78, s=textstr, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
    elif pos == "left":
        ax.text(x=0.02, y=0.98, s=textstr, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
    if plot_gaussian:
        (mu, sigma) = norm.fit(data1[(data1>lim_inf)*(data1<lim_sup)])
        y = norm.pdf(x[:-1], mu, sigma)
        l = plt.plot(x[:-1], y, 'r--', linewidth=2,label='Gaussian fit')
    #Set legend and labels
    ax.set_xlim((lim_inf,lim_sup))
    plt.locator_params(axis="x", nbins=7)
    ax.legend(loc = 'best', edgecolor="black",fontsize=20)
    fig.tight_layout()
    plt.savefig(plotname)
    plt.close()




########################################################################################################################


#Read Train data

#inputfilename = cfg['storage']['dataPath_datasets']+cfg['experiment']['train']+'.h5'

# Load the data for this tutorial
# - This dataset has 1M pulses from LZ sim data
# - Each pulse (object) has 17 features a one label that can be either 1=S1, 2=S2, 3=SE
#   - NOTE: further down labels are changed to 0=S1, 1=S2, 3=SE due to limitations in the to_categorical functions of Keras
filename_Xtrain = '/lstore/lattes/borjasg/DataScienceSchool/DataScience/input_2x_S2.txt'
filename_Ytrain = '/lstore/lattes/borjasg/DataScienceSchool/DataScience/output_2x_S2.txt'

#data = pd.read_csv(filename)
#data.describe()


#Xtrain = np.loadtxt(filename_Xtrain)
Ytrain = np.loadtxt(filename_Ytrain)


#traceLength = Xtrain.shape[1]



#print("Xtrain shape: ",Xtrain.shape)
print("Ytrain shape: ",Ytrain.shape)



######################################################################################




#Plots
name_to_plot = "./imgs/EDA/distance_Y_train.pdf"
name_label = r"$d_{\rm train}$"

plot_simple_histogram(data1=Ytrain,plotname=name_to_plot,name1="Train",xlabel_title=name_label)

