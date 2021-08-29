# -*- coding: utf-8 -*-
"""
Sample training script to guide future development. 

Reference: 
[1] P. K. Kota, Y. Pan, H. Vu, M. Cao, R. G. Baraniuk, and G. Bao, "The Need 
for Transfer Learning in CRISPR-CasOff-Target Scoring," Aug. 2021.

bioRxiv: 
    
"""

import numpy as np
import random 
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping as ES
from numpy.random import seed
from tensorflow.random import set_seed
from sklearn.metrics import roc_auc_score

#Custom modules
from siamcrispr import ml_data_utils, custom_evaluation_functions, network_utils

def get_files(data_directory): 
    """ for aggregating multiple files into a training/cv set or external set
    """
    allfiles = os.listdir(data_directory)
    dataFilepaths = []
    for f in range(len(allfiles)): 
        filesplit = allfiles[f].split('.')
        if  len(filesplit) > 1 and filesplit[1] == 'pkl':   
            dataFilepaths.append(data_directory + '\\' + allfiles[f])
    return dataFilepaths

"""
Basic inputs
"""
rngseed=2 # for shuffling dataset before CV splitting
testPct = 0.20 # on proxy dataset. [0,1] - referred to as "Proxy Validation" in [1]
batch_size = 128
num_epochs = 10000 # go until searly stopping trigger
patienceSetting = 10 # early stoppping
cv = 5
lr0 = 1e-5
optim = RMSprop(learning_rate=lr0)
earlystopper = ES(monitor = 'val_loss', patience = patienceSetting, verbose = 1, mode = 'min', restore_best_weights = True) 

localData = get_files(r'parsed_datasets\proxy_dataset') 
    # Proxy Dataset [1]: merged CRISTA training set and Wang 2020 review dataset, excluding datapoints
    # with TrueOT gRNAs
           
# Preprocessor settings
downsample = None
posProp = None

# Model settings
modelTrain = True #whether or not the model's trained (e.g., S1C_ut)
modelSave = [False, 'train_demo_']
margin=1 # contrastive_loss setting

modelChoice = network_utils.SCNN # all Siamese nets used in [1] 
#modelChoice = network_utils.LearnedDistanceSCNN # Hybrid
seeds=[1] # initializations

"""
Data Preprocessing
"""

maxLength = 26 #overrode in [1] 

preprocessorLocal = ml_data_utils.PreprocessOneHot4D(max_length=maxLength, num_downsample=downsample, pos_prop=posProp, strip_dash=True, threshold=0)
preprocessorExternal = ml_data_utils.PreprocessOneHot4D(max_length=maxLength, strip_dash=True, threshold=0)

# Load the local datasets
for f in range(len(localData)): 
    print('preprocessing local set # ' + str(f+1))
    gtemp, otemp, ltemp, input_shape, raw_data = preprocessorLocal.preprocess(localData[f],rngseed=rngseed)
    if f == 0: 
        gRNAs, OTs, labels = (gtemp, otemp, ltemp)
        all_local_raw = raw_data
    else: 
        gRNAs = np.concatenate((gRNAs, gtemp), axis=0)
        OTs = np.concatenate((OTs, otemp), axis=0)
        labels = np.concatenate((labels, ltemp), axis=0)
        all_local_raw.append(raw_data)
 
if gRNAs.shape[0] != OTs.shape[0] or gRNAs.shape[0] != np.size(labels): 
    raise ValueError('Data size mismatch')

cvMasks, testMask = ml_data_utils.split_by_gRNAs(all_local_raw, testPct, kfold=cv, rngseed=rngseed, split_size_tol=1)

#Sanity check - confirm that indexing is correct (grouping by gRNAs alphabetizes, rngseeds must be the same, etc. lots of potential for error)
totalGRNA = np.unique(gRNAs, axis=0).shape[0]
totalSplits = np.unique(gRNAs[testMask], axis=0).shape[0]
for k in range(cvMasks.shape[1]): 
    totalSplits += np.unique(gRNAs[cvMasks[:,k]], axis=0).shape[0]
if totalGRNA != totalSplits: 
    raise ValueError('gRNA numbers mismatch. Data split is not keeping unique guides separate')
gR_test = gRNAs[testMask]
OT_test = OTs[testMask]
Y_test = labels[testMask]


# for visualization - see gRNA split between folds
gRtrainFold = np.zeros((cv,2))
gRvalFold = np.zeros((cv, 2))
for i in range(cv): 
    gRvalFold[i,0] = np.unique(gRNAs[cvMasks[:,i]], axis=0).shape[0]
    gRvalFold[i,1] = np.sum(cvMasks[:,i])
    
    trainInds = (np.sum(cvMasks, axis=1) - cvMasks[:,i]).astype('bool')
    gRtrainFold[i,0] = np.unique(gRNAs[trainInds], axis=0).shape[0]        
    gRtrainFold[i,1] = np.sum(trainInds)

"""
Model Definition and Training
"""
coreModel = modelChoice(input_shape, numFilters=[16384], kWidth=[8]) # e.g., for S1C


if modelTrain: 
    allData = np.zeros([0,7]) # mean/stdev ROC-AUC for train, cv, and test ("Proxy Validation") + ROC-AUC of the ensemble
else: 
    allData = np.zeros((0, 1)) #just the ensemble


if not modelTrain: 
    ROCs = np.zeros((len(seeds), 1))
        
for s in range(len(seeds)):     
    seed(seeds[s]) 
    set_seed(seeds[s])
    random.seed(seeds[s])    
    
    """
    Training with cross-validation and evaluation on test set
    """    
    roc_cv = [0]*cv
    counter = 1
    
    if modelTrain: 
        ROCs = np.zeros([cv, 3])        
        cvc=1
                
        for k in range(cv): 
            valInds = cvMasks[:,k]
            trainInds = (np.sum(cvMasks, axis=1) - cvMasks[:,k]).astype('bool')
            gR_train = gRNAs[trainInds]
            gR_val = gRNAs[valInds]
            OT_train = OTs[trainInds]
            OT_val = OTs[valInds]
            Y_train = labels[trainInds]
            Y_val = labels[valInds]
            
            #treating imbalanced dataset
            wt0 = int(np.sum(Y_train == 1)/ np.size(Y_train) *100)
            wt1 = 100-wt0
            class_weight = {0:wt0, 1:wt1}
                                
            model = coreModel.construct()                                    
            model.compile(loss=network_utils.get_contrastive_loss(margin=margin), optimizer=optim)                
            
            print(model.summary())                
            
            cvc+=1
            
            Y_train = np.asarray(Y_train).astype('float32') # TF 2
            Y_val = np.asarray(Y_val).astype('float32') # TF 2
            
            model.fit([gR_train, OT_train], Y_train, batch_size=batch_size, epochs=num_epochs, validation_data=([gR_val, OT_val], Y_val), class_weight = class_weight, callbacks=[earlystopper])        
        
            Y_pred_train = model.predict([gR_train, OT_train])
            Y_pred_val = model.predict([gR_val, OT_val])
            
            if gR_test.size>0:
                Y_pred_test = model.predict([gR_test, OT_test])
            else:
                Y_pred_test = np.array([])
            Y_preds = [Y_pred_train, Y_pred_val, Y_pred_test]
            Y_real = [Y_train, Y_val, Y_test]
                        
            #Evaluate prediction ROCs for individual models
            for d in range(len(Y_preds)): #eval on different datasets    
                if len(Y_preds[d]) != 0: 
                    ROCs[counter-1,d] = 1 - roc_auc_score(Y_real[d].astype('bool'), Y_preds[d][:,0]) 
                                    
            if counter == 1:
                Y_ens_test = Y_preds[2]
            else:            
                Y_ens_test = np.concatenate((Y_ens_test, Y_preds[2]), axis=1)
                    
            if modelSave[0]: 
                coreFilename =  modelSave[1] + '_cv' + str(counter) + '_seed' + str(seeds[s])                                             
                model.save_weights(coreFilename + '_weights.h5')             
                
            counter = counter + 1
        #end CV for loop
        
        #Can do whatever from here (e.g. save to a file, format properly, etc.)
        meanROCs = np.mean(ROCs, axis=0)
        stdROCs = np.std(ROCs, axis=0)    
        roc_summary = np.zeros([1, 6])
        roc_summary[0][np.arange(3)*2] = meanROCs
        roc_summary[0][np.arange(3)*2 + 1] = stdROCs
                    
    
        # ensemble performance on test
        temp = np.unique(Y_ens_test)
        thresholds = np.sort(np.concatenate((temp-1e-6, temp+1e-6))) #ensure no intermediate threshold is missed

        ROCensemble = custom_evaluation_functions.custom_ROC_ensemble(Y_real[2], Y_ens_test, thresholds)
        roc_summary = np.append(roc_summary, ROCensemble)[None,:]
        
        allData = np.vstack((allData,roc_summary))
        print(roc_summary)
        print(allData)    
  
    
    else: # not trained - just immediately initialize/evaluate
        model = coreModel.construct()            
        model.compile(loss=network_utils.get_contrastive_loss(margin=margin), optimizer=optim)
        Y_pred_local = model.predict([gRNAs, OTs])
        Y_real = [labels]
        Y_preds = [Y_pred_local]
        
        roc_summary = np.zeros((1, len(Y_preds)))
        for d in range(len(Y_preds)): #eval on different datasets    
            thresholds = np.linspace(0,1.01*max(Y_preds[d]),1000).tolist()             
            roc_summary[0,d] = custom_evaluation_functions.custom_ROC(Y_real[d], Y_preds[d], thresholds, plotCheck = 0, saveCheck=0)
        
        if modelSave[0]: 
            coreFilename =  modelSave[1] + '_cv' + str(counter) + '_seed' + str(seeds[s]) 
            model.save_weights(coreFilename + '_untrained_weights.h5')
                        
        allData = np.vstack((allData,roc_summary))
            
print(allData)