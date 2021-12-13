# -*- coding: utf-8 -*-

"""
Sample script to evaluate saved S1C on your own dataset

Reference: 
[1] P. K. Kota, Y. Pan, H. Vu, M. Cao, R. G. Baraniuk, and G. Bao, "The Need 
for Transfer Learning in CRISPR-CasOff-Target Scoring," Dec. 2021

bioRxiv: rev 2 updated December 2021; rev1 originally August 2021.
    
"""
import numpy as np
from tensorflow.keras.backend import clear_session
from sklearn.metrics import roc_auc_score

from siamcrispr import ml_data_utils, network_utils, custom_evaluation_functions


###
# Modify as needed
filename = (r'../Supplementary_tables/Table_S2_TrueOT_v1_1.xlsx') # TrueOT

#zero-indexed columns in file. See ml_data_utils.crispr_read_excel for details
colgRNA = 1 
colOT = 2 
colLabel = 4 # use None if no label available
skip_rows = 2 #number of rows to skip in xlsx file parsing
dropDuplicates=True
sheetName= 'TrueOT_v1.1_1211_reformat'
###

# pass excel info (column numbers and such)
seqDF = ml_data_utils.crispr_read_excel(filename, colgRNA, colOT, colLabel, \
                                        skipRows=skip_rows, sheetName=sheetName) 

# Main
if dropDuplicates:
    seqDF = seqDF.drop_duplicates()
    
modelFiles = []
model_ensemble = []    
clear_session() #speedup if running several times
modelChoice = 'scnn' 
modelClass = network_utils.SCNN 

coreModel = modelClass((4,26,1), kWidth=[6], numFilters=[8192])
preproc= ml_data_utils.PreprocessOneHot4D(max_length=26, strip_dash=True)

for cv in range(3):    
    model = coreModel.construct()
    model.load_weights('S1C\\S1C_k6_cv' + str(cv+1) + '.h5')
    model_ensemble.append(model)
print('Models loaded') 

# Preprocess    
all_gRNAs, all_OTs, all_labels, _, _= preproc.preprocess(seqDF, shuffle=False, groupGRNA=False) 
print('Data encoding complete')

Y_preds = np.zeros((seqDF.shape[0], 3))
for m in range(len(model_ensemble)): 
    Y_preds[:,m] = model_ensemble[m].predict([all_gRNAs, all_OTs])[:,0]
    print('Ensemble member #{} complete'.format(m+1))

print('S1C raw output available in Y_preds variable')

if colLabel is not None:     
    temp = np.unique(Y_preds)
    thresholds = np.sort(np.concatenate((temp-1e-6, temp+1e-6)))
    ROC_AUC = custom_evaluation_functions.custom_ROC_ensemble(seqDF['label'], Y_preds, thresholds)
    
    
    #Sanity check results. Invert scores for built-in ROC functions
    ROC_AUC_builtin = roc_auc_score(np.asarray(seqDF['label']).astype('int'), \
                                    -np.median(Y_preds, axis=1)) #verifies custom ROC method
        
    print('ROC_AUC: ' + str(ROC_AUC))
    
