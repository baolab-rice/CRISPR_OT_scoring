"""
Pairwise comparisons
This script generates all baseline data reported in Table 2 and 3 in our paper [1]. It also generates
S1C results for the selected initialization that's available in custom_scoring\S1C. 

Significance was based on the performance of the S1C with 5 different random initialization seeds, 
whereas here we only regenerate the data for the chosen seed. Raw model files for the other seeds
are available upon request, but all seeds performed extremely similarly (low standard deviations in ROC-AUC and PR-AUC)

Reference: 
[1] P. K. Kota, Y. Pan, H. Vu, M. Cao, R. G. Baraniuk, and G. Bao, "The Need 
for Transfer Learning in CRISPR-CasOff-Target Scoring," Aug. 2021.

bioRxiv: 
    
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from siamcrispr import network_utils, ml_data_utils, custom_evaluation_functions


def pr_auc(y_true, y_preds): 
    prec, rec, _ = precision_recall_curve(y_true, y_preds)    
    return auc(rec, prec)


seqFiles = r'TrueOText_v10+allscores_format.xlsx'
maskFile = r'TrueOText_pairs_overlap_mask_loop.xlsx'

with open(r'21-05-26_trueOTv10_indelMask.pkl', 'rb') as f: 
    indelMask = pickle.load(f)
indelMask = indelMask.astype('bool')    
sheetName= 'TrueOText_v10+allscores_format'

col_g = 1
col_OT = 2
col_label = 5
removeDuplicates=True #based on {gRNA,OT,label} unique triplets

#maskCompare = ['Cropit', 'Hsu', 'CCTOP', 'COSMID', 'MIT', 'CNN_std', 'elevation', 'CFD', 'predictCRISPR', 'CRISTA', 'Net_ori']
maskCompare = ['Cropit', 'Hsu', 'CCTOP', 'MIT', 'CFD','COSMID', 'CNN_std', 'elevation',  'predictCRISPR', 'CRISTA', 'Net_ori']
all_labels = [0]

# S1C comparison: 
lf =  network_utils.get_contrastive_loss()# Loss function 
modelPreproc = ml_data_utils.PreprocessOneHot4D(max_length=26, strip_dash=True)
cv = 5
modelDir = r'..\\..\\custom_scoring\\S1C'
coreModel =  network_utils.SCNN((4,26,1), numFilters=[16384], kWidth=[8])
modelFiles = []
for i in range(cv):                                 
    modelFiles.append(modelDir + '\\S1C_cv' + str(i+1))


df = pd.read_excel (seqFiles, sheet_name=sheetName)
currentData = df.iloc[:,[col_g, col_OT, col_label]]    
colNames = list(currentData)
currentData = currentData.rename(columns={colNames[0]: 'gRNA', colNames[1]:'OT', colNames[2]:'label'})    

if removeDuplicates: 
    dupMaskBool = np.invert(np.asarray(currentData.duplicated()))
    dupMask = np.where(dupMaskBool==True)

all_gRNAs, all_OTs, all_labels1, _, _= modelPreproc.preprocess(currentData, shuffle=False, groupGRNA=False) 
all_labels = np.asarray(df['TrueOT'])
        
# Evaluate masks - assume on all files 
df = pd.read_excel(seqFiles, sheet_name=sheetName)
datapoints = []

# Compare over all, mm only, indel
ROC_compare = np.zeros((len(maskCompare), 3)) 
PVR_compare = np.zeros((len(maskCompare), 3))
datapoints = np.zeros((len(maskCompare), 3))


for m in range(len(maskCompare)):  
    predTemp = df[maskCompare[m]]
    mask = np.invert(np.asarray(predTemp.isnull())) # datapoints that were actually able to be scored        
    try: 
        gRoverlapMask = np.invert(np.asarray(pd.read_excel(maskFile, sheet_name=maskCompare[m]))[:,0])        
        if gRoverlapMask.size != mask.size:
            raise ValueError('Masks of different length')
    except: 
        gRoverlapMask = np.ones( (np.size(mask))).astype(bool) 
    finalMask = np.logical_and(mask, gRoverlapMask) # datapoints with no gR overlap *and* were actually scored by the competing algorithm
    
    if removeDuplicates: 
        finalMask = np.logical_and(finalMask, dupMaskBool)
    
    finalIndelMask = np.logical_and(finalMask, indelMask)
    finalMmMask = np.logical_and(finalMask, np.invert(indelMask))
    
            
    Y_preds = np.asarray(df[maskCompare[m]])
    
    
    ROC_compare[m, 0] = roc_auc_score(all_labels[finalMask], Y_preds[finalMask])
    PVR_compare[m,0] = pr_auc(all_labels[finalMask], Y_preds[finalMask])

    if sum(finalIndelMask) == 0: 
        ROC_compare[m,1] = ROC_compare[m,0]
        PVR_compare[m,1] = PVR_compare[m,0]
    else: 
        ROC_compare[m,1] = roc_auc_score(all_labels[finalMmMask], Y_preds[finalMmMask])
        ROC_compare[m,2] = roc_auc_score(all_labels[finalIndelMask], Y_preds[finalIndelMask])
        
        PVR_compare[m,1] = pr_auc(all_labels[finalMmMask], Y_preds[finalMmMask])
        PVR_compare[m,2] = pr_auc(all_labels[finalIndelMask], Y_preds[finalIndelMask])

    datapoints[m,0] = sum(finalMask)
    datapoints[m,1] = sum(finalMmMask)
    datapoints[m,2] = sum(finalIndelMask)
    
    print('{} subset complete'.format(maskCompare[m]))

#Evaluate S1C on the same subsets of data
Y_preds = np.zeros((np.size(all_labels), len(modelFiles)))    
for m in range(len(modelFiles)):         
    model = coreModel.construct()
    model.load_weights(modelFiles[m] + '.h5')        
    
    Y_preds[:,m] = model.predict([all_gRNAs, all_OTs])[:,0]

temp = np.unique(Y_preds)
thresholds = np.sort(np.concatenate((temp-1e-6, temp+1e-6)))    
#thresholds = np.linspace(0,1.01*np.max(Y_preds),1000).tolist()  
#np.unique(np.convolve(np.sort(Y_preds[:,0]), np.ones(2), 'valid')/2)
ROC_AUCs = custom_evaluation_functions.custom_ROC_ensemble(all_labels, Y_preds, thresholds)
PVR_AUCs = custom_evaluation_functions.pvr_auc_ensemble(all_labels, Y_preds, thresholds)


print('Eval of S1C on full TrueOT complete')


# Evaluate masks - assume on all files 
df = pd.read_excel(seqFiles, sheet_name=sheetName)
datapoints = []


ROC_model_compare = np.zeros((len(maskCompare), 3))
PVR_model_compare = np.zeros((len(maskCompare), 3))
datapoints = np.zeros((len(maskCompare), 3))

for m in range(len(maskCompare)):  
    predTemp = df[maskCompare[m]]
    mask = np.invert(np.asarray(predTemp.isnull())) # datapoints that were actually able to be scored
        
    try: 
        gRoverlapMask = np.invert(np.asarray(pd.read_excel(maskFile, sheet_name=maskCompare[m]))[:,0])
        #gRoverlapMask = pwMaskTable[:,m]
        if gRoverlapMask.size != mask.size:
            raise ValueError('Masks of different length')
    except: 
        gRoverlapMask = np.ones( (np.size(mask))).astype(bool) 
    finalMask = np.logical_and(mask, gRoverlapMask) # datapoints with no gR overlap *and* were actually scored by the competing algorithm    
    
    if removeDuplicates: 
        finalMask = np.logical_and(finalMask, dupMaskBool)
    
    finalIndelMask = np.logical_and(finalMask, indelMask)
    finalMmMask = np.logical_and(finalMask, np.invert(indelMask))
    
    datapoints[m,0] = sum(finalMask)
    datapoints[m,1] = sum(finalMmMask)
    datapoints[m,2] = sum(finalIndelMask)
    
        
                    
    if Y_preds.shape[1] == 1: # not an actual ensemble
        ROC_model_compare[m,0] = roc_auc_score(all_labels[finalMask], -Y_preds[finalMask,0])
        PVR_model_compare[m,0] = pr_auc(all_labels[finalMask], -Y_preds[finalMask,0])
        ROC_model_compare[m,1] = roc_auc_score(all_labels[finalMmMask], -Y_preds[finalMmMask,0])
        PVR_model_compare[m,1] = pr_auc(all_labels[finalMmMask], -Y_preds[finalMmMask,0])
        if np.sum(finalIndelMask) != 0: 
            ROC_model_compare[m,2] = roc_auc_score(all_labels[finalIndelMask], -Y_preds[finalIndelMask,0])
            PVR_model_compare[m,2] = pr_auc(all_labels[finalIndelMask], -Y_preds[finalIndelMask,0])
    else: 
        temp = np.unique(np.median(Y_preds, axis=1)) # above which the ensemble goes from 0->1 on its voting
        thresholds = np.sort(np.concatenate((temp-1e-6, temp+1e-6)))
        
        ROC_model_compare[m, 0] = custom_evaluation_functions.custom_ROC_ensemble(all_labels[finalMask], Y_preds[finalMask,:], thresholds)
        PVR_model_compare[m, 0] = custom_evaluation_functions.pvr_auc_ensemble(all_labels[finalMask], Y_preds[finalMask,:], thresholds)
        ROC_model_compare[m, 1] = custom_evaluation_functions.custom_ROC_ensemble(all_labels[finalMmMask], Y_preds[finalMmMask,:], thresholds)
        PVR_model_compare[m, 1] = custom_evaluation_functions.pvr_auc_ensemble(all_labels[finalMmMask], Y_preds[finalMmMask,:], thresholds)
        ROC_model_compare[m, 2] = custom_evaluation_functions.custom_ROC_ensemble(all_labels[finalIndelMask], Y_preds[finalIndelMask,:], thresholds)
        PVR_model_compare[m, 2] = custom_evaluation_functions.pvr_auc_ensemble(all_labels[finalIndelMask], Y_preds[finalIndelMask,:], thresholds)
                
    print('S1C on {} subset complete'.format(maskCompare[m]))
    
#
print('For Tables 2 and 3 in [1], see variables ''ROC_model_compare'', ''ROC_compare'', ''PVR_model_compare'', ''PVR_compare'', and ''datapoints''.')
print("These are S1C ROC-AUC, baseline ROC-AUC, S1C PR-AUC, baseline PR-AUC, and n (datapoints) respectively")
print("Note the S1C values will be close but not exact to the values in Tables 2 and 3 since those Tables report an average over 5 seeds, whereas only one seed is loaded in this script.")
print('Column 1 is on the entire subset. Column 2 is on ``Other'' (non-bulge) datapoints within the subset. Column 3 is on bulge datapoints within the subset')