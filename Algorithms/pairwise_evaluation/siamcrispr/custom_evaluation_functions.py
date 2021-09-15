# -*- coding: utf-8 -*-
"""
ROC-AUC and PR-AUC analysis for Siamese models.
Custom functions needed for ensembling and ease of interpretability with 'distance'-based outputs
- large distance (raw model output) is more dissimilar between gRNA and target, or more likely to be a negative label
- rather than negating model outputs (negative distances), created these functions to avoid confusion

Ensemble functions work with non-ensembles too (just use one-column inputs)

Reference: 
[1] P. K. Kota, Y. Pan, H. Vu, M. Cao, R. G. Baraniuk, and G. Bao, "The Need 
for Transfer Learning in CRISPR-CasOff-Target Scoring," Aug. 2021.

bioRxiv: 
"""

import numpy as np
from sklearn.metrics import auc

def custom_ROC_ensemble(y_true, y_pred_prob, thresholds, plotCheck=False):     
    """ y_pred_prob should be (datapoints) x (number of models) """    
    fprs = np.zeros(len(thresholds))
    tprs = np.zeros(len(thresholds))

    y = np.copy(y_pred_prob)
    y = (y_pred_prob[:,:,None] < thresholds).astype(int)
    y = np.round((np.sum(y, axis=1) / y.shape[1]))
    
    
    TPFN = y[y_true==1,:] == y_true[y_true==1,None]
    TP = np.sum(TPFN, axis=0)
    FN = TPFN.shape[0] - TP
    TNFP = y[y_true==0,:] == y_true[y_true==0,None]
    TN = np.sum(TNFP, axis=0)
    FP = TNFP.shape[0] - TN
    
    tprs = TP / (TP+FN)
    fprs = FP / (FP+TN)
            
    if plotCheck:
        return auc(fprs,tprs), fprs, tprs
    else: 
        return auc(fprs,tprs) 


def pvr_auc_ensemble(y_true, y_pred_prob, thresholds, saveCheck = False, plotCheck=False, path=''):

    pr = np.zeros(len(thresholds))
    rec = np.zeros(len(thresholds))

    y = np.copy(y_pred_prob)
    y = (y_pred_prob[:,:,None] < thresholds).astype(int)
    y = np.round((np.sum(y, axis=1) / y.shape[1]))
    
    
    TPFN = y[y_true==1,:] == y_true[y_true==1,None]
    TP = np.sum(TPFN, axis=0)
    FN = TPFN.shape[0] - TP
    TNFP = y[y_true==0,:] == y_true[y_true==0,None]
    TN = np.sum(TNFP, axis=0)
    FP = TNFP.shape[0] - TN
    
    pr = TP / (TP+FP)
    rec = TP / (TP+FN)
    pr[np.isnan(pr)] = 1 #seems to be by definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve    
        
    
    #pdb.set_trace()
    if plotCheck:
        return auc(rec,pr), pr, rec
    else: 
        return auc(rec,pr) 
     

