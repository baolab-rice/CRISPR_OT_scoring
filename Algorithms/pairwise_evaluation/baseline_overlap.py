from copy import deepcopy
import pandas as pd
import pdb
import numpy as np
import pickle
from siamcrispr import ml_data_utils


# determine the extent of overlap betwee TrueOTExtended and training gRNAs from some of the existing framework
def lenRange(RNAlist):
    maxLength = 0
    minLength = 9999
    lenList = []
    for gRNA in RNAlist:
        lenList.append(len(gRNA))
#        if len(gRNA) != 23:
#            print(gRNA)
        if len(gRNA) > maxLength:
            maxLength = len(gRNA)
        if len(gRNA) < minLength:
            minLength = len(gRNA)
    lenSet = set(lenList)
    return minLength, maxLength, lenSet




# Load in TrueOT for mask generation /comparison
trueOT = ml_data_utils.crispr_read_excel('TrueOT_v1_1_allscores.xlsx', 1, 2, colLabel=4, sheetName='TrueOT', skipRows=2)
TrueOT_extended_gRNAs = np.array(trueOT['gRNA'])
OT = np.array(trueOT['OT'])
labels = np.array(trueOT['label'])
    
TrueOT_extended_gRNAs = list(set(TrueOT_extended_gRNAs))
print("TrueOT gRNA count (including PAM) after removing duplication: ", len(TrueOT_extended_gRNAs))
print("Length of TrueOT: ", lenRange(TrueOT_extended_gRNAs))


with open('../../custom_scoring/parsed_datasets/proxy_dataset/Proxy_Dataset.csv', 'rb') as file:
    proxyDataset = pd.read_csv(file)
proxy_gRNA = proxyDataset['gRNA']
proxy_gRNAs_noPAM = []
for i in range(proxy_gRNA.shape[0]):
    proxy_gRNAs_noPAM.append(proxy_gRNA[i][:20])


#CRISPR-NET data. Authors use I-n and II-n notation for various datsets. 
# Using II_4 for elevation's mask as well (see notes below))
A = pd.read_pickle(r'training_sets\CRISPR-NET\cd33 (dataset II-1).pkl')
II_1 = list(A[0]['WTSequence'])    # CFD data
A = pd.read_pickle(r'training_sets\CRISPR-NET\hmg_data (dataset II-2).pkl') #- Haeussler
II_2 = list(A['30mer'])
A = pd.read_pickle(r'training_sets\CRISPR-NET\guideseq_data (dataset II-4).pkl') #Tsai 2015 guide-seq
II_4 = list(A['20mer'])


#CRISPOR = Haeussler. Used for CNNstd training
CNNstd_gRNAs, OT, labels = ml_data_utils.load_pkl_data(r'training_sets\CNN_std\CRISPOR_parsed.pkl',888)
CNNstd_gRNAs = list(set(CNNstd_gRNAs))
CNNstd_gRNAs_noPAM = []
glens = []
for gRNA in CNNstd_gRNAs:
    CNNstd_gRNAs_noPAM.append(gRNA[:20])
    glens.append(len(gRNA))
CNNstd_gRNAs_noPAM = list(set(CNNstd_gRNAs_noPAM))
print("CNNstd gRNA count after removing duplication: ", len(CNNstd_gRNAs))
print("CNNstd gRNA no PAM count after removing duplication: ", len(CNNstd_gRNAs_noPAM))
print("Length of CNNstd: ", lenRange(CNNstd_gRNAs))


#Verified, straight from CRISTA_Train (they posted explicit training set)
CRISTA_gRNAs, OT, labels = ml_data_utils.load_pkl_data(r'training_sets\CRISTA\20-07-07_CRISTA_train_thresh0.pkl',888)
CRISTA_gRNAs = list(set(CRISTA_gRNAs))
CRISTA_gRNAs_noPAM = []
for gRNA in CRISTA_gRNAs:
    CRISTA_gRNAs_noPAM.append(gRNA[:20])
CRISTA_gRNAs_noPAM = list(set(CRISTA_gRNAs_noPAM))
print("CRISTA gRNA count after removing duplication: ", len(CRISTA_gRNAs))
print("CRISTA gRNA no PAM count after removing duplication: ", len(CRISTA_gRNAs_noPAM))
print("Length of CRISTA: ", lenRange(CRISTA_gRNAs))

#CD33 data
CD33 = pd.read_excel(r'training_sets\elevation\CD33_data_postfilter.xlsx', sheet_name='CD33_data_postfilter', skiprows=0)
CD33_gRNAs = CD33.iloc[:, [2]]
CD33_gRNAs = CD33_gRNAs['WTSequence'].tolist()
CD33_only_noPAM = []
for gRNA in CD33_gRNAs:
    if len(gRNA) != 21: #skip all truncations #pk mod
        CD33_only_noPAM.append(gRNA[:20])


elevation_gRNAs = []
elevation_gRNAs.extend(CD33_gRNAs)
elevation_gRNAs.extend(II_4) # Tsai GUIDE-SEQ
elevation_gRNAs = list(set(elevation_gRNAs))
elevation_gRNAs_noPAM = []
for gRNA in elevation_gRNAs:
    if len(gRNA) != 21: #skip all truncations #pk mod
        elevation_gRNAs_noPAM.append(gRNA[:20])
elevation_gRNAs_noPAM = list(set(elevation_gRNAs_noPAM))
print("Elevation gRNA count after removing duplication: ", len(elevation_gRNAs))
print("Elevation gRNA no PAM count after removing duplication: ", len(elevation_gRNAs_noPAM))
print("Length of Elevation: ", lenRange(elevation_gRNAs))


CRISPR_NET_gRNAs = []
#"optimal combination" noted in paper is I/1, II/1, II/2, II/3, II/4
#CircleSeq is dataset I/1 in Lin 2020 (CRISPR-NET)
CircleSeq = pd.read_csv(r'training_sets\CRISPR-NET\CIRCLE_seq_10gRNA_wholeDataset.csv') 
CircleSeq_gRNAs = list(CircleSeq['sgRNA_seq'])
for i in range(len(CircleSeq_gRNAs)): 
    if len(CircleSeq_gRNAs[i]) != 21: #skip all truncations #pk mod
        CircleSeq_gRNAs[i] = CircleSeq_gRNAs[i].replace('_', '')
        CircleSeq_gRNAs[i] = CircleSeq_gRNAs[i].replace('-', '')    
        CircleSeq_gRNAs[i] = CircleSeq_gRNAs[i]
CircleSeq_gRNAs = list(set(CircleSeq_gRNAs))
CRISPR_NET_gRNAs.extend(CircleSeq_gRNAs)


#This is II/3 - SITESEQ
siteseq_gRNAs = ['GGGGCCACTAGGGACAGGATTGG', 'GGGTGGGGGGAGTTTGCTCCTGG', 'GCAAAACTCAACCCTACCCCAGG'
                 'CTCGTCTGATAAGACAACAGTGG', 'GGAATCCCTTCTGCAGCACCTGG', 'ATAGGAGAAGATGATGTATAGGG',
                 'GCATACAGTGATTTGATGAAAGG', 'GCTGATGTAGTCACTCTTGAGGG', 'GGTGGACAAGCGGCAGATAGCGG']
CRISPR_NET_gRNAs.extend(siteseq_gRNAs)

#added in II/1, II/2, II/4
CRISPR_NET_gRNAs.extend(II_1)
CRISPR_NET_gRNAs.extend(II_2) # Haeussler as used in CRISPR-NET
CRISPR_NET_gRNAs.extend(II_4) # II/4 from their source

CRISPR_NET_gRNAs = list(set(CRISPR_NET_gRNAs))
CRISPR_NET_gRNAs_noPAM = []
glens = []
for gRNA in CRISPR_NET_gRNAs:
    if len(gRNA) == 20 or len(gRNA) == 23: #pk mod
        CRISPR_NET_gRNAs_noPAM.append(gRNA[:20])
    if len(gRNA) == 21: #pk mod
        CRISPR_NET_gRNAs_noPAM.append(gRNA[:-3])
        
    elif len(gRNA) == 46:
        CRISPR_NET_gRNAs_noPAM.append(gRNA[:20])
        CRISPR_NET_gRNAs_noPAM.append(gRNA[23:-3])
    glens.append(len(gRNA))
    
    
glens = np.unique(glens)
CRISPR_NET_gRNAs_noPAM = list(set(CRISPR_NET_gRNAs_noPAM))
print("CRISPR-NET gRNA count after removing duplication: ", len(CRISPR_NET_gRNAs))
print("CRISPR-NET  no PAM count after removing duplication: ", len(CRISPR_NET_gRNAs_noPAM))
print("Length of CRISPR-NET : ", lenRange(CRISPR_NET_gRNAs))


# Peng 2018 (predictCRISPR)
Peng2018_gRNAs = []
Peng2018hight_gRNAs, OT, labels = ml_data_utils.load_pkl_data(r'training_sets\predictCRISPR\combined_Peng2018_neg_highthroughput.pkl',888)
Peng2018hight_gRNAs = list(set(Peng2018hight_gRNAs))
print("Peng2018 high throughput gRNA count after removing duplication: ", len(Peng2018hight_gRNAs))
Peng2018_gRNAs.extend(Peng2018hight_gRNAs)

Peng2018lowt_gRNAs, OT, labels = ml_data_utils.load_pkl_data(r'training_sets\predictCRISPR\combined_Peng2018_neg_lowthroughput.pkl',888)
Peng2018lowt_gRNAs = list(set(Peng2018lowt_gRNAs))
print("Peng2018 gRNA lowthroughput count after removing duplication: ", len(Peng2018lowt_gRNAs)) #pk mod the comment from high->lowthroughput
Peng2018_gRNAs.extend(Peng2018lowt_gRNAs)

Peng2018_gRNAs = list(set(Peng2018_gRNAs))
Peng2018_gRNAs_noPAM = []
glens = []
for gRNA in Peng2018_gRNAs:
    Peng2018_gRNAs_noPAM.append(gRNA[:20])
    glens.append(len(gRNA))
glens_Peng = np.unique(glens)
Peng2018_gRNAs_noPAM = list(set(Peng2018_gRNAs_noPAM))
print("Peng2018 gRNA count after removing duplication: ", len(Peng2018_gRNAs))
print("Peng2018 gRNA no PAM count after removing duplication: ", len(Peng2018_gRNAs_noPAM))
print("Length of Peng 2018: ", lenRange(Peng2018_gRNAs))

proxy_in_TrueOT = []
CNNstd_in_TrueOT = []
elevation_in_TrueOT = []
CRISPR_NET_in_TrueOT = []
CRISTA_in_TrueOT = []
Peng2018_in_TrueOT = []


preproc = ml_data_utils.PreprocessOneHot4D()


proxy_preproc = preproc.array_encode(proxy_gRNAs_noPAM, 20)
CNNstd_preproc = preproc.array_encode(CNNstd_gRNAs_noPAM, 20)
elevation_preproc = preproc.array_encode(elevation_gRNAs_noPAM, 20)
CRISPR_NET_preproc = preproc.array_encode(CRISPR_NET_gRNAs_noPAM, 20)
CRISTA_preproc = preproc.array_encode(CRISTA_gRNAs_noPAM, 20)
Peng_preproc = preproc.array_encode(Peng2018_gRNAs_noPAM, 20)

minMMcount = np.zeros((6,36))

counter = 0
print('ignore PAM in gRNA overlap check')
for full_gRNA in TrueOT_extended_gRNAs:        
    
    gRNA = full_gRNA[:20]
    gR_preproc = preproc.array_encode([gRNA], 20)    
    
    if (gRNA in CNNstd_gRNAs_noPAM):
        CNNstd_in_TrueOT.append(True)        
    else:
        CNNstd_in_TrueOT.append(False)    
    diff = np.abs(gR_preproc - CNNstd_preproc)
    minMMcount[0, counter] = np.min(np.sum((np.sum(diff, axis=1)>0), axis=1), axis=0)    
    
    if (gRNA in elevation_gRNAs_noPAM):
        elevation_in_TrueOT.append(True)
    else:
        elevation_in_TrueOT.append(False)        
    diff = np.abs(gR_preproc - elevation_preproc)
    minMMcount[1, counter] = np.min(np.sum((np.sum(diff, axis=1)>0), axis=1), axis=0)    
    
    if (gRNA in CRISPR_NET_gRNAs_noPAM):
        CRISPR_NET_in_TrueOT.append(True)
    else:
        CRISPR_NET_in_TrueOT.append(False)
    diff = np.abs(gR_preproc - CRISPR_NET_preproc)
    minMMcount[2, counter] = np.min(np.sum((np.sum(diff, axis=1)>0), axis=1), axis=0)    
    
    if (gRNA in CRISTA_gRNAs_noPAM):
        CRISTA_in_TrueOT.append(True)
    else:
        CRISTA_in_TrueOT.append(False)
    diff = np.abs(gR_preproc - CRISTA_preproc)
    minMMcount[3, counter] = np.min(np.sum((np.sum(diff, axis=1)>0), axis=1), axis=0)    
    
    if (gRNA in Peng2018_gRNAs_noPAM):
        Peng2018_in_TrueOT.append(True)
    else:
        Peng2018_in_TrueOT.append(False)
    diff = np.abs(gR_preproc - Peng_preproc)
    minMMcount[4, counter] = np.min(np.sum((np.sum(diff, axis=1)>0), axis=1), axis=0)    
        
    
    #12/1/2021 add
    if (gRNA in proxy_gRNAs_noPAM):
        proxy_in_TrueOT.append(True)        
    else:
        proxy_in_TrueOT.append(False)    
    diff = np.abs(gR_preproc - proxy_preproc)
    minMMcount[5, counter] = np.min(np.sum((np.sum(diff, axis=1)>0), axis=1), axis=0)    
        
    counter+=1    
    

closestNonEqual = np.min(minMMcount[minMMcount>0])
print('For all gRNAs in TrueOT, the closest *nonidentical* gRNA in a baseline''s training set is {} mismatches away'.format(int(closestNonEqual)))
numTrueOTguidesForPW = np.sum(minMMcount > 0, axis=1) #these numbers reflected in Table 1 - bioRxiv revision 1

minMMcountNan = deepcopy(minMMcount)
minMMcountNan[minMMcountNan==0]=np.nan
avgMinMMBaselines = np.nanmean(minMMcountNan, axis=1)


print('Number of gRNAs in CNNstd in TrueOT: ', sum(CNNstd_in_TrueOT))
print('Number of gRNAs in elevation in TrueOT: ', sum(elevation_in_TrueOT))
print('Number of gRNAs in CRISPR-NET in TrueOT: ', sum(CRISPR_NET_in_TrueOT))
print('Number of gRNAs in CRISTA in TrueOT: ', sum(CRISTA_in_TrueOT))
print('Number of gRNAs in Peng2018 in TrueOT: ', sum(Peng2018_in_TrueOT))
print('Number of gRNAs in Proxy in TrueOT: ', sum(proxy_in_TrueOT))


data = {'TrueOT gRNA': TrueOT_extended_gRNAs, 
        'CNN_std_in_TrueOT': CNNstd_in_TrueOT,
        'elevation_in_TrueOT': elevation_in_TrueOT,
        'CRISPR-NET_in_TrueOT': CRISPR_NET_in_TrueOT,
        'CRISTA_in_TrueOT': CRISTA_in_TrueOT,
        'predictCRISPR_in_TrueOT': Peng2018_in_TrueOT}

df = pd.DataFrame(data, columns=['TrueOT gRNA', 'CNN_std_in_TrueOT', 
                                 'elevation_in_TrueOT', 'CRISPR-NET_in_TrueOT', 'CRISTA_in_TrueOT',
                                 'predictCRISPR_in_TrueOT'])
    
#True means the gRNA *IS* in the training set of the baseline model, False means it is not
# the mask later on for checking what to include should invert - 'False' gRNA points should be included in pairwise comparisons
df.to_excel('TrueOT_v1_1_gRNA_overlap.xlsx')




