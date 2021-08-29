"""
Module for handling data for CRISPR OT scoring. 
Sequence preprocessing (cleaning data and encoding sequences) and data splitting 
(unique gRNAs in various splits)

Reference: 
[1] P. K. Kota, Y. Pan, H. Vu, M. Cao, R. G. Baraniuk, and G. Bao, "The Need 
for Transfer Learning in CRISPR-CasOff-Target Scoring," Aug. 2021.

bioRxiv: 
    
"""
import numpy as np
import random
import pickle
import pdb
import pandas as pd
from copy import deepcopy
from pandas import concat
from abc import ABC, abstractmethod

class CustomException(Exception):
    def __init__(self, message): 
        super().__init__(message)        
    pass


class Preprocessor(ABC):
    def __init__(self, num_downsample=None, pos_prop=None, strip_dash=False, threshold=0, pad=-1, split_by_gRNA=False):
        """
        num_downsample: int
            Number of datapoints to get from file
        pos_prop: float (0,1)
            If not None, forces positive labels of a certain proportion in downsampling 
            Example: if pos_prop = 0.25 and downsample=100, then 25 positive labels will be drawn
                prints a warning if you don't actually get 25 labels - gives all positive labels and
                the corresponding number of negative labels to maintain the proportion 
                - e.g. in same example if only 10 positive labels available, returns 10 positive labels and
                30 negative labels
        strip_dash: boolean
            Whether or not to remove '-' from sequences
        threshold: float >= 0
            Threshold above which to call labels 'positive'. Labels in raw data may be continuous
        pad: {-1, 0, 1}
            -1 -> zero pad on left side to encoding length
            0 -> no padding
            1 -> zero pad on right side to encoding length            
        """        
        self.num_downsample = num_downsample
        self.pos_prop = pos_prop
        self.strip_dash = strip_dash
        self.threshold = threshold
        self.pad = pad
        
        
    def preprocess(self, dataset, shuffle=True, rngseed=None, groupGRNA=True):                 
        """
        dataset is dataframe with 'gRNA', 'OT', and 'label' columns
        """
        
        if type(dataset) is str: # assume it's a .pkl file to load 
            openfile = open(dataset, 'rb')
            raw_data = pickle.load(openfile)
            openfile.close()     
        else: 
            raw_data = dataset 
        
        labels = raw_data['label'].values
        labels = (labels > self.threshold).astype('int')
        
        #Downsample if applicable (can still do this on a pre-encoded dataset)
        if self.num_downsample is not None: 
            inds = self.downsample(labels)    
            raw_data = raw_data.iloc(inds)
            
        if groupGRNA:
            #from https://stackoverflow.com/questions/45585860/shuffle-a-pandas-dataframe-by-groups
            raw_data, groups = group_gRNAs(raw_data, shuffle=shuffle, rngseed=rngseed)
        
        
        gRNAs = raw_data['gRNA'].values
        OTs = raw_data['OT'].values
        labels = raw_data['label'].values
        
        # get rid of extra spaces if any
        for i in range(len(gRNAs)):             
            gRNAs[i] = gRNAs[i].replace(' ', '')
            OTs[i] = OTs[i].replace(' ','')
            
        if self.strip_dash is True: 
            for i in range(len(gRNAs)):
                gRNAs[i] = gRNAs[i].replace('-', '')
                OTs[i] = OTs[i].replace('-','')
                              
        gRNAs_preprocessed, OTs_preprocessed, inputShape = self.encode(gRNAs, OTs)         
        
        return gRNAs_preprocessed, OTs_preprocessed, labels, inputShape, raw_data

        

    def downsample(self, labels): 
        """ Not used in [1]
        """
        #Downsample if applicable before one-hot encoding
        if self.pos_prop is None: 
            inds = np.random.choice(len(labels), size=self.num_downsample, replace=False)
        else: 
            posChoices = np.where(labels == 1)[0] # make positive choices! :/
            negChoices = np.where(labels==0)[0]
            numPos = np.size(posChoices)
            if numPos == 0: #note files this happens with - probably shouldn't happen
                print('Warning: ZERO positive samples in file - drawing 1 random sample')                
                numPosSelect = 0
                numNegSelect = 1                      
            if numPos < round(self.pos_prop*self.num_downsample):
                print('Warning: not enough positive samples for downsampling with pos_prop')
                numPosSelect = numPos
                numNegSelect = round((1-self.pos_prop)/(self.pos_prop) * numPosSelect)
            else: 
                numPosSelect = round(self.pos_prop*self.num_downsample) 
                numNegSelect = self.num_downsample - numPosSelect
            posInds = np.random.choice(posChoices, size=numPosSelect, replace=False)
            negInds = np.random.choice(negChoices, size=numNegSelect, replace=False)
            inds = np.concatenate((posInds, negInds))
        return inds
    
    def get_max_len(self, gRNAs, OTs):         
        #Get max length between all gRNA and target sequences
        if self.max_length is None: #unspecified 
            max1 = len(max(OTs, key = len))
            max2 = len(max(gRNAs, key = len))
            maxLength = max([max1, max2])
        else: 
            maxLength = self.max_length
        return maxLength
            
    @abstractmethod
    def encode(self):
        pass
    
class PreprocessorPreEncoded(Preprocessor): 
    """ maybe helpful in develompent to save encoding time - not used in final results for [1]
    """
    def preprocess(self, filepath_withr, rngseed=None): 
        gRNAs_preprocessed, OTs_preprocessed, labels = load_pkl_encoded_data(filepath_withr)            
        labels = (labels > self.threshold).astype('int')

        # TO DO Likely have to change the way indexing works with pre-encoded data                
        if self.num_downsample is not None: 
            inds = self.downsample(labels)    
            gRNAs_preprocessed = gRNAs_preprocessed[inds]
            OTs_preprocessed = OTs_preprocessed[inds]
            labels = labels[inds]    
           
        inputShape = [gRNAs_preprocessed.shape[0], gRNAs_preprocessed.shape[1], None]
        return gRNAs_preprocessed, OTs_preprocessed, labels, inputShape

class PreprocessOneHot4D(Preprocessor):    
    
    def __init__(self, num_downsample=None, pos_prop=None, strip_dash=False, threshold=0, pad=-1, max_length=None):         
        """ maxLength sets the encoding length and zero pads anything shorter
        """
        super().__init__(num_downsample, pos_prop, strip_dash, threshold, pad)
        self.max_length = max_length
        self.baseEncodings = {'A': np.array([1, 0, 0, 0]),   \
             'C': np.array([0, 1, 0, 0]),   \
             'G': np.array([0, 0, 1, 0]),   \
             'T': np.array([0, 0, 0, 1]),   \
             'N': np.array([0.25, 0.25, 0.25, 0.25]),   \
                 'R': np.array([0.5, 0, 0.5, 0]), \
                     '-': np.array([0,0,0,0])                     
                 }
        self.encoding_dim = 4
    def encode(self, gRNAs, OTs):        
        maxLength = self.get_max_len(gRNAs, OTs)
        
        #one-hot encode with zero-padding for anything shorter than maxLength
        gRNAs_preprocessed = self.array_encode(gRNAs, encodeLength=maxLength)
        OTs_preprocessed = self.array_encode(OTs, encodeLength=maxLength)
                
        #Define input shape to the network of an individual datapoint (e.g. training example). Add a 1 at the last dimension
        inputShape = gRNAs_preprocessed.shape[1:]        
        
        return gRNAs_preprocessed, OTs_preprocessed, inputShape
    
    def encode_seq(self, seq, encodeLength=0): 
        seq = seq.upper() # in case lower case a-c-g-t used
        if encodeLength == 0 or self.pad==0: #unspecified or overwrite
            encodeLength = len(seq)        
            
        encoding = np.zeros((self.encoding_dim,encodeLength))        
        lenDiff = encodeLength - len(seq)
        for j, nuc in enumerate(seq):
            if self.pad == 0 or self.pad == 1: 
                encoding[:,j] = self.baseEncodings[nuc]
            elif self.pad == -1:  # left pad
                encoding[:,j+lenDiff] = self.baseEncodings[nuc]
            else: 
                raise ValueError('Invalid value for self.pad. Must be one of {-1, 0, 1}')
        return encoding
    
    def array_encode(self, seqs, encodeLength): 
        if len(np.shape(seqs)) != 1: 
            raise CustomException('Must pass one-dimensional array of sequences')
            
        for i, seq in enumerate(seqs):
            enc = self.encode_seq(seq, encodeLength)
            enc = self.encode_reshape(enc)
            if i == 0:            
                arraySeqs = enc     
            else:
                arraySeqs = np.concatenate((arraySeqs, enc), axis=0)                                            
    
        return arraySeqs

    def encode_reshape(self, enc): 
        return np.reshape(enc, (1, self.encoding_dim, np.shape(enc)[1], 1)) # e.g. (1, 4, 23, 1)
        
    
class PreprocessOneHot5Ddel(PreprocessOneHot4D):    
    """ Not used in [1] 
    """
    def __init__(self, num_downsample=None, pos_prop=None, strip_dash=False, threshold=0, pad=-1, max_length=None):         
        #super(Preprocessor, self).__init__(num_downsample, pos_prop, strip_dash, threshold, pad)
        super().__init__(num_downsample, pos_prop, strip_dash, threshold, pad)
        self.max_length = max_length
        
        self.baseEncodings = {'A': np.array([1, 0, 0, 0, 0]),   \
             'C': np.array([0, 1, 0, 0, 0]),   \
             'G': np.array([0, 0, 1, 0, 0]),   \
             'T': np.array([0, 0, 0, 1, 0]),   \
             'N': np.array([0.25, 0.25, 0.25, 0.25, 0]),   \
                 'R': np.array([0.5, 0, 0.5, 0, 0]), \
                     ' ': np.array([0, 0, 0, 0, 0]), \
                     '-': np.array([0, 0, 0, 0, 1])
                 }
        self.encoding_dim = 5
        
class PreprocessOneHot4Drnn(PreprocessOneHot4D): 
    """Not used in [1] but small shape adjustments for developing RNNs
    """
    def encode_reshape(self, enc): 
        enc = np.transpose(enc) #e.g. 23 x 4                
        return np.reshape(enc, (1, np.shape(enc)[0], np.shape(enc)[1]))                
        

class Preprocess4DSCNN(PreprocessOneHot4D): 
    """ Pairs with netweork_utils.LearnedDistancePreprocWithSCNN and network_utils.PositionDistanceSCNN
    """
    def __init__(self, scnn, num_downsample=None, pos_prop=None, strip_dash=False, threshold=0, pad=-1, max_length=None):         
    
        super().__init__(num_downsample, pos_prop, strip_dash, threshold, pad)
        self.max_length = max_length
        self.baseEncodings = {'A': np.array([1, 0, 0, 0]),   \
             'C': np.array([0, 1, 0, 0]),   \
             'G': np.array([0, 0, 1, 0]),   \
             'T': np.array([0, 0, 0, 1]),   \
             'N': np.array([0.25, 0.25, 0.25, 0.25]),   \
                 'R': np.array([0.5, 0, 0.5, 0]), \
                     '-': np.array([0,0,0,0])                     
                 }
        self.encoding_dim = 4
        self.scnn = scnn
        
    def preprocess(self, dataset, shuffle=True, rngseed=None, groupGRNA=True):
        gR, OT, labels, inputShape, raw_data = super().preprocess(dataset, shuffle, rngseed, groupGRNA)        
                
        gR_OT_dist = self.scnn.predict([gR, OT])
        return gR_OT_dist, None, labels, gR_OT_dist.shape[1:], raw_data
    
    
def load_pkl_data(datafile, rngseed=None, shuffle=True):
    """
    Load in gRNA, OT, and label data. Returns preprocessed data shuffled but grouped by gRNA. 
        If you want gRNAs mixed throughout train/test, be sure to shuffle after this function            
            
    INPUTS: 
        datafile: path to the .pkl data file as string with r in front (e.g. r'..\datasets\parsing\().pkl')
            Data file should contain a DataFrame with columns 'gRNA', 'OT', and 'label'
        rngseed: Data is shuffled by gRNA here in case of further preprocessing.
        
        
    """
    openfile = open(datafile, 'rb')
    raw_data = pickle.load(openfile)
    openfile.close()
    
    random.seed(rngseed)
    
    if shuffle is True: 
        #Shuffle all data, but keep grouped by gRNAs
        #from https://stackoverflow.com/questions/45585860/shuffle-a-pandas-dataframe-by-groups
        groups = [raw_data for _, raw_data in raw_data.groupby('gRNA')]
        random.shuffle(groups)
        raw_data = concat(groups).reset_index(drop=True)
    
    # Obtain the relevant columns
    gRNAs = raw_data['gRNA'].values
    OTs = raw_data['OT'].values
    labels = raw_data['label'].values
    
    return gRNAs, OTs, labels


    
def load_pkl_encoded_data(datafile):
    """ Loads in pre-encoded data as is"""
    openfile = open(datafile, 'rb')
    raw_data = pickle.load(openfile)
    openfile.close()
    gRNAs, OTs, labels = raw_data[0:3]
    return gRNAs, OTs, labels
    
def get_max_len(filepath_withr):
    """
    downsample: int
        Number of datapoints to get from file
    pos_prop: float (0,1)
        If not None, forces positive labels of a certain proportion in downsampling 
        Example: if pos_prop = 0.25 and downsample=100, then 25 positive labels will be drawn
            prints a warning if you don't actually get 25 labels - gives all positive labels and
            the corresponding number of negative labels to maintain the proportion 
            - e.g. in same example if only 10 positive labels available, returns 10 positive labels and
            30 negative labels
    """
    
    gRNAs, OTs, labels = load_pkl_data(filepath_withr)    
    for i in range(len(gRNAs)): 
        gRNAs[i] = gRNAs[i].replace(' ', '')
        OTs[i] = OTs[i].replace(' ','')
        
    max1 = len(max(OTs, key = len))
    max2 = len(max(gRNAs, key = len)) 
    return max([max1, max2])        
        
def split_by_gRNAs(raw_data, test_pct, kfold=1, split_size_tol=0.20, rngseed=None):

    groups = [raw_data for _, raw_data in raw_data.groupby('gRNA')]
    random.seed(rngseed)
    #Shuffle all data, but keep grouped by gRNAs        
    random.shuffle(groups)                  
    
    
    totalSize = raw_data.shape[0]
    testSize = round(test_pct * totalSize)
    trainValSize = totalSize - testSize
    kfoldSize = trainValSize / kfold
 
    
    groupSizes = []
    for g in range(len(groups)): 
        groupSizes.append(groups[g].shape[0])        
   
    #populate trainval to minimize split size error
    buildSize = 0
    trainValGroupInds = np.array([]).astype(int)
    for g in range(len(groups)): 
        currentErr = abs(buildSize - trainValSize) / trainValSize 
        buildTemp = buildSize + groupSizes[g]
        newErr = abs(buildTemp - trainValSize) / trainValSize 
        if newErr < currentErr: 
            buildSize+=groupSizes[g]
            trainValGroupInds = np.append(trainValGroupInds, g)
            currentErr = abs(buildSize - trainValSize) / trainValSize 
    if currentErr > split_size_tol: 
        raise ValueError('could not achieve desired split')        
    testGroupInds = np.setdiff1d(np.arange(len(groups)), trainValGroupInds)
            
    # split trainval into kfolds 
    kfoldGroupInds = [ [] ] * kfold
    kfoldGroupSize = [0]*kfold
    #kOrder = np.arange(kfold)
    
    for g in range(np.size(trainValGroupInds)): 
        #np.random.shuffle(kOrder)
        kOrder = np.argsort(kfoldGroupSize) # prioritize smallest groups
        for k in range(kfold): 
            #currentErr = (kfoldGroupSize[kOrder[k]] - kFoldSize) / kFoldSize 
            buildTemp = kfoldGroupSize[kOrder[k]] + groupSizes[trainValGroupInds[g]]
            newErr = (buildTemp - kfoldSize) / kfoldSize 
            if newErr < split_size_tol: # as long as not too big
                temp = deepcopy(kfoldGroupInds[kOrder[k]])                
                temp.append(trainValGroupInds[g])
                kfoldGroupInds[kOrder[k]] = temp 
                kfoldGroupSize[kOrder[k]] += groupSizes[trainValGroupInds[g]]
                break # don't add to multiple groups, move on to next group
    
    # check kfold
    for k in range(kfold): 
        err = abs(kfoldGroupSize[k] - kfoldSize)/ kfoldSize
        if err > split_size_tol: 
            raise ValueError('Could not split kfold groups within desired error tolerance. Consider increasing split_size_tol')
    
    cvMasks = np.zeros((raw_data.shape[0], kfold))
    testMask = np.zeros((raw_data.shape[0]))
    # generate all masks 
    for k in range(kfold): 
        for ind in kfoldGroupInds[k]: 
            cvMasks[np.array(groups[ind].index), k] = 1            
    for ind in testGroupInds: 
        testMask[np.array(groups[ind].index)] = 1           
    cvMasks = cvMasks.astype('bool')
    testMask = testMask.astype('bool')
 
        
    
    return cvMasks, testMask

def group_gRNAs(raw_data, shuffle=True, rngseed=None): 
    #from https://stackoverflow.com/questions/45585860/shuffle-a-pandas-dataframe-by-groups
    groups = [raw_data for _, raw_data in raw_data.groupby('gRNA')] #**group order will be alphabetized**
    if shuffle is True:
        random.seed(rngseed)
        #Shuffle all data, but keep grouped by gRNAs        
        random.shuffle(groups)        
    raw_data = concat(groups).reset_index(drop=True)
    
    return raw_data, groups
    

def crispr_read_excel(filename, colgRNA, colOT, colLabel, sheetName='Sheet1', skipRows=0):
    parsedData = pd.DataFrame(columns=['gRNA', 'OT', 'label'])    
    df = pd.read_excel(filename, sheet_name=sheetName, skiprows=skipRows)    
    parsedData = df.iloc[:,[colgRNA, colOT, colLabel]]
    colNames = list(parsedData)

    return parsedData.rename(columns={colNames[0]: 'gRNA', colNames[1]:'OT', colNames[2]:'label'})
    