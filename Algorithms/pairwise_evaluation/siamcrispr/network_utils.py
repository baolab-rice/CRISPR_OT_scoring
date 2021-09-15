# -*- coding: utf-8 -*-
"""
Module for construction and development of Siamese neural networks for CRISPR OT scoring

Reference: 
[1] P. K. Kota, Y. Pan, H. Vu, M. Cao, R. G. Baraniuk, and G. Bao, "The Need 
for Transfer Learning in CRISPR-CasOff-Target Scoring," Aug. 2021.

bioRxiv: 
    
"""

import numpy as np
import pickle
from math import nan
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,  MaxPooling2D, Flatten, Dense, Input, Lambda, Dropout, LSTM, Bidirectional, Reshape
from tensorflow.keras import regularizers 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import Constant, RandomUniform
from abc import ABC, abstractmethod

 
from .ml_data_utils import PreprocessOneHot4D
 
class SiameseNetwork(ABC):     
    
    def construct(self): 
        """ Return the Keras model itself        
        """
        if self.check_conditions() is False: 
            raise ValueError("Conditions not met for this Siamese Net")
            
        shared_net = self.create_shared_net() # network that both gRNA and OT pass through
                
        if self.shared_weights_file is not None: 
            shared_net.load_weights(self.shared_weights_file)
            shared_net.trainable = self.shared_trainable                
        
        # Process gRNA and OTs in parallel        
        input_gRNAs = Input(shape = self.input_shape, name = 'gR_input')
        input_targets = Input(shape = self.input_shape, name = 'Targets_Input')
        
        processed_gRNAs = shared_net(input_gRNAs)
        processed_targets = shared_net(input_targets)    
        
        distance = self.seq_distance(processed_gRNAs, processed_targets)            
        model = Model([input_gRNAs, input_targets], outputs = distance)

        return model        

    @abstractmethod
    def create_shared_net(self): 
        """ Create a Keras model through which both gRNA and OT will pass
        """
        pass
    
    @abstractmethod
    def seq_distance(self, proc_gR, proc_OT): 
        """ Return a scalar distance value based on processed gRNA and OT through
        the shared net
        """
        pass
    
    @abstractmethod
    def check_conditions(self): 
        """ if applicable - verify some conditions are met for network to function properly
        """
        pass
    
    

class SCNN(SiameseNetwork): 
    """ Used heavily in [1] for S1C, S1C_ut, S1C_mini, S1C_dense, S2C
    Basic Siamese Convolutional Neural Network - shared net = conv and dense layers.
    """
    def __init__(self, inputShape, kWidth=[8], numFilters=[16384], kStride=(1,1), \
                 kHeight=None, denseLayers=[], conv_initializer='glorot_uniform', maxPoolShape=(1,2), \
                     poolStride=(1,1), convl2reg=0, denseRegs=[0,0], margin=1, \
                         shared_weights_file=None, shared_trainable=True):            
        """
        Parameters (some self explanatory)
        ----------
        inputShape : tuple
            shape of encoded gRNA or target sequence passed to shared network. 
            E.g., in [1], (4,26,1) is used
        kWidth : list of ints
            width of convolutions for each conv layer. E.g., [4,6] would 
            indicate width 4 in first layer, 6 in second layer
        numFilters : list of ints
            number of filters in each conv layer
        kHeight : None or int, optional
            height of first convolution. **code currently assumes this is the 
            height of the encoding of sequences** (i.e., 4 for one-hot of DNA/RNA).
            The default is None.
        denseLayers : list of ints or empty, optional
            Number of dense neurons in each dense layer. The default is [].
        conv_initializer : string, optional
            option to pass a different conv initializer. The default is 'glorot_uniform'.        
        denseRegs : 2-length list, optional
            l1 and l2 regularization strength on dense layers. The default is [0,0].
        margin : float, optional
            margin parameter for contrastive loss. The default is 1.
        shared_weights_file : string, optional
            file path to load in weights for the shared network. The default is None.
        shared_trainable : boolean, optional
            whether or not the shared network is trainable. Setting to False
            freezes the shared net weights. The default is True.
        Returns
        -------
        None.

        """            
        if len(kWidth) != len(numFilters): 
            raise ValueError('Must pass in same number of kernel widths and numFilters')
        self.input_shape = inputShape
        self.kw = kWidth
        self.kh = kHeight
        self.F = numFilters
        self.denseLayers = denseLayers
        self.conv_init = conv_initializer
        self.convl2reg = convl2reg
        self.denseRegs = denseRegs
        self.maxPoolShape = maxPoolShape #None or size of pooling window e.g. 2 -> pool = (1,2)
        self.poolStride = poolStride
        self.kStride = kStride
        self.shared_weights_file = shared_weights_file
        self.shared_trainable = shared_trainable

            
    def create_shared_net(self): 
        if self.kh is None: 
            kHeight = self.input_shape[0]
        else:
            kHeight = self.kh   
        
        inp = Input(shape = self.input_shape, name = 'Input')        
        x = Conv2D(filters = self.F[0], strides=self.kStride, kernel_size = (kHeight,self.kw[0]), \
                   activation = 'relu', name = 'firstconv', kernel_regularizer=regularizers.l2(self.convl2reg), \
                       kernel_initializer=self.conv_init)(inp)    
        if self.maxPoolShape is not None: 
            x = MaxPooling2D(pool_size = self.maxPoolShape, strides = self.poolStride, name = 'First_MaxPool2D')(x)
                    
        for i in range(np.size(self.F)-1):                             
            x = Conv2D(filters = self.F[i+1], kernel_size = (1,self.kw[i+1]), activation = 'relu', \
                       kernel_regularizer=regularizers.l2(self.convl2reg), kernel_initializer=self.conv_init)(x)            
            if self.maxPoolShape is not None: 
                x = MaxPooling2D(pool_size = self.maxPoolShape, strides = self.poolStride)(x)
            
        if np.size(self.denseLayers) > 0: 
            x = Flatten(name = 'Flatten')(x)    
            
        for i in range(np.size(self.denseLayers)):    
            x = Dense(self.denseLayers[i], name=('Dense'+str(i+1)), kernel_regularizer=regularizers.l1_l2(l1=self.denseRegs[0], l2=self.denseRegs[1]), activation='relu')(x)            
    
        return Model(inp, x)
    
    def seq_distance(self, proc_gR, proc_OT): 
        proc_gR = Flatten(name='FlattenGR')(proc_gR)
        proc_OT = Flatten(name='FlattenOT')(proc_OT)
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='eucl_dist')([proc_gR, proc_OT])        
        return distance
    def check_conditions(self):
        return True
    
class BlankSiamese(SCNN): 
    """ Not used in [1]
    Doesn't transform the input. Passes directly to euclidean distance.
    A mismatch counter without alignment. 
    """
    def __init__(self, input_shape): 
        self.input_shape = input_shape
        self.shared_weights_file = None
    def create_shared_net(self): 
        x = Input(shape = self.input_shape, name = 'Input')        
        return Model(x, x) 
    
    
class CosineSCNN(SCNN): 
    """ Not used in [1]
    cosine distance (1- cosine similarity) as distance metric. Suggest a change of 'margin' accordingly
    """
    def seq_distance(self, proc_gR, proc_OT): 
        proc_gR = Flatten(name='FlattenGR')(proc_gR)
        proc_OT = Flatten(name='FlattenOT')(proc_OT)
        distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape, name='cosine_dist')([proc_gR, proc_OT])        
        return distance
    
class L2SquaredSCNN(SCNN): 
    """ Not used in [1]
    Squared Euclidean distance
    """
    def seq_distance(self, proc_gR, proc_OT): 
        proc_gR = Flatten(name='FlattenGR')(proc_gR)
        proc_OT = Flatten(name='FlattenOT')(proc_OT)
        distance = Lambda(l2_squared, output_shape=eucl_dist_output_shape, name='l2_sq_dist')([proc_gR, proc_OT])        
        return distance
        
class LearnedDistanceSCNN(SCNN): 
    """ Used in [1] for Hybrid S1C trained on Proxy TrainCV (needed for training the whole architecture)    
    Position-wise distance fed into dense layers to learn nonlinear distance model
    """
    def __init__(self, inputShape, kWidth=[8], numFilters=[16384], posDense=[256], kStride=1, \
                 kHeight=None, denseLayers=[], conv_initializer='glorot_uniform', maxPoolShape=(1,2), \
                     poolStride=(1,1), convl2reg=0, denseRegs=[0,0], margin=1, \
                         posDropout=0, visDropout=False, postPoolShape=None, postPoolStride=(1,5), \
                             shared_weights_file=None, shared_trainable=True, initSum=False):
        """        
        New Parameters. All 'pos' are generally new - refer to processing of *position*
        squared distance vector (PSDV)
        ----------
        posDense : list of ints, optional
            Dense neurons for each layer in post processing network. The default is [256].
        posDropout : float (0,1), optional
            dropout parameter for dense layers. The default is 0.2.
        visDropout : float (0,1) or False, optional
            dropout value for the visible layer of the postprocessing network (PSDV). 
            The default is False.
        postPoolShape : tuple or None, optional
            max pooling on PSDV. The default is None.
        postPoolStride : tuple, optional
            stride on pooling (if applicable) for PSDV. The default is (1,5).
        initSum : boolean, optional
            Whether to initialize dense weights such that they're equivalent
            to summing the PSDV. If True, the network performs equivalently to the 
            S1C at initializatoin and is (hopefully) trained for improvement. 
            The default is False.
        """
        if len(kWidth) != len(numFilters): 
            raise ValueError('Must pass in same number of kernel widths and numFilters')
        self.input_shape = inputShape
        self.kw = kWidth
        self.kh = kHeight
        self.F = numFilters
        self.denseLayers = denseLayers
        self.convl2reg = convl2reg
        self.denseRegs = denseRegs
        self.conv_init=conv_initializer
        self.maxPoolShape = maxPoolShape #None or size of pooling window e.g. 2 -> pool = (1,2)
        self.poolStride = poolStride
        self.kStride = kStride
        self.shared_weights_file = shared_weights_file
        self.shared_trainable = shared_trainable
        
        # new inputs for this subclass
        self.pos_dense = posDense 
        self.pos_dropout = posDropout
        self.vis_dropout = visDropout
        self.post_pool_shape = postPoolShape
        self.post_pool_stride = postPoolStride
        self.init_sum = initSum
        
    def seq_distance(self, proc_gR, proc_OT): 
        pos_sq_distance = Lambda(position_sq_distance, output_shape=position_distance_output_shape, name='position_distances')([proc_gR, proc_OT])

        # to arrange scaling of weights if initSum is True        
        denseScale=1
        if self.post_pool_shape is not None: 
            preLength = pos_sq_distance.shape[2]
            pos_sq_distance = MaxPooling2D(pool_size = self.post_pool_shape, strides = self.post_pool_stride, name = 'post_maxpool')(pos_sq_distance)
            postLength = pos_sq_distance.shape[2]
            denseScale = preLength/postLength
            
        dist = Flatten(name = 'Flatten_pos_dist')(pos_sq_distance)    
        if self.vis_dropout: # dropout on visible layer
            dist = Dropout(self.vis_dropout)(dist)
        
        for d in range(len(self.pos_dense)):
            if self.init_sum: 
                dist = Dense(self.pos_dense[d], kernel_initializer=Constant((1*denseScale)/self.pos_dense[d] * 1/(1-self.pos_dropout),), bias_initializer='zeros', activation='relu')(dist)                        
            else: 
                dist = Dense(self.pos_dense[d], activation='relu')(dist)
                #dist = Dense(self.pos_dense[d], activation='relu', kernel_initializer=RandomUniform(minval=0, maxval=1))(dist)
            dist = Dropout(self.pos_dropout)(dist)                        
        
        distance = self.final_neuron(dist, denseScale)    
        return distance
    
    def final_neuron(self, dist, denseScale): # initialize with summation
        if self.init_sum:
            if len(self.pos_dense) == 0: 
                dist2 = Dense(1, kernel_initializer=Constant(1*denseScale,), bias_initializer='zeros', activation='relu')(dist)                        
            else:                
                dist2 = Dense(1, kernel_initializer=Constant(1,), activation='relu')(dist) # help initialize at 
        else: 
            dist2 = Dense(1, activation='relu')(dist)
            #dist2 = Dense(1, activation='relu', kernel_initializer=RandomUniform(minval=0, maxval=1))(dist)
            
        return dist2        
    
    def check_conditions(self): 
        # No dense layers - position-wise distances between filter outputs
        if np.size(self.denseLayers) > 0: 
            return False
        else:
            return True
        
        
class LearnedDistancePreprocWithSCNN(LearnedDistanceSCNN): 
    """ Used in [1] for transfer learning Hybrid S1C. Helpful to compute the S1C's output just once
    and treat the S1C as a preprocessor since the conv weights are frozen. 
    
    See PositionDistanceSCNN class below.
    
    Position-wise distance fed into dense layers to learn nonlinear distance model
    """
    def __init__(self, inputShape, posDense=[512], posDropout=0.2, \
                 visDropout=False, postPoolShape=None, postPoolStride=(1,5), initSum=False):
        self.input_shape = inputShape
        self.pos_dense = posDense 
        self.pos_dropout = posDropout
        self.vis_dropout = visDropout
        self.post_pool_shape = postPoolShape
        self.post_pool_stride = postPoolStride
        self.init_sum = initSum
        
    def construct(self): 
        if self.check_conditions() is False: 
            raise ValueError("Conditions not met for this Siamese Net")
                    
        # Process gRNA and OTs in parallel        
        input_pos_distance = Input(shape = self.input_shape, name = 'gR_OT_dist')        
        distance = self.seq_distance(input_pos_distance)            
        model = Model(input_pos_distance, outputs = distance)
        return model        


    def seq_distance(self, pos_sq_distance):
        # have the position squared distance vector as an input in this model, so skip to dense layers
        denseScale=1
        if self.post_pool_shape is not None: 
            preLength = pos_sq_distance.shape[2]
            pos_sq_distance = MaxPooling2D(pool_size = self.post_pool_shape, strides = self.post_pool_stride, name = 'post_maxpool')(pos_sq_distance)
            postLength = pos_sq_distance.shape[2]
            denseScale = preLength/postLength
            
        dist = Flatten(name = 'Flatten_pos_dist')(pos_sq_distance)    
        if self.vis_dropout: # dropout on visible layer
            dist = Dropout(self.vis_dropout)(dist)
        
        for d in range(len(self.pos_dense)):
            if self.init_sum: 
                dist = Dense(self.pos_dense[d], kernel_initializer=Constant((1*denseScale)/self.pos_dense[d] * 1/(1-self.pos_dropout),), bias_initializer='zeros', activation='relu')(dist)                        
            else: 
                dist = Dense(self.pos_dense[d], activation='relu')(dist)
            dist = Dropout(self.pos_dropout)(dist)            
        
        distance = self.final_neuron(dist, denseScale)
    
        return distance
    
    def check_conditions(self): 
        return True
        
class BLSTMPreprocWithSCNN(LearnedDistanceSCNN): 
    """ Not used in [1], and not finalized, but plausible extension of Hybrid models.
    RNN as postprocessing network for Hybrid Siamese models (rather than dense layers)
    Position-wise distance fed into dense layers to learn nonlinear distance model
    """
    def __init__(self, inputShape, nodes=16, denseClassifier=[], dropout=0):
        self.input_shape = inputShape
        self.nodes = nodes 
        self.dense_class = denseClassifier
        self.dropout = dropout
        
    def construct(self): 
        if self.check_conditions() is False: 
            raise ValueError("Conditions not met for this Siamese Net")
                    
        # Process gRNA and OTs in parallel        
        input_pos_distance = Input(shape = self.input_shape, name = 'gR_OT_dist')        
        distance = self.seq_distance(input_pos_distance)            
        model = Model(input_pos_distance, outputs = distance)
        return model        


    def seq_distance(self, pos_sq_distance):         
        #lstm_layer = LSTM(self.nodes, kernel_regularizer=regularizers.l1_l2(l1=denseRegs[0], l2=denseRegs[1]), return_sequences=True)
        pos_sq = Reshape((pos_sq_distance.shape[2], 1))(pos_sq_distance)
        
        lstm_layer = LSTM(self.nodes, return_sequences=False)
        out = Bidirectional(lstm_layer)(pos_sq)
        
        out = Flatten()(out)
        for d in range(len(self.dense_class)):
            out = Dense(self.dense_class[d], activation='relu')(out)
            out = Dropout(self.pos_dropout)(out)            

        out = self.final_neuron(out)

        return out
    
    def final_neuron(self, out): 
        #standard sigmoid output for classification - not learning a nonlinear 'distance'
        out = Dense(1, activation='sigmoid')(out)
        return out
    
    def check_conditions(self): 
        return True
    
    
class HybridSigmoidSCNN(LearnedDistanceSCNN): 
    """ Not used in [1] - end with sigmoid output instead of ReLU (lets go of ``distance" analogy)
    """
    def final_neuron(self, dist, denseScale):         
        distance = Dense(1, activation='sigmoid')(dist)
        return distance
    
class HybridSigmoidPreprocWithSCNN(LearnedDistancePreprocWithSCNN): 
    """ Not used in [1] - end with sigmoid output instead of ReLU (lets go of ``distance" analogy)
    """
    def final_neuron(self, dist, denseScale): 
        distance = Dense(1, activation='sigmoid')(dist)
        return distance
              
class PositionDistanceSCNN(LearnedDistanceSCNN): 
    """ Used in [1] - useful for preprocessing/fast training of Hybrid S1C (or similar) model with
    fixed shared net weights (only training post-processing network)
    
    Position-wise distance basically serves as preprocessor
    """
    def __init__(self, inputShape, kWidth=[8], numFilters=[16384], posDense=[256], kStride=1, \
                 kHeight=None, denseLayers=[], maxPoolShape=(1,2), conv_initializer='glorot_uniform', \
                     poolStride=(1,1), convl2reg=0, denseRegs=[0,0], margin=1, \
                         posDropout=0, visDropout=False, \
                             shared_weights_file=None, shared_trainable=True):
        if len(kWidth) != len(numFilters): 
            raise ValueError('Must pass in same number of kernel widths and numFilters')
        self.input_shape = inputShape
        self.kw = kWidth
        self.kh = kHeight
        self.F = numFilters
        self.denseLayers = denseLayers
        self.convl2reg = convl2reg
        self.conv_init = conv_initializer
        self.denseRegs = denseRegs
        self.maxPoolShape = maxPoolShape #None or size of pooling window e.g. 2 -> pool = (1,2)
        self.poolStride = poolStride
        self.kStride = kStride
        self.shared_weights_file = shared_weights_file
        self.shared_trainable = shared_trainable    
        
    def seq_distance(self, proc_gR, proc_OT): 
        pos_sq_distance = Lambda(position_sq_distance, output_shape=position_distance_output_shape, name='position_distances')([proc_gR, proc_OT])
                
        return pos_sq_distance      
    
            
def get_contrastive_loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        margin_pred = (K.maximum(margin-y_pred, 0))
        return K.mean(y_true * y_pred + (1-y_true) * margin_pred)
    return contrastive_loss

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)    
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes    
    return (shape1[0], 1)

def l2_squared(vects):
    x, y = vects
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)    
    return K.maximum(sum_square, K.epsilon())

def cosine_distance(vects): #https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras/52021481
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1-K.sum(x * y, axis=-1, keepdims=True) #pk - added the 1- so it ranges [0,2]    

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def position_distance_output_shape(shapes):
    shape1, shape2 = shapes
    #return (shape1[0], 1)
    return (shape1[0], shape1[1], shape1[2], 1)

def position_sq_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x-y), axis = 3, keepdims = True)
    return (K.maximum(sum_square, K.epsilon()))


def pipeline(seq_df, model_ensemble, preproc=PreprocessOneHot4D(max_length=26, strip_dash=True), sort_scores=True): 
    """
        
    Parameters
    ----------
    seq_df : pandas dataframe or string
        IF dataframe: Dataframe with columns 'gRNA' and 'OT'. 
            Optionally can have 'label' too, if available/desired. 
            Will make a blank nan label column if it doesn't exist        
        IF string: dataframe storked in a .pkl ('pickled') file with same
            conditions noted above        
    model_ensemble : list
        List of pre-loaded Keras models used in ensemble.  
    preproc : ml_data_utils.Preprocessor object
        Default value corresponds to the preprocessor used in the final, 
        published model. This should only be changed if appropriate (e.g. new model)    
    sort_scores: boolean
        Default: True. Whether or not to sort scores from most similar to least similar (lowest->highest 'distance')        
    Returns
    -------
    scores : numpy array
        Result of scoring algorithm for all data passed in

    """
    if len(model_ensemble) != 5: 
        print('Warning: 5 models used in originally published ensemble - check files')
        
    if type(seq_df) is str: 
        openfile = open(seq_df, 'rb')
        seq_df = pickle.load(seq_df)
        openfile.close()     
    if 'label' not in seq_df.columns:
        seq_df['label'] = nan
    
    Y_preds = np.zeros((seq_df['gRNA'].count(), len(model_ensemble)))    

    # Preprocess    
    all_gRNAs, all_OTs, all_labels, _, _= preproc.preprocess(seq_df, shuffle=False, groupGRNA=False) 
    print('Data encoding complete')

    for m in range(len(model_ensemble)): 
        Y_preds[:,m] = model_ensemble[m].predict([all_gRNAs, all_OTs])[:,0]
    
    scores = np.median(Y_preds, axis=1) #past the median, the ensemble flips label. So if need to sort, use this instead of mean
    seq_df['score'] = scores    
    if sort_scores: 
        seq_df = seq_df.sort_values(by=['score']) #smallest -> largest score value
    
    return seq_df

    
def full_h5_split(h5file, file_append='_weights'):     
    """
    Take an .h5 file with the model weights and architecture and save them separately
    as a separate .h5 (only weights) and .json. Ideally, this is a more portable format
    between versions
    
    Parameters
    ----------
    h5file : string
        filepath to full h5 file    
    file_append : string, optional
        add-on to filepath to save new files. The default is '_weights' for h5.

    Returns
    -------
    None.
    """    
    h5fileTrim = h5file[:-3]
    
    model = load_model(h5file, custom_objects={'contrastive_loss':get_contrastive_loss()})
    #model = load_model(h5file, custom_objects={'contrastive_loss':custom_keras.contrastive_loss})
    
    model_json = model.to_json()
    with open( h5fileTrim  + '.json', "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(h5fileTrim+file_append+'.h5')    
    
    
    