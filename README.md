# Revision Notes: 

+ Rev. 1 (August 2021): original bioRxiv submission

+ Rev. 2 (December 2021): updated version + models reflecting our bioRxiv update (planning to post, Dec. 2021)

# Dataset preparation and data processing for the CRISPR OT scoring paper

## Supplementary tables 
**Folder: Supplementary tables**

Table S1-S4 of the CRISPR off-target scoring paper.

+ Table S1  
Detailed information of studies included in TrueOT dataset, including PUBMED ID, the detection method of targets, cell type, number of evaluated targets and true targets, and the summary of off-target profile assessment.

+ Table S2  
Detailed information of the TrueOT dataset, in which all 1903 datapoints were evaluated through experimental measurement of target sequence modification frequency.

+ Table S3 (new as of Rev2): 
Detailed information of +/- labels, bulge/non-bulge datapoints in all datasets used in study. 

+ Table S4 (previously Table S3 in Rev1)
Summary of datasets utilized in ML-based off-target scoring model development, supporting the pairwise comparison section. 

## TrueOT: The benchmark dataset we proposed, and its original data sources. 

**Folder: TrueOT_unprocessed_data**

The original data sources of our manually curated true positive list collected from 11 studies (as also described in Table S1): 

+ 2020_Lazzarotto
Lazzarotto, Cicera R., et al. "CHANGE-seq reveals genetic and epigenetic effects on CRISPR–Cas9 genome-wide activity." Nature biotechnology 38.11 (2020): 1317-1327.

+ 2020_Shapiro
Shapiro, Jenny, et al. "Increasing CRISPR Efficiency and Measuring Its Specificity in HSPCs Using a Clinically Relevant System." Molecular Therapy-Methods & Clinical Development 17 (2020): 1097-1107.

+ 2020_Vaidyanathan  
Vaidyanathan, Sriram, et al. "High-efficiency, selection-free gene repair in airway stem cells from cystic fibrosis patients rescues cftr function in differentiated epithelia." Cell stem cell 26.2 (2020): 161-171. 

+ 2019_Gomez_Ospina  
Gomez-Ospina, Natalia, et al. "Human genome-edited hematopoietic stem cells phenotypically correct Mucopolysaccharidosis type I." Nature communications 10.1 (2019): 1-14. 

+ 2013_Fu  
Fu, Yanfang, et al. "High-frequency off-target mutagenesis induced by CRISPR-Cas nucleases in human cells." Nature biotechnology 31.9 (2013): 822-826. 

+ 2014_Cho  
Cho, Seung Woo, et al. "Analysis of off-target effects of CRISPR/Cas-derived RNA-guided endonucleases and nickases." Genome research 24.1 (2014): 132-141. 

+ 2019_Pavel-Dinu  
Pavel-Dinu, Mara, et al. "Gene correction for SCID-X1 in long-term hematopoietic stem cells." Nature communications 10.1 (2019): 1-15.

+ 2019_Park  
Park, So Hyun, et al. "Highly efficient editing of the beta-globin gene in patient-derived hematopoietic stem and progenitor cells to treat sickle cell disease." Nucleic Acids Research 47.15 (2019): 7955-7972. 

+ 2015_Kim  
Kim, Daesik, et al. "Digenome-seq: genome-wide profiling of CRISPR-Cas9 off-target effects in human cells." Nature methods 12.3 (2015): 237.

+ 2015_Wang  
Wang, Xiaoling, et al. "Unbiased detection of off-target cleavage by CRISPR-Cas9 and TALENs using integrase-defective lentiviral vectors." Nature biotechnology 33.2 (2015): 175. 

+ 2016_Kim  
Kim, Daesik, et al. "Genome-wide target specificities of CRISPR-Cas9 nucleases revealed by multiplex Digenome-seq." Genome research 26.3 (2016): 406-415. 

## S1C Scoring Algoirthm
**Folder: custom_scoring**

Contains files for using our S1C model on any dataset. 

### Version notes for using our custom models: 
Known version *requirements*
- h5py < 3.0 (we used 2.10.0)
- tensorflow 2.x (we used 2.4.1)
- python > 3 (we used 3.8.0)

Other necessary packages (with versions as we used them, but likely not as strict)
- numpy 1.19.5
- pandas 1.2.3
- sklearn 0.24.1
- openpyxl 3.0.7

A full package list is available at custom_scoring/package_list.txt

We recommend running S1C_on_input_data.py and sample_model_train.py after cloning this repo and modifying an environment with the above version info to ensure things are working correctly. 

### Running S1C on new input data
S1C_on_input_data.py: script for running S1C model on any input XLSX file. Currently set for TrueOT, but can simply change the file info with your own dataset. Use an Excel (.xlsx file) with columns of guide RNAs, DNA targets, and (optionally) labels. You may also use a CSV, although see the code for custom_scoring/siamcrispr/ml_data_utils.crispr_read_csv when editing. See the top few lines of this script and edit accordingly. 

### Training S1C on data
sample_model_train.py: sample training script for training the S1C on the Proxy TrainCV. Any training dataset should be stored as a CSV with a 'gRNA', 'OT', and 'label' columns. More columns are fine, but they will be ignored.  Change the filepath in line 49 to point the localData to a foler with your stored .csv dataset. 

### Other details
- custom_scoring/S1C/: raw model files
- custom_scoring/parsed_datasets/: pre-parsed datasets (TrueOT and Proxy Dataset) for model training. We originally used .pkl files storing pandas dataframes, but for broader compatibility in environments, we saved these same datasets in .csv's and load them with Pandas in the code (bypassing common issues with pickle). The .pkl versions are still available.
- custom_scoring/siamcrispr: module files for our Siamese networks 

## Baseline Algorithms: Off-target scores calculation 
**Folder: Algorithm**  

The processing scripts adapted from previous publications. Please see details of each algorithm inside the folder. Their compositions of training set are available in Table S4.  


+ CCTOP off-target scores were computed based on the formula in the original paper:  
Stemmer, Manuel, et al. "CCTop: an intuitive, flexible and reliable CRISPR/Cas9 target prediction tool." PloS one 10.4 (2015).

+ Code for the MIT score (Hsu score), and CROP-IT score was adapted from the CRISPOR review:  
Haeussler, Maximilian, et al. "Evaluation of off-target and on-target scoring algorithms and integration into the guide RNA selection tool CRISPOR." Genome biology 17.1 (2016): 148.  
https://github.com/maximilianh/crisporPaper

+ Code for the CFD score was obtained from the authors:  
Doench, John G., et al. "Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9." Nature biotechnology 34.2 (2016): 184.  
Code available in supplementary: https://www.nature.com/articles/nbt.3437#MOESM9 

+ Elevation algorithm was installed based on authors instructions:  
Listgarten, Jennifer, et al. "Prediction of off-target activities for the end-to-end design of CRISPR guide RNAs." Nature biomedical engineering 2.1 (2018): 38-47.  
https://github.com/microsoft/Elevation 

+ predictCRISPR algorithm was installed based on authors instructions:  
Peng, Hui, et al. "Recognition of CRISPR/Cas9 off-target sites through ensemble learning of uneven mismatch distributions." Bioinformatics 34.17 (2018): i757-i765.  
https://github.com/penn-hui/OfftargetPredict 

+ CNN_std algorithm was installed based on authors instructions:  
Lin, Jiecong, and Ka-Chun Wong. "Off-target predictions in CRISPR-Cas9 gene editing using deep learning." Bioinformatics 34.17 (2018): i656-i663.  
https://github.com/MichaelLinn/off_target_prediction

+ Code for the CRISTA score was obtained from the authors:  
Abadi, Shiran, et al. "A machine learning approach for predicting CRISPR-Cas9 cleavage efficiencies and patterns underlying its mechanism of action." PLoS computational biology 13.10 (2017): e1005807.  
http://crista.tau.ac.il/download.html

+ Code for the COSMID score was obtained from the authors:  
Cradick, Thomas J., et al. "COSMID: a web-based tool for identifying and validating CRISPR/Cas off-target sites." Molecular Therapy-Nucleic Acids 3 (2014): e214.  
https://crispr.bme.gatech.edu/

+ CRISPR_NET algorithm was installed based on authors instructions:  
Lin, Jiecong, et al. "CRISPR‐Net: A Recurrent Convolutional Network Quantifies CRISPR Off‐Target Activities with Mismatches and Indels." Advanced Science 7.13 (2020): 1903562.
https://codeocean.com/capsule/9553651/tree/v1

## Pairwise Comparisons
**Folder: Algorithms/pairwise_evaluation**

Contains all files necessary to regenerate key results of Tables 2 and 3, i.e. the baseline algorithms' performance on appropriate subsets of TrueOT and the S1C. 

pairwise_compare.py: Can run this to directly get the results

baseline_overlap.py: Loads in training data of ML baselines and determines which guideRNAs in TrueOT need to be excluded from pairwise comparisons.

overlap_mask_confirm.py: Generates boolean masks for all of TrueOT for pairwise_comparisons

notes.txt: contains other fine-grained info of other files in folder. 

