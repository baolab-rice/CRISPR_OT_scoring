# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:27:34 2020

@author: Hooooaaanng
"""

import pandas as pd

trueotext = pd.read_excel(r'TrueOTlist_1008_extended_aligned_upto12MM_v10.xlsx', sheet_name='Sheet1')
#print(type(trueotext))
print(trueotext)                                                        
    


overlap_by_gRNA = pd.read_excel(r'TrueOTextended_gRNA_overlap_confirm.xlsx', sheet_name="Sheet1")
overlap_by_gRNA = overlap_by_gRNA.rename(columns={"TrueOT_extended gRNA" :"wildtype_seq"   })
print(overlap_by_gRNA)
    
    
truefalsemask = pd.merge(trueotext, overlap_by_gRNA, on="wildtype_seq", how="left")
print(truefalsemask)
print(len(truefalsemask))   
print(list(truefalsemask))  

truefalsemask.to_excel("TrueOText_pairs_overlap_rawmasks_confirm.xlsx")