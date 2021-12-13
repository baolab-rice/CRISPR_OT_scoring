import pandas as pd

trueotext = pd.read_excel(r'TrueOT_v1_1_allscores.xlsx', sheet_name='TrueOT', skiprows=2)
#print(type(trueotext))
print(trueotext)                                                        
    


overlap_by_gRNA = pd.read_excel(r'TrueOT_v1_1_gRNA_overlap.xlsx', sheet_name="Sheet1")
overlap_by_gRNA = overlap_by_gRNA.rename(columns={"TrueOT gRNA" :"wildtype_seq"   })
print(overlap_by_gRNA)
    
    
truefalsemask = pd.merge(trueotext, overlap_by_gRNA, on="wildtype_seq", how="left")
print(truefalsemask)
print(len(truefalsemask))   
print(list(truefalsemask))  

truefalsemask.to_excel("TrueOT_v1_1_rawmasks.xlsx")