# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:37:45 2020

@author: Hooooaaanng
"""
import pickle
import pandas as pd

openfile = open('../raw_crisporOffs.pickle', 'rb')
raw_data = pickle.load(openfile)
openfile.close()
#print(raw_data)

allData = pd.DataFrame(columns=['gRNA', 'OT', 'label'])
#print(allData)
i = 0
for gRNA in raw_data.keys():
    print(gRNA)
    #print('gRNA', gRNAs)
    for OT in raw_data[gRNA]:
        allData.loc[i] = [gRNA, OT, raw_data[gRNA][OT]]
#        print(allData)
        i = i + 1


allData.sort_values('gRNA') # group by gRNA 
allData.to_pickle('CRISPOR_parsed.pkl')


