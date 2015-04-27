# -*- coding: utf-8 -*-
"""

@author: nadukrow
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('training_data.csv')
data.shape # 1000 rows, 6 columns
data.columnns # Prints Patient ID, Resp, PR Seq, RT Seq, Viral load count, CD4+ count

data.Resp.value_counts() #To count the number of positive vs negative prognosis after treatment. 794 neg 206 pos.

data[data.Resp==0].describe() # Compare the VL and CD4 count of all respondents with 0
data[data.Resp==1].describe() #Compare the VL and CD4 count of all respondents with 1

data[['Resp', 'VL-t0', 'CD4-t0']][:] # This is to look the the indicators with as well as the associated response.
indicators_col = data[['Resp', 'VL-t0', 'CD4-t0']][:] # This is to solely focus our attention to the indicators and their associated reponse.

indicators_col[data.Resp==0].mean() # We can see the avg VL and CD4 count for respondents with 0.
indicators_col[data.Resp==1].mean() #See the same with respondents who had a positive prognosis.


