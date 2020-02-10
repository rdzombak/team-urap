from GoalCongruence import *
import GoalCongruenceWord2Vec as gcw2v

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

master_data = pd.read_csv(r'test_data/master_table.csv')
master_obj = Cosine_Sim(master_data, gcw2v.word2vec_embed)
master_diag = Diagnostics(master_obj)

sns.lmplot(x='Calculated Values', y='Degree of goal alignment  (1 = lo, 5 = hi)', data=master_diag.team_quant)
plt.show()

#Creating qualitative 1-5 score from quantitative cosine similarity results
results = pd.crosstab(master_diag.team_ord['Degree of goal alignment  (1 = lo, 5 = hi)'], master_diag.team_ord['Calculated Labels'])