import GoalCongruence as gc
import GoalCongruenceWord2Vec as gcw2v

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

master_data = pd.read_csv(r'~/Desktop/dummytest.csv')
master_check = gc.check_accuracy(master_data, gc.cosine_sim(gc.table_by_group(gc.goals_pruner(master_data)), gcw2v.word2vec_embed))
sns.lmplot(x='Calculated Values', y='Degree of goal alignment  (1 = lo, 5 = hi)', data=master_check)
plt.show()

#Creating qualitative 1-5 score from quantitative cosine similarity results
binned_master = gc.binner(master_check, 'Calculated Values', np.arange(1, 6))
results = pd.crosstab(binned_master['Degree of goal alignment  (1 = lo, 5 = hi)'], binned_master['Calculated Labels'])