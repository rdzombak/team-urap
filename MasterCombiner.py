import GoalCongruence as gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

E27_1 = gc.pd.read_csv("test_data/First_CheckIn_E27 copy.csv")
E27_2 = gc.pd.read_csv("test_data/Second_CheckIn_E27 copy.csv")
E27_f = gc.pd.read_csv("test_data/Final_CheckIn_E27 copy.csv")

H4L_1 = gc.pd.read_csv("test_data/First_Check_H4l copy.csv")
H4L_2 = gc.pd.read_csv("test_data/Second_Check_H4L copy.csv")
H4L_f = gc.pd.read_csv("test_data/Final_Check_H4L copy.csv")

lst_of_values = [E27_1, E27_2, E27_f, H4L_1, H4L_2, H4L_f]
series_of_labeled = pd.Series([])
for df in lst_of_values:
    series_of_labeled = series_of_labeled.append(df.iloc[:, 3].dropna())

sns.distplot(series_of_labeled, kde=False, rug=True)

#Grabbing all the dataframes and forming one big dataframe with necessary data
master_data = pd.DataFrame([])
for n in np.arange(6):
    series = pd.Series([])
    for df in lst_of_values:
        corrected = gc.verify_identity(df)
        series = series.append(corrected.iloc[:, n], ignore_index=True)
    master_data[lst_of_values[0].columns[n]] = series
master_data = gc.team_renamer(master_data)

master_check = gc.check_accuracy(master_data, gc.cosine_sim(gc.table_by_group(gc.goals_pruner(master_data)), gc.tfidf_embed))
sns.lmplot(x='Calculated Values', y='Degree of goal alignment  (1 = lo, 5 = hi)', data=master_check)
plt.show()

#Creating qualitative 1-5 score from quantitative cosine similarity results
binned_master = gc.binner(master_check, 'Calculated Values', np.arange(1, 6))
results = pd.crosstab(binned_master['Degree of goal alignment  (1 = lo, 5 = hi)'], binned_master['Calculated Labels'])
