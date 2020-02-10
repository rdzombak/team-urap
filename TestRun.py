from GoalCongruence import *

test_data = pd.read_excel(r'~/Desktop/dummytest.xlsx')
master_data = pd.read_csv(r'test_data/master_table.csv')

sorted_test_data= series_by_team(goals_pruner(test_data))
sorted_master_data = series_by_team(goals_pruner(master_data))

test_tester = Cosine_Sim(sorted_test_data, tfidf_embed)

master_tester = Cosine_Sim(sorted_master_data, tfidf_embed)