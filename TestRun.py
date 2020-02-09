from GoalCongruence import *

test_data = pd.read_excel(r'~/Desktop/dummytest.xlsx')
master_data = pd.read_csv(r'test_data/master_table.csv')

test_check = Cosine_sim(table_by_group(goals_pruner(test_data)), tfidf_embed)

class_tester = Cosine_sim(table_by_group(goals_pruner(master_data)), tfidf_embed)
