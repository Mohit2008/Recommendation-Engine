import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
# --- Read Data --- #
a = time.time()
data = pd.read_csv('../data/transaction_data.csv')
data_germany = data.drop('user', 1)
b = time.time()
print("Data imported completed in "+ str((b-a)))
data_ibs= pd.DataFrame(cosine_similarity((data_germany)), index=data_germany.index, columns=data_germany.index)
data_ibs.convert_objects(convert_numeric=True)
c = time.time()
print("Item similarity matrix completed in "+ str((c-a)))
data_neighbours=pd.DataFrame(np.array([data_ibs[c].nlargest(10).index.values for c in data_ibs]), index=data_germany.index)
d = time.time()
print("Neighbouring items completed in "+ str((d-a)))
data_ibs_sort = pd.DataFrame(np.array([data_ibs[c].nlargest(10) for c in data_ibs]), index=data_germany.index)

data_neighbours.to_csv("../records/out.txt", sep="\t")

# def getScore(history, similarities):
#     a=sum(history * similarities) / sum(similarities)
#     return a
#
#
# # Create a place holder matrix for similarities, and fill in the user name column
# data_sims = pd.DataFrame(index=data.index, columns=data.columns)
# data_sims.ix[:, :1] = data.ix[:, :1]
#
# # Loop through all rows, skip the user column, and fill with similarity scores
# for i in range(0, len(data_sims.index)):
#     for j in range(1, len(data_sims.columns)):
#         user = data_sims.index[i]
#         product = data_sims.columns[j]
#
#         if data.ix[i][j] == 1:
#             data_sims.ix[i][j] = 0
#         else:
#             product_top_names = data_neighbours.ix[product]
#             product_top_sims = data_ibs_sort.ix[product]
#             user_purchases = data_germany.ix[user, product_top_names]
#
#             data_sims.ix[i][j] = getScore(user_purchases, product_top_sims)
# e = time.time()
# print("User rating for each product completed in "+ str((e-a)))
# data_sims.to_csv("sims.txt", sep="\t")
#
# # # Get the top songs
# data_recommend = pd.DataFrame(index=data_sims.index, columns=['user', '1', '2', '3', '4', '5', '6'])
# data_recommend.ix[0:, 0] = data_sims.ix[:, 0]
#
# # Instead of top song scores, we want to see names
# for i in range(0, len(data_sims.index)):
#     data_recommend.ix[i, 1:] = data_sims.ix[i, :].sort_values(ascending=False).ix[1:7, ].index.transpose()
# f = time.time()
# print("Predictions ready in "+ str((f-a)))
# # Print a sample
# print(data_recommend.ix[:10, :4])
