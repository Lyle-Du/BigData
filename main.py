import pandas as pd
import os

file_dir = os.path.abspath('')
data_set_dir = os.path.join(file_dir, 'dataset')

# original graph
df = pd.read_csv("dataset/facebook_graph.csv")

# training graph, with some edges hidden
# set random_state in order to sample same samples
train_df = df.sample(frac=0.5, random_state=123)
# keep undirected graph property after sampling
reindex_train_df = train_df.reindex(columns=['target', 'source'])
reindex_train_df.columns = ['source', 'target']
train_df = pd.concat([train_df, reindex_train_df], ignore_index=True)
train_df = train_df.sort_values(by=['source', 'target']).reset_index(drop=True)
train_df.to_csv(os.path.join(data_set_dir, 'training_set.csv'), index=False)


