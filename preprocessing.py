import pandas as pd
import os

file_dir = os.path.abspath('')
data_set_dir = os.path.join(file_dir, 'dataset')


def read_excel(filename):
    path = os.path.join(data_set_dir, filename)
    return pd.read_excel(path, index_col=0)


facebook_table = read_excel('test.xlsx')

# create index to name and name to index mapping
names = list(facebook_table.columns)
index_to_name_table = pd.DataFrame(names, columns=['name'])
index_to_name_table.to_csv(os.path.join(data_set_dir, 'index_to_name.csv'))


# transform table structure and save the new tables

def transform_table(table, name_list):
    rows = []
    for i, row in enumerate(table.itertuples()):
        src = name_list.index(row[0])
        for j, edge in enumerate(row[1:]):
            if j > i and edge == 1:
                rows.append((src, j))

    res = pd.DataFrame(rows, columns=['source', 'target'])
    return res


facebook_graph = transform_table(facebook_table, names)
facebook_graph.to_csv(os.path.join(data_set_dir, 'facebook_graph.csv'), index=False)

print('# of edges: %d' % len(facebook_graph))
print(facebook_graph.all)
