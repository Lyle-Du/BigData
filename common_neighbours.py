from pyspark import SparkConf, SparkContext, SQLContext
from itertools import combinations
import datetime

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
rows = sqlContext.read.csv('dataset/facebook_graph.csv', header=True).rdd

edges = rows.map(lambda row: (int(row.source), int(row.target)))
adj_list = edges.groupByKey().mapValues(list)
print(adj_list.collect())

# input format: (node, [node1, node2, node3, ....])
common_neighbours = adj_list.flatMap(lambda item: [(edge, item[0]) for edge in combinations(item[1], 2)]) \
    .groupByKey().mapValues(list)

# input format: ((node1, node2), [node3, node4, node5, ...])
sim_scores = common_neighbours.map(lambda item: (item[0], len(item[1]))) \
    .reduceByKey(lambda n1, n2: n1 + n2) \
    .sortByKey()

sim_scores.saveAsTextFile('output/common_neighbours' + str(datetime.datetime.now()))

sc.stop()
