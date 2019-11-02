from pyspark import SparkConf, SparkContext, SQLContext
from itertools import combinations
import math

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
rows = sqlContext.read.csv('dataset/facebook_graph.csv', header=True).rdd

edges = rows.map(lambda row: (int(row.source), int(row.destination)))
adj_list = edges.groupByKey().mapValues(list)

# compute inverse log of neighbour count for every node
inverse_log = adj_list.map(lambda item: (item[0], 1/math.log(len(item[1]))))


common_neighbours = adj_list.flatMap(lambda item: [(edge, item[0]) for edge in combinations(item[1], 2)]) \
 			    	 		.groupByKey().mapValues(list)

node_to_node_pair = common_neighbours.flatMap(lambda item: [(node, item[0]) for node in item[1]])

# node_to_node_pair: (node1, (node2, node3))
# inverse_log: (node1, inverse_log)
# after join: (node1, ((node2, node3), inverse_log))
# after map: ((ndoe2, node3), inverse_log)

sim_scores = node_to_node_pair.join(inverse_log) \
                              .map(lambda item: item[1]) \
                              .reduceByKey(lambda n1, n2: n1 + n2) \
                              .sortByKey()

sim_scores.saveAsTextFile('output/adamic_adar')

sc.stop()
