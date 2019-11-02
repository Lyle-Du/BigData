from pyspark import SparkConf, SparkContext, SQLContext
from itertools import combinations

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
rows = sqlContext.read.csv('dataset/facebook_graph.csv', header=True).rdd

edges = rows.map(lambda row: (int(row.source), int(row.destination)))
# adj_list = edges.groupByKey().mapValues(list)
neighbours_count = edges.map(lambda item: (item[0], 1)) \
						.reduceByKey(lambda n1, n2: n1 + n2)

def compute_preferential_attachment(pairs):
	p1, p2 = pairs
	n1, n1_neighbour_count = p1
	n2, n2_neighbour_count = p2
	return ((n1, n2), n1_neighbour_count * n2_neighbour_count)

sim_scores = neighbours_count.cartesian(neighbours_count) \
 			    			 .filter(lambda item: item[0][0] < item[1][0]) \
 			    			 .map(compute_preferential_attachment)


sim_scores.saveAsTextFile('output/preferential_attachment')

sc.stop()
