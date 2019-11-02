from pyspark import SparkConf, SparkContext, SQLContext
from itertools import combinations

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
rows = sqlContext.read.csv('dataset/facebook_graph.csv', header=True).rdd

edges = rows.map(lambda row: (int(row.source), int(row.destination)))
adj_list = edges.groupByKey().mapValues(list)

def compute_jaccard(pairs):
	p1, p2 = pairs
	n1, n1_neighbours = p1
	n2, n2_neighbours = p2
	n1_neighbours = set(n1_neighbours)
	n2_neighbours = set(n2_neighbours)
	intersection = len(n1_neighbours.intersection(n2_neighbours)) 
	union = len(n1_neighbours.union(n2_neighbours))
	return ((n1,n2), intersection / union)

sim_scores = adj_list.cartesian(adj_list) \
 			    	 .filter(lambda item: item[0][0] < item[1][0]) \
 			    	 .map(compute_jaccard)


sim_scores.saveAsTextFile('output/jaccard_coeffcient')

sc.stop()
