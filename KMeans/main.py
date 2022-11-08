import numpy as np
import random
from math import dist
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter
import sys
random.seed(0)
sys.setrecursionlimit(500)

############################################### Train data #########################################################

data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7],
                    converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
labels = []
for label in dates:
  if label < 20000301:
    labels.append('winter')
  elif 20000301 <= label < 20000601:
    labels.append('lente')
  elif 20000601 <= label < 20000901:
    labels.append('zomer')
  elif 20000901 <= label < 20001201:
    labels.append('herfst')
  else: # from 01-12 to end of year
    labels.append('winter')

############################################### Functions for K-Means #################################################

def getLabels(clusters, data, labels):
  labelsPerCluster = []
  for cluster in clusters:
    clusterLabels = []
    for item in cluster:
      clusterLabels.append(labels[item])
    labelsPerCluster.append(clusterLabels)
  return labelsPerCluster


def getCentroids(data, centroids, k):
  # Create a cluster list with k number of lists
  clusters = [[] for _ in range(k)]

  # Save the old centroids
  old_centroids = deepcopy(centroids)

  # Group the new clusters
  for sample_index, datapoint in enumerate(data):
    distances = []

    # Append the distance to the centroids
    for centroid in centroids:
      distances.append(dist(datapoint, centroid))

    # Get the index of the closest centroid
    closest_centroid_index = np.argmin(distances)

    # Append the index of the datapoint to the correct cluster
    clusters[closest_centroid_index].append(sample_index)

  # Calculate new centroids
  for cluster_index, cluster in enumerate(clusters):
    cluster_mean = np.mean(data[cluster], axis=0)
    centroids[cluster_index] = cluster_mean

  # Check if the any centroids have changed
  centroid_distances = [ dist(old_centroids[index], centroids[index]) for index in range(k) ]
  # Return the centroids and the clusters if the centroids haven't changed
  if sum(centroid_distances) == 0:
    return centroids, clusters

  # Calculate the new centroids and clusters
  return getCentroids(data, centroids, k)


def makeScreePlot(k, data):
  # Defining the plot arrays
  k_array = []
  ag_distance_array = []

  # Calculate the mean centroids distance for every "K"
  for k_index in range(k):
    # Create K number of random centroids from the dataset without duplicates (k_index+1 since k starts at 0 )
    random_centroids = random.sample(list(data), k_index+1)

    # Get the calculated centroids together with the made clusters
    centroids, clusters = getCentroids(data, random_centroids, k_index+1)

    centroid_sum_distances = []

    # Get the sum of distances from a centroid to all datapoints in its cluster
    for centroid_index, centroid in enumerate(centroids):
      total_distance = 0
      for datapoint in clusters[centroid_index]:
        total_distance += dist(centroid, data[datapoint])

      # Append the total distance
      centroid_sum_distances.append(total_distance)

    # Append the values for the plot
    k_array.append(k_index+1)
    ag_distance_array.append(np.mean(centroid_sum_distances))

  # Plot the results
  plt.plot(k_array, ag_distance_array)
  plt.xlabel("K")
  plt.ylabel("Mean distance per centroid")
  plt.show()

############################################### Main #########################################################

if __name__ == "__main__":
  # Normalise data
  data_normed = ((data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)))

  # Pick K number of random centroids
  k = 10

  # Plot the aggregate intra-cluster distance per K to see which K is the best to choose
  makeScreePlot(k, data_normed)

  # Pick the best K
  k = 4

  # Create K number of random centroids from the dataset without duplicates
  random_centroids = random.sample(list(data_normed), k)

  # Get the centroids and clusters
  centroids, clusters = getCentroids(data_normed, random_centroids, k)

  # Convert the cluster data to their labels
  labels = getLabels(clusters, data_normed, labels)

  # Print the amount of labels per cluster
  for label_index,row in enumerate(labels):
    print(f"Cluster {label_index+1} has labels: {Counter(labels[label_index]).most_common(4)} ")
