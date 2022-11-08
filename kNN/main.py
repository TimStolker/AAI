import numpy as np
from collections import Counter
from math import dist

################################################# Train data ######################################################

# Get the train data from the csv
data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7],
                     converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

# Get the dates from the csv
dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
labels = []

# Get the seasonal labels from the dates
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

################################################# Test data ######################################################

# Get the validation data from the csv
validation_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7],
                     converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

# Get the dates from the csv
validation_dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])

# Get the seasonal labels from the dates
validation_labels = []
for label in validation_dates:
  if label < 20010301:
    validation_labels.append('winter')
  elif 20010301 <= label < 20010601:
    validation_labels.append('lente')
  elif 20010601 <= label < 20010901:
    validation_labels.append('zomer')
  elif 20010901 <= label < 20011201:
    validation_labels.append('herfst')
  else: # from 01-12 to end of year
    validation_labels.append('winter')

################################################# Days test ######################################################

# Get the days data from the csv
days_data = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7],
                     converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

################################################# KNN algorithm ######################################################

# Normalise the imported data with the train data
def normaliseData(data_to_normalise, data):
  return ( (data_to_normalise-data.min(axis=0)) / (data.max(axis=0)-data.min(axis=0)) )

# Returns the predicted label of the data point with k as number of neighbours to check
def knn(k, data_point, data, labels):
  # Declare the used lists
  distances = []
  k_neighbor_labels = []

  # Fill the distance list with euclidean distances between data_point and item
  for item in data:
    distances.append(dist(data_point, item))

  # Sort the "k" number of distances
  k_indices = np.argsort(distances)[:k]

  # Fill the k_neighbour_labels with the labels gathered from the indices of the closest distance
  for distanceIndex in k_indices:
    k_neighbor_labels.append(labels[distanceIndex])
  most_common_count = Counter(k_neighbor_labels)

  # Check for tie
  if len(most_common_count) > 1:
    if most_common_count.most_common(2)[0][1] == most_common_count.most_common(2)[1][1]:
      # Call knn again with k reduced by 1
      return knn(k-1, data_point, data, labels)
  # Return the most common label
  return most_common_count.most_common(1)[0][0]

# Returns the accuracy of correctness over the train data
def getAccuracy(k, train_data, train_labels, validation_data, validation_labels):
  correctCount = 0
  # Loop through every data point in the validation data
  for labelIndex, datapoint in enumerate(validation_data):
    prediction = knn(k, datapoint, train_data, train_labels)
    # Check is the prediction is correct
    if prediction == validation_labels[labelIndex]:
      correctCount += 1

  return correctCount/ len(validation_data)*100

################################################# Call tree ######################################################

if __name__ == "__main__":
  # Normalise the data sets
  data_normed            = normaliseData(data, data)
  validation_data_normed = normaliseData(validation_data, data)
  days_normed            = normaliseData(days_data, data)

  # Use k as 63 since this has the highest accuracy
  k = 63

  # Get the accuracy
  result = getAccuracy(k, data_normed, labels, validation_data_normed, validation_labels)
  print(f"Accuracy: {result} %")

  # Print the predicted season per day
  for index, day in enumerate(days_normed):
    print(f"Dag {index+1} is in de {knn(k, day, data_normed, labels)}")
