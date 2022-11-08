import csv
import random
import copy
import matplotlib.pyplot as plt

random.seed(0)

with open('RondeTafel.csv', 'r', newline='') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=';',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)

    knightsDict = {}
    names = []
    for index, row in enumerate(csv_data):
        # Get the names
        if index <= 1:
            names = row[1:]
            continue
        knightsDict[row[0]] = dict([(names[name_index], float(item)) for name_index, item in enumerate(row[1:])])
    print(knightsDict)

def fitness(individual, knightsDict):
    """"Calculate the total affinity
    Parameters:
    population: The list of knights
    knightsDict: The dictionary of knights with each their affinity to the other knights

    Returns: the total affinity
    """

    totalAffinity = 0
    # Loop through the individual and calculate the affinity to the right neighbours
    for personIndex, person in enumerate(individual):
        leftNeighbour = individual[personIndex-1]
        if personIndex == len(individual)-1:
            rightNeighbour = individual[0]
        else:
            rightNeighbour = individual[personIndex+1]

        #leftAffinity  = knightsDict[person][leftNeighbour] * knightsDict[leftNeighbour][person]
        rightAffinity = knightsDict[person][rightNeighbour] * knightsDict[rightNeighbour][person]
        totalAffinity += (rightAffinity)
    return totalAffinity


def mutate(individual):
    """"Function that swaps 2 random seats with each other
    Paramters:
    individual: List of knights

    Returns: the mutated knights list
    """

    index1 = round(random.uniform(0, len(individual)-1))
    index2 = round(random.uniform(0, len(individual)-1))
    # When the indices are the same, randomise the second one
    while index2 == index1:
        index2 = round(random.uniform(0, len(individual) - 1))

    # Swap the seats
    saveSeat = individual[index1]
    individual[index1] = individual[index2]
    individual[index2] = saveSeat

    return individual


def crossover(eliteParent, randomParent):
    """"Function that makes new offspring
    Parameters:
    eliteParent: a selected 'elite', a list of knights with a high affinity
    randomParent: a list of knights who's affinity was not 'elite'

    Returns: a new list of knights with a random part of the elite parent, filled with the missing knights in the
    order of the random parent"""

    # Randomize the size of the part between 2 and 8
    knightsSize = random.randint(2, len(eliteParent[1]) - 4)
    # Randomize the start index
    startIndex = random.randint(0,len(eliteParent[1])-1)

    newIndividual = []
    # Add the random part of knights to the new list
    for listIndex in range(startIndex, startIndex+knightsSize):
        if listIndex >= len(eliteParent[1])-1 :
            listIndex -= len(eliteParent[1])
        newIndividual.append(eliteParent[1][listIndex])

    # Append the missing knights with the order they are in from randomParent
    for person in randomParent[1]:
        if person not in newIndividual:
            newIndividual.append(person)
    return newIndividual


if __name__ == '__main__':
    epochs = 100
    populationSize = 500

    eliteSize = 25
    randomParentSize = 150

    mutatePercentage = 5

    population = []
    # Create a random population of "populationSize" individuals
    for _ in range(populationSize):
        individual = copy.deepcopy(names)
        random.shuffle(individual)
        population.append(individual)

    rankings = []
    # Rank every individual
    for individual in population:
        rankings.append( [(fitness(individual, knightsDict)), individual] )
    rankings.sort(reverse=True)

    bestAffinity = []
    for _ in range(epochs):

        offspring = []
        # Add the elites
        for eliteTable in range(eliteSize):
            offspring.append(rankings[eliteTable][1])

        # Add random parents who are not elites
        for _ in range(randomParentSize):
            offspring.append(random.choice(population))

        # Append crossovers
        while len(offspring) < populationSize:
            crossoverChild = crossover(random.choice( rankings[:eliteSize]) , random.choice(rankings[eliteSize:]) )
            offspring.append(crossoverChild)

        # Mutate
        for individualIndex, individual in enumerate(offspring):
            if random.randint(0,100) < mutatePercentage :
                mutated = mutate(individual)
                offspring[individualIndex] = mutated

        # Rank the fitness per individual
        rankings = []
        for individual in offspring:
            rankings.append([(fitness(individual, knightsDict)), individual])
        rankings.sort(reverse=True)

        # Append the affinity so that it can be plotted later
        bestAffinity.append(rankings[0][0])

    plt.plot(bestAffinity)
    plt.ylabel('affinity')
    plt.xlabel('epoch')
    plt.show()

    for index in range(0, len(rankings[0][1]) - 1):
        knight = rankings[0][1][index]
        r_neighbor = rankings[0][1][index + 1]
        print(f"({knightsDict[knight][r_neighbor]}x{knightsDict[r_neighbor][knight]}) {r_neighbor}")
    knight = rankings[0][1][11]
    r_neighbor = rankings[0][1][0]
    print(f"({knightsDict[knight][r_neighbor]}x{knightsDict[r_neighbor][knight]}) {r_neighbor}")

    print("Best ranking: ",rankings[0])
