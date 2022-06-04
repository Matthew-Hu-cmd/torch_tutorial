import numpy as np
import matplotlib.pyplot as plt


DNA_SIZE = 19           # DNA length
POP_SIZE = 200           # Population size
MATING_RATE = 0.8        # Gene recombination probability
MUTATION_RATE = 0.005    # Mutation probability
N_GENERATIONS = 500      # Number of generations
X_BOUND = [0.01, 5]         # Upper and lower bounds
min_VALUE = {}           #用来储存每一代中的最小值
mini_value = 0


# The object equation

def F(x): return np.exp(x)/x + x/np.exp(x)
# Obtain fitness for selection
def Get_Fitness(val): return 1/val + 1e-6


# Decode binary DNA to range of x bound
def Translate_DNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2**DNA_SIZE-1) * X_BOUND[1]


# Get survived individual
def Select(pop, fitness):
    index = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=fitness/fitness.sum())
    return pop[index]


# Gene recombination
def Mating (parent, pop):
    if np.random.rand() < MATING_RATE:
        index = np.random.randint(0, POP_SIZE, size=1)

        cross_points = np.random.randint(0, 2, size=DNA_SIZE)
        parent[cross_points] = pop[index, cross_points]
    return parent


# Mutation process
def Mutation(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))


plt.ion()
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))
plt.show()

for _ in range(N_GENERATIONS):
    F_values = F(Translate_DNA(pop))

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(Translate_DNA(pop), F_values, s=200, lw=0, c='red', marker='*')
    plt.pause(0.05)

    fitness = Get_Fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :], np.min(F_values))
    min_VALUE[_] = np.min(F_values)
    pop = Select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = Mating(parent, pop_copy)
        child = Mutation(child)
        parent[:] = child

    if _ == N_GENERATIONS-1:
        for index in range(1, N_GENERATIONS-1):
            if min_VALUE[index] <= min_VALUE[index-1]:
                mini_value = min_VALUE[index]

        print('最小值为', mini_value)

plt.ioff(); plt.show()
