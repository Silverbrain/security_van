from operator import contains
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
import csv

def read_input_file(input_file_name: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    read_input_file will take a string as input which is the name of the input file
    and returns a tuple of two arrays containing value and weight of the bags.
    '''
    
    bag_wght = []
    bag_val = []
    
    with open(input_file_name, 'r') as bags_file:
        for line in bags_file.readlines(): #going through each line
            if contains(line, 'weight'): #check for 'weight' in the line
                bag_wght.append(float(line.split(':')[1]))
            elif contains(line, 'value'): #going through each line
                bag_val.append(float(line.split(':')[1]))
    #converting the lists to numpy arrays
    return np.array(bag_val), np.array(bag_wght)

def fitness_population(pop:np.ndarray) -> np.ndarray:
    '''
    The `fitness_population` function is used to calculate the fitness of the
    whole population array.
    '''
    
    #weight of each solution in the poplulation
    sol_wght = np.sum(pop * bag_wght, axis = 1)

    #distance between the weight of a solution and the weight limit
    delta = wght_limit - sol_wght
    
    #check if the weight passed the limit; if so, yeild 0
    delta[delta >= 0] = 0
    
    #calculate the penalty for each solution
    penalty = 1 - np.abs(delta) / 100
    
    #calculate the f(s) for each solution
    f = np.sum(pop * (bag_val / bag_wght), axis = 1)

    #penalise the solutions that crossed the weight limit
    fitness = f * penalty
    
    return fitness

def fitness_solution(sol:list) -> float:
    '''
    The `fitness_solution` function will calculate the fitness for a single solution.
    '''

    #weight of the solution
    sol_wght = np.sum(sol * bag_wght)

    #distance between the weight of the solution and the weight limit
    delta = wght_limit - sol_wght
    
    #check if the weight passed the limit; if so, yeild 0
    delta = 0 if delta >= 0 else delta
    
    #calculate the penalty for the solution
    penalty = 1 - np.abs(delta) / 100
    
    #calculate the f(s) for the solution
    f = np.sum(sol * (bag_val / bag_wght))

    #penalise the solutions that crossed the weight limit
    fitness = f * penalty
    
    return fitness

def tournament_selection(fitness : np.ndarray, t_size : int) -> int:
    '''
    To select each parent this function should run. it will get the array of fitness
    values of the whole population and returns the index of the selected parent.
    '''

    #randomly nominate t_size parents' index
    idx_list = np.random.randint(0, len(fitness), t_size)
    #parent nominees
    parent = np.argmax(fitness[idx_list])
    #checking which of the nominated parents is more fit and returning its index.
    return parent

def crossover(A : list, B: list) -> tuple[list, list]:
    '''
    Crossover function takes two parents A & B and after selecting a random crosover
    point it swaps the values of the parents ofter that point and returns the new
    lists as children C & D.
    '''

    #assigning a random number from which the crossover takes place
    pivot = np.random.randint(0, len(A))

    #swaping the values around the pivot point
    C = np.concatenate((A[:pivot], B[pivot:]), axis = 0)
    D = np.concatenate((B[:pivot], A[pivot:]), axis = 0)

    return C, D

def Mutate(child : list, mutation_rate : int) -> list:
    '''
    Mutate function performes the mutation on one child at a time. It takes a
    solution and the mutation rate _number of times mutation process is performed
    on the given solution_ and returns the mutated child.
    '''
    
    for i in range(mutation_rate):
        m_idx = np.random.randint(0, len(child))
        child[m_idx] = not child[m_idx]
    
    return child

def weakest_replace(population : np.ndarray, fit_scores : np.ndarray, new_child : list):
    '''
    weakest_replace takes the whole population, their fitness scores and the new
    child. It compares the fitness of the given child with the weakest fitness
    of the population and replace the weakest if the new child has higher fitness.
    In case of a tie, it randomly choose whether to overwrite the weakest member.
    '''
    
    #calculation the fitness of the new child
    new_child_fitness = fitness_solution(new_child)
    
    #finding the worst solution's index
    worst_idx = np.argmin(fit_scores)
    #finding the worst solution's fitness value
    worst_fitness = fit_scores[worst_idx]

    #checking if the fitness of the new child is higher than the worst solution
    if new_child_fitness > worst_fitness:
        #overwriting the worst solution with the new child
        population[worst_idx] = new_child
        fit_scores[worst_idx] = new_child_fitness
    #in case of the worst solution and the new child have the same fitness then randomly choos one of theme
    elif new_child_fitness == worst_fitness:
        if np.random.binomial(1, 0.5):
            population[worst_idx] = new_child
    
    return population

def EA(Population: np.ndarray, t_size : int, m_rate : int, testing : bool = False) -> np.ndarray:
    '''
    EA "Evolutionary Algorithm" takes an initial population and performs the
    algorithm 10,000 times. The goal of the function is to maximise the fitness
    score. At the end it returns the Populatoin after 10,000 generation.
    '''
    best_fit = 0
    best_fit_duration = 0
    generation = 0

    #saving the fitness value for each solution in a vector.
    fit_scores = fitness_population(Population)
    generation += len(Population)

    while generation < 10000:
        if best_fit_duration >= 100:
            break

        #using tournament selection twice to ditermine the two parents
        parent_A_idx = tournament_selection(fit_scores, t_size)
        parent_B_idx = tournament_selection(fit_scores, t_size)

        #Run crossover function on the selected parents
        C, D = crossover(Population[parent_A_idx], Population[parent_B_idx])

        #Run mutation on the C and D to get the new solutions
        E, F = Mutate(C, m_rate), Mutate(D, m_rate)

        Population = weakest_replace(Population, fit_scores, E)
        Population = weakest_replace(Population, fit_scores, F)
        generation += 2

        if np.max(fit_scores) > best_fit:
            best_fit = np.max(fit_scores)
            best_fit_duration = 0
        else:
            best_fit_duration += 1

    
    return Population, fit_scores, generation

t0 = time.time()    #setting the starting time to calculate the total runing time of the program

# The next two variables are the lists of the weights and values of the bags in input file.
# i.g. the Bag1 in the input file corresponds to the two elemnts bag_wght[0] and bag_val[0]
bag_val, bag_wght = read_input_file('BankProblem.txt')

wght_limit = 285.0 #weghit limit of the van; specified in the input file.

result_file_name = 'results.csv'
with open(result_file_name, 'w') as res_file:
    #if the program is testing the algorithm set True. Set False to evaluate the algorithm
    testing = False
    
    if testing:
        ### trial variables
        p_size = [10, 50, 100, 150, 200]        # initial population size
        t_size = [5]                # number of solutions that would be selected for tournament
        m_rate = [3]                # number of times the mutation process applies to a solution
        random_seed_pool = [0]      # list of fixed number to be used as seed for the random seed.
    else:
        ### trial variables
        p_size = np.arange(10, 201, 10)       # initial population size
        t_size = [[2, int(0.25 * x), int(0.5 * x), int(0.75 * x), x - 1] for x in p_size] # number of solutions that would be selected for tournament
        #m_rate = np.arange(1, 11, 1)          # number of times the mutation process applies to a solution
        m_rate = [np.arange(1, int(0.3 * p_size), 1) for x in p_size]
        random_seed_pool = range(5)           # list of fixed number to be used as seed for the random seed.

        columns = ['population_size', 'tournament_size', 'mutation_rate', 'lap', 'value', 'weight', 'fitness', 'convergence_gen','runing_duration']
        csv_wrt = csv.writer(res_file)
        csv_wrt.writerow(columns)
    
    '''
    In case of testing, the algorithm will run with one setting
    otherwise it will run for each setting and saves the statistics
    of each run in a CSV file.
    '''
    runstats = []
    for p_idx, p in enumerate(p_size):
        for t in t_size[p_idx]:
        #for t in t_size:
            for m in m_rate[p_id]:
                for s in random_seed_pool:
                    ts = time.time()

                    #setting the value of random seed to get the same initial population every time for comparison purposes.
                    np.random.seed(s) 

                    #generating the initial population.
                    Population = np.random.binomial(1, 0.5, (p ,len(bag_val)))

                    #Runing the evolutionary algorithm on the population
                    Population, fit_scores, convergence = EA(Population, t, m, testing)

                    for i in range(len(fit_scores)):
                        idx = np.argmax(fit_scores)
                        sol = Population[idx]
                        if np.sum(sol * bag_wght) > wght_limit:
                            fit_scores[idx] = -1

                    tf = time.time()

                    if(fit_scores[idx] > 0):
                        #['population_size', 'tournament_size', 'mutation_rate', 'lap', 'value', 'weight', 'fitness', 'convergence_gen','runing_duration']
                        data = [p, t, m, s,np.sum(sol * bag_val), np.sum(sol * bag_wght).round(1), fit_scores[idx].round(4), convergence,(tf - ts)]
                        print(data, 'elapsed: {}'.format((tf - t0)))
                        if not testing:
                            csv_wrt.writerow(data)
                    else:
                        print('No valid solution has been found!')

t1 = time.time()
print(f'Total execution time: {round((t1 - t0), 2)}s')