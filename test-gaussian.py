import functools

import time
import pygad
import numpy as np
import pylab as pl

NUM_GAUSSIANS = 4
MIN_A = 0.1
MAX_A = 10
MIN_X = -10
MAX_X = 10
MIN_SIGMA = 0.1
MAX_SIGMA = 4

def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x-x0)**2/(2 * sigma**2))

def get_function(x_array, magnitudes, centers, sigmas):

    y_array = np.zeros(len(x_array))
    for A, x0, sigma in zip(magnitudes, centers, sigmas):
        y_array += gaussian(x_array, A, x0, sigma)

    return y_array


def generate_reference_data(num_gaussians=4, num_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    magnitudes = np.random.random(num_gaussians) * (MAX_A - MIN_A) + MIN_A
    centers = np.rint(np.random.random(num_gaussians) * (MAX_X - MIN_X) + MIN_X).astype(int)
    sigmas = np.random.random(num_gaussians) * (MAX_SIGMA - MIN_SIGMA) + MIN_SIGMA

    x_sampling = np.random.random(num_points) * (MAX_X - MIN_X) + MIN_X
    y_sampling = get_function(x_sampling, magnitudes, centers, sigmas)
    
    
    print('## Ref DATA: ', magnitudes, centers, sigmas)

    return x_sampling, y_sampling, magnitudes, centers, sigmas

def fitness_func_generic(solution, solution_idx, x_sampling, y_sampling):
    sol_magnitudes = solution[::3]
    sol_centers = solution[1::3]
    sol_sigmas = solution[2::3]
    
    # print(sol_magnitudes, sol_centers, sol_sigmas)

    y_solution = get_function(x_sampling, sol_magnitudes, sol_centers, sol_sigmas)
    abs_diff = np.sum((y_solution - y_sampling)**2)

    return -abs_diff # We want to maximize

def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_choice_idx = np.random.choice(NUM_GAUSSIANS, 2)

        for i in range(NUM_GAUSSIANS):
            if i in random_choice_idx:
                tmp1, tmp2 = parent1[i*3:i*3+3].copy(), parent2[i*3:i*3+3].copy()
                parent1[i*3:i*3+3], parent2[i*3:i*3+3] = tmp2, tmp1

        offspring.append(parent1)
        offspring.append(parent2)
        idx += 1

    return np.array(offspring)

def init_population():
    arr = np.array([[9.828777687373648, -1, 2.8941941389448065, 3.1214341355622133, 2,
        1.5586198325646219, 3.731150283353509, -9, 1.4574595137944635,
        4.928027125093832, -2, 1.5202631450354684],
       [4.976211544944846, -9, 0.9793322101661324, 1.4308113231803548, 6,
        2.7365257258454436, 1.4350755466525917, 0, 3.0989241918763817,
        3.0751175204299392, -1, 1.594960841620833],
       [9.765616301536106, 0, 3.6509279091491402, 1.8937487131902837, -4,
        0.4181245721531718, 1.1803064805457257, 2, 2.869415429967693,
        3.2698674109467087, 4, 3.304951150323801],
       [4.34079267945555, -6, 0.48364833243873684, 6.47270648861852, 3,
        2.223054179326852, 3.7089582915978685, -8, 0.8384969107619181,
        9.793769853451925, -9, 1.9994365165806494],
       [0.645620541484392, 3, 0.9658022263007219, 9.787620313891173, -2,
        3.9115050587420854, 6.608218947893362, 4, 2.6307806626167305,
        0.37395771811222833, 5, 2.873533809980333],
       [4.954209152537596, -1, 2.3248352768018146, 0.7727461553643926, 6,
        3.8316483665863603, 1.674609870187686, -3, 3.467178033670548,
        0.6488070518743082, 1, 1.2601052826920367],
       [3.83336725988063, 7, 0.34342787174989176, 6.270159199468753, 6,
        1.252786616111475, 1.2571686253361587, 2, 2.916085267633314,
        6.367231163704638, 1, 3.074925498045217],
       [8.56327593213708, -8, 3.8521836406933265, 7.537810613012452, 6,
        3.0055042509681904, 8.364274071242026, -1, 1.9535293772931646,
        6.874984688392503, -5, 1.8625252057419774],
       [3.119978414728664, 8, 3.3075516556913485, 5.44002210149905, 0,
        2.0500741551601678, 3.766187600651649, 11, 2.2446486652399247,
        2.3383571335651947, -9, 0.4151012944260418],
       [6.329366120205058, 3, 2.2228694192616367, 5.385840632971384, 8,
        0.5924659364721638, 0.6351305280419405, 4, 1.7735070939976487,
        9.264566452024832, -1, 3.014157174320839],
       [5.9346526969552995, -2, 2.6341879771877057, 3.5739102994498673,
        1, 3.591166085423126, 2.7978212898613157, 2, 3.9327804367964356,
        2.851457499284224, -4, 1.2485457960369142],
       [6.821258669940339, 8, 0.30238883162000907, 8.758629230859617, -5,
        0.7148491734152148, 9.089527119246597, 1, 1.8601501053104443,
        2.0111796522199734, -6, 2.1727603977218384],
       [5.10124016080063, -4, 3.488130725435277, 3.766151549049506, 5,
        3.19997063319342, 6.198859333361211, 0, 0.8462885176019888,
        6.545659377175668, -1, 2.004944474810385],
       [6.902610097613777, 6, 2.2375229888438186, 7.21671044745548, -1,
        3.3079449750965226, 8.546042057358845, 4, 1.308968788546819,
        4.109566776445583, 9, 0.24006230377125382],
       [3.8438693847285235, -3, 1.5027885071862428, 8.798924421365447,
        -8, 2.568871245079223, 8.749103979332205, 7, 3.775308396275222,
        3.150702634416449, 9, 0.8551775267101478],
       [3.5628898200525594, -8, 2.3613572886519814, 0.1418242572381793,
        -16, 1.5324968009056996, 7.614433442590088, -7,
        2.1360711129897925, 4.616143998390505, 0, 2.7514783451226594],
       [4.025309880239535, -7, 3.174048114606116, 8.05119243620658, 0,
        1.167624187536398, 8.143824426209573, -1, 3.5776924387115785,
        0.4472922227392614, 3, 0.7010133334366071],
       [2.690342371762831, -5, 3.646616945365154, 3.602340423275395, -3,
        2.615489244707967, 3.750118656628795, -2, 1.0785574916917717,
        7.541056007188112, -1, 2.194066377201915],
       [3.235068907754611, -9, 0.6432445818421606, 1.5826093938726715, 1,
        0.19125684562637413, 3.857804700932779, 4, 0.3853006605139807,
        3.046987006340373, 3, 1.4206901303413821],
       [8.623320004796094, -7, 3.766950967829815, 9.96572691013853, -2,
        3.488166238204136, 8.156157722754633, 0, 0.9351913154590232,
        9.764448863833803, 1, 0.5267226701780541]], dtype=object)
    
    return arr

def launch(ga: pygad.GA):
    if ga.valid_parameters == False:
        raise ValueError("Error calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

    # Reset the variables that store the solutions and their fitness after each generation. If not reset, then for each call to the run() method the new solutions and their fitness values will be appended to the old variables and their length double. Some errors arise if not reset.
    # If, in the future, new variables are created that get appended after each generation, please consider resetting them here.
    best_solutions = [] # Holds the best solution in each generation.
    best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.
    solutions = [] # Holds the solutions in each generation.
    solutions_fitness = [] # Holds the fitness of the solutions in each generation.

    stop_run = False

    ## INIT
    population = ga.population
    last_generation_parents = ga.last_generation_parents
    last_generation_parents_indices = ga.last_generation_parents_indices
    previous_generation_fitness = ga.previous_generation_fitness
    fitness_func = ga.fitness_func
    last_generation_offspring_crossover = None
    last_generation_offspring_mutation = None
    
    num_offspring = ga.num_offspring
    num_parents_mating = ga.num_parents_mating
    num_genes = ga.num_genes
    num_generations = ga.num_generations
    
    keep_parents = ga.keep_parents
    # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
    # last_generation_fitness = ga._cal_pop_fitness(population,
    #                                               last_generation_parents,
    #                                               last_generation_parents_indices,
    #                                               previous_generation_fitness,
    #                                               fitness_func)
    
    # best_solution, best_solution_fitness, best_match_idx = ga._best_solution(pop_fitness=last_generation_fitness,
    #                                                                      population=population)

    # print(best_solution, best_solution_fitness, best_match_idx)
    
    # Appending the best solution in the initial population to the best_solutions list.
    if ga.save_best_solutions:
        best_solutions.append(best_solution)

    # Appending the solutions in the initial population to the solutions list.
    if ga.save_solutions:
        solutions.extend(population.copy())
        
        
    for generation in range(num_generations):
        
        last_generation_fitness = ga._cal_pop_fitness(population,
                                                    last_generation_parents,
                                                    last_generation_parents_indices,
                                                    previous_generation_fitness,
                                                    fitness_func)
        
        best_solution, best_solution_fitness, best_match_idx = ga._best_solution(pop_fitness=last_generation_fitness,
                                                                            population=population)

        # call on_fitness
        # if not (on_fitness is None):
        #     on_fitness(last_generation_fitness)
        
        # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
        best_solutions_fitness.append(best_solution_fitness)
        
        if ga.save_solutions:
            solutions_fitness.extend(last_generation_fitness)
            
        # Selecting the best parents in the population for mating.
        if callable(ga.parent_selection_type):
            last_generation_parents, last_generation_parents_indices = ga.select_parents(last_generation_fitness, num_parents_mating, ga)
        else:
            last_generation_parents, last_generation_parents_indices = ga.select_parents(last_generation_fitness, num_parents=num_parents_mating)

        # call on_parents
        # if not (self.on_parents is None):
        #     self.on_parents(self, self.last_generation_parents)
        
        # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        if ga.crossover_type is None:
            if num_offspring <= ga.keep_parents:
                last_generation_offspring_crossover = last_generation_parents[0:num_offspring]
            else:
                last_generation_offspring_crossover = np.concatenate((last_generation_parents, population[0:(num_offspring - last_generation_parents.shape[0])]))
        else:
            # Generating offspring using crossover.
            if callable(ga.crossover_type):
                last_generation_offspring_crossover = ga.crossover(last_generation_parents,
                                                                   (num_offspring, num_genes),
                                                                   ga)
            else:
                last_generation_offspring_crossover = ga.crossover(last_generation_parents,
                                                                            offspring_size=(num_offspring, num_genes))

        # call on_crossover
        # if not (self.on_crossover is None):
        #     self.on_crossover(self, self.last_generation_offspring_crossover)
            
            
        # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
        if ga.mutation_type is None:
            last_generation_offspring_mutation = last_generation_offspring_crossover
        else:
            # Adding some variations to the offspring using mutation.
            if callable(ga.mutation_type):
                last_generation_offspring_mutation = ga.mutation(last_generation_offspring_crossover, ga)
            else:
                last_generation_offspring_mutation = ga.mutation(last_generation_offspring_crossover)
             
        # call on_mutation
        # if not (self.on_mutation is None):
        #     self.on_mutation(self, self.last_generation_offspring_mutation)
           
        # Update the population attribute according to the offspring generated.
        if (keep_parents == 0):
            population = last_generation_offspring_mutation
        elif (keep_parents == -1):
            # Creating the new population based on the parents and offspring.
            population[0:last_generation_parents.shape[0], :] = last_generation_parents
            population[last_generation_parents.shape[0]:, :] = last_generation_offspring_mutation
        elif (keep_parents > 0):
            parents_to_keep, _ = ga.select_parents(last_generation_fitness, num_parents=keep_parents)
            population[0:parents_to_keep.shape[0], :] = parents_to_keep
            population[parents_to_keep.shape[0]:, :] = last_generation_offspring_mutation
        
        generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

        previous_generation_fitness = last_generation_fitness.copy()

        # Appending the best solution in the initial population to the best_solutions list.
        if ga.save_best_solutions:
            best_solutions.append(best_solution)

        # Appending the solutions in the initial population to the solutions list.
        if ga.save_solutions:
            solutions.extend(population.copy())

        # call on_generation            
        # # If the callback_generation attribute is not None, then cal the callback function after the generation.
        # if not (self.on_generation is None):
        #     r = self.on_generation(self)
        #     if type(r) is str and r.lower() == "stop":
        #         # Before aborting the loop, save the fitness value of the best solution.
        #         _, best_solution_fitness, _ = self.best_solution()
        #         self.best_solutions_fitness.append(best_solution_fitness)
        #         break
        
        if not ga.stop_criteria is None:
            for criterion in ga.stop_criteria:
                if criterion[0] == "reach":
                    if max(last_generation_fitness) >= criterion[1]:
                        stop_run = True
                        break
                elif criterion[0] == "saturate":
                    criterion[1] = int(criterion[1])
                    if (generations_completed >= criterion[1]):
                        if (best_solutions_fitness[generations_completed - criterion[1]] - best_solutions_fitness[generations_completed - 1]) == 0:
                            stop_run = True
                            break

        if stop_run:
            break

        time.sleep(ga.delay_after_gen)
        
    # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
    last_generation_fitness = ga._cal_pop_fitness(population,
                                                last_generation_parents,
                                                last_generation_parents_indices,
                                                previous_generation_fitness,
                                                fitness_func)

    best_solution, best_solution_fitness, best_match_idx = ga._best_solution(pop_fitness=last_generation_fitness,
                                                                    population=population)

            
    # Save the fitness of the last generation.
    if ga.save_solutions:
        solutions_fitness.extend(last_generation_fitness)

    # Save the fitness value of the best solution.
    best_solution, best_solution_fitness, best_match_idx = ga._best_solution(pop_fitness=last_generation_fitness,
                                                    population=population)
    best_solutions_fitness.append(best_solution_fitness)

    best_solution_generation = np.where(np.array(best_solutions_fitness) == np.max(np.array(best_solutions_fitness)))[0][0]
    # After the run() method completes, the run_completed flag is changed from False to True.
    run_completed = True # Set to True only after the run() method completes gracefully.

    # call on_stop
    # if not (self.on_stop is None):
    #     self.on_stop(self, self.last_generation_fitness)

    # Converting the 'best_solutions' list into a NumPy array.
    best_solutions = np.array(best_solutions)

    # Converting the 'solutions' list into a NumPy array.
    solutions = np.array(solutions)
            
    return best_solution, best_solution_fitness, best_match_idx, generations_completed, best_solutions_fitness

if __name__ == "__main__":
    import sys

    x_range = MAX_X - MIN_X
    x_dense = np.linspace(MIN_X - 0.2 * x_range, MAX_X + 0.2 * x_range, 1000)

    SEED = 102
    x_sampling, y_sampling, magnitudes, centers, sigmas = generate_reference_data(seed=SEED)
    
    def fitness_func(solution, solution_idx):
        return fitness_func_generic(solution, solution_idx, x_sampling, y_sampling)

    fitness_function = fitness_func


    num_generations = 100000
    num_parents_mating = 6

    sol_per_pop = 20
    num_genes = NUM_GAUSSIANS * 3

    # Genes are [A_1, x0_1, sigma_1, A_2, x0_2, sigma_2, ...]
    gene_space = [{'low': MIN_A, 'high': MAX_A}, {'low': MIN_X, 'high': MAX_X}, {'low': MIN_SIGMA, 'high': MAX_SIGMA}] * NUM_GAUSSIANS

    parent_selection_type = "tournament"
    keep_parents = 3

    ## crossover_type = "single_point" # maybe replace with only splitting at multiples of 3

    mutation_type = "random"
    mutation_num_genes = 4

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        gene_space=gene_space,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        # crossover_type=crossover_func,
                        crossover_type='scattered',
                        mutation_type=mutation_type,
                        mutation_num_genes=mutation_num_genes,
                        # random_mutation_min_val=-0.1,
                        # random_mutation_max_val=0.1,
                        #save_solutions=True,
                        initial_population=init_population(),
                        gene_type=[
                            float, int, float, 
                            float, int, float, 
                            float, int, float, 
                            float, int, float, 
                        ],
                        allow_duplicate_genes=False,
                        )

    print('^^^')
    # solution, solution_fitness, solution_idx, generations_completed, best_solutions_fitness = launch(ga=ga_instance)
    print('!!!')
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    sol_magnitudes = solution[::3]
    sol_centers = solution[1::3]
    sol_sigmas = solution[2::3]
    print(f"Parameters of the best solution : {sol_magnitudes}, {sol_centers}, {sol_sigmas}")
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    # Plot actual curve
    pl.plot(x_dense, get_function(x_dense, magnitudes, centers, sigmas), '-k', alpha=0.5)
    # Plot each sub-Gaussian
    for A, x0, sigma in zip(magnitudes, centers, sigmas):
        pl.plot(x_dense, get_function(x_dense, [A], [x0], [sigma]), '-b', alpha=0.2)
    # Show sampling points
    pl.plot(x_sampling, y_sampling, 'o')

    # Show best solution
    sol_magnitudes = solution[::3]
    sol_centers = solution[1::3]
    sol_sigmas = solution[2::3]
    pl.plot(x_dense, get_function(x_dense, sol_magnitudes, sol_centers, sol_sigmas), '-r', linewidth=2, alpha=0.5)
    for A, x0, sigma in zip(sol_magnitudes, sol_centers, sol_sigmas):
        pl.plot(x_dense, get_function(x_dense, [A], [x0], [sigma]), '-r', alpha=0.2)

    pl.savefig('gaussion.png')

    # ga_instance._plot_fitness(generations_completed, best_solutions_fitness, save_dir='./fit.png')
    ga_instance.plot_fitness(save_dir='./fit.png')