from matplotlib.pyplot import savefig
import pygad
import numpy as np

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""

function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

def fitness_func(solution, solution_idx):
    output = np.sum(solution*function_inputs)
    fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    return fitness

num_generations = 60 # Number of generations.
num_parents_mating = 3 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 5 # Number of solutions in the population.
num_genes = len(function_inputs)

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

def init_population():
    arr = np.array([[-3.86683978, -1.42426841,  1.84769636, -1.62040419,  2.2942856 ,
        -2.09489378],
       [ 0.90129396, -1.0486806 ,  0.83886175, -3.57902594,  3.74970208,
        -0.22878525],
       [-2.88600778, -0.6004969 ,  3.96332325, -2.78628233, -3.39826605,
        -2.74519805],
       [ 2.793464  , -3.70127358, -2.04269264, -2.05806292,  1.57205084,
        -0.23999523],
       [-3.97664555, -1.26734206, -0.72414112, -1.86606189,  1.85095238,
        -3.80947897]])
    
    return arr

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type='rank',
                        crossover_type='two_points',
                       fitness_func=fitness_func,
                       initial_population=init_population(),
                       on_generation=on_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = pygad.load(filename=filename)
loaded_ga_instance.plot_fitness(save_dir='./demo.png')