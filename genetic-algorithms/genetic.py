import random
import numpy as np 
from typing import Tuple

# constants
MAX_WEIGHT = 250
BOXES = {
    1: (20, 6),
    2: (30, 5),
    3: (60, 8),
    4: (90, 7),
    5: (50, 6),
    6: (70, 9),
    7: (30, 4),
    8: (30, 5),
    9: (70, 4),
    10: (20, 9),
    11: (20, 2),
    12: (60, 1)
}
CHROMOSOME_SIZE = len(BOXES)

class Genetic:
    """
    Abstraction for running the genetic algorithm for the knapsack problem.
    """
    def __init__(self, generations:int, population_size:int, mutation_probability:float) -> None:
        """
        Initializes the Genetic class
        Args:
            generations (int): The number of generations to run the genetic algorithm
            population_size (int): The size of the population for the genetic algorithm
            mutation_probability (float): The probability that a chromosome will be mutated at a random position. Must be between 0 and 1.
        """
        self.generations = generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability

    def fitness(self, chromosome:np.array) -> int:
        """
        The fitness function - first we check the sum of the weights of the objects in the box. If this is less than the max weight, then we return the sum of the importances. Otherwise, we return 0, as a chromosome with a heavier weight than the max weight should be heavily penalized. 
        Args:
            chromosome (np.array): A chromosome whose fitness we want to calculate
        Returns:
            int: The sum of the costs or 0
        """
        score = sum([c*BOXES[i+1][0] for i, c in enumerate(chromosome)])
        if score > MAX_WEIGHT:
            score = 0
        else:
            score = sum([c*BOXES[i+1][1] for i, c in enumerate(chromosome)])
        return score

    def generate_chromosome(self) -> np.array:
        """
        Creates a chromosome at random. If a chromosome is initialized as heavier than the max weight, it is recreated.
        Returns:
            np.array: a randomly initialized chromosome
        """
        chromosome = np.zeros(CHROMOSOME_SIZE)
        for i in range(len(chromosome)):
            # random initialization
            if random.uniform(0,1) < .5: 
                chromosome[i] = 1
            else:
                chromosome[i] = 0
        score = sum([c*BOXES[i+1][0] for i, c in enumerate(chromosome)])
        if score > MAX_WEIGHT:
            self.generate_chromosome() # create a new one if it exceeds the max weight
        return chromosome
        
    def generate_population(self) -> np.array:
        """
        Creates an random initial population
        Returns:
            np.array: a randomly initialized population of shape self.population_size, CHROMOSOME_SIZE
        """
        population = []
        for _ in range(self.population_size):
            chromosome = self.generate_chromosome()
            population.append(chromosome)
        return np.array(population)
    
    def find_best(self, population:np.array) -> np.array:
        """
        Finds the most fit chromosome in a population
        Args:
            population (np.array): The population of shape self.population_size, CHROMOSOME_SIZE
        Returns:
            np.array: The most fit chromosome as determined by the fitness function of shape CHROMOSOME_SIZE
        """
        fitnesses = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            fitnesses[i] = self.fitness(population[i])
        max_id = fitnesses.argmax() # argmax finds the index with the highest value
        return population[max_id]

    def cull(self, population:np.array, to_cull:float=.5) -> np.array:
        """
        The culling function - first we find the fitness for each chromosome, sorting by this value to find the top performing chromosomes. We then cull a to_cull proportion of the population.
        Args:
            population (np.array): The population of shape self.population_size, CHROMOSOME_SIZE
            to_cull (float): proportion to cull, set by default to .5 as per the assignment instructions
        Returns:
            np.array: The parents of the next generation of the genetic algorithm of shape self.population_size*to_cull, CHROMOSOME_SIZE
        """
        fitnesses = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            f = self.fitness(population[i])
            fitnesses[i] = f
        
        population = population[fitnesses.argsort()[::-1]] # argsort returns the indices that would sort the array
        cull_idx = int(to_cull*population.shape[0])
        parents = population[:cull_idx] # culling operation
        return parents
    
    def crossover(self, parent1:np.array, parent2:np.array) -> Tuple[np.array, np.array]: 
        """
        First fringe operation: crossover will take a random point in the chromosome and swap the values between two parents.
        Args:
            parent1 (np.array): first parent to be swapped
            parent2 (np,array): second parent to be swapped
        Returns:
            tuple: the two children which result from the crossover operation
        """
        crossover_point = random.randint(0, CHROMOSOME_SIZE-1)
        child1 = np.hstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.hstack([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, chromosome: np.array) -> np.array:
        """
        Second fringe operation: mutuate will randomly swap a 1 to a 0 or a 0 to a 1.
        Args:
            chromosome (np.array): chromosome to be mutated 
        Returns:
            np.array: mutated chromosome
        """
        mutation_point = random.randint(0, CHROMOSOME_SIZE-1)
        if chromosome[mutation_point] == 0:
            chromosome[mutation_point] = 1
        else:
            chromosome[mutation_point] = 0
        return chromosome

    def algorithm(self, to_cull:float=.5, print_results:bool=True) -> Tuple[int, int, np.array]:
        """
        Runs the genetic algorithm. Initializes a population and for each generation it culls underperforming chromosomes. Then it creates pairs of parents so that they can be passed into `crossover`. Each child is then added to the new population. Once the generations are over, we find the best performer and report that as the result.
        Args:
            to_cull (float): proportion to cull, set by default to .5 as per the assignment instructions
            print_results (bool): whether or not to print the weights and importances of the best performer
        Returns:
            tuple: the total weight, total importance and chromosome of the best performer
        """
        population = self.generate_population() # initalizes population
        for _ in range(self.generations): # iterates through each generation
            parents = self.cull(population, to_cull=to_cull) # culls
            len_parent = len(parents)//2 
            paired_parents = list(zip(parents[:len_parent], parents[len_parent:]))
            
            children = []
            for pair in paired_parents:
                child1, child2 = self.crossover(*pair) # crossover
                if random.uniform(0,1) < self.mutation_probability:
                    child1 = self.mutate(child1) # mutation
                if random.uniform(0,1) < self.mutation_probability:
                    child2 = self.mutate(child2) # mutation
                children.append(child1) 
                children.append(child2)

            population = np.vstack([np.array(parents), np.array(children)]) # add children to population to create the new generation

        best = self.find_best(population) # find the best peroformer

        # get the final weight and importance
        total_weight = 0
        total_importance = 0
        for i, b in enumerate(best):
            if b == 0:
                continue
            weight, importance = BOXES[i+1]
            total_weight += b*weight
            total_importance += b*importance
            if print_results:
                print(f"Object {i+1} included, weight: {weight}, importance: {importance}") # print results

        return total_weight, total_importance, best




