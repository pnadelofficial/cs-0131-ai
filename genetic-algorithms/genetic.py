import random
import numpy as np 

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
    def __init__(self, generations, population_size, mutation_probability):
        self.generations = generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability

    def fitness(self, chromosome):
        # score = sum([c*BOXES[i+1][0] for i, c in enumerate(chromosome)])
        # if score > MAX_WEIGHT:
        #     score = 0
        # else:
        #     score = sum([c*BOXES[i+1][1] for i, c in enumerate(chromosome)])
        # return score
        score = 0
        for i, c in enumerate(chromosome):
            weight, _ = BOXES[i+1]
            score += weight*c
        if score > MAX_WEIGHT:
            f = 0
        else:
            score = 0
            for i, c in enumerate(chromosome):
                _, cost = BOXES[i+1]
                score += cost*c
            f = score
        return f

    def generate_chromosome(self):
        chromosome = np.zeros(CHROMOSOME_SIZE)
        for i in range(len(chromosome)):
            if random.uniform(0,1) < .5:
                chromosome[i] = 1
            else:
                chromosome[i] = 0
        # score = sum([c*BOXES[i+1][0] for i, c in enumerate(chromosome)])
        score = 0
        for i, c in enumerate(chromosome):
            weight, _ = BOXES[i+1]
            score += weight*c
        if score > MAX_WEIGHT:
            self.generate_chromosome() # create a new one if it exceeds the max weight
        return chromosome
        
    def generate_population(self):
        population = []
        for i in range(self.population_size):
            chromosome = self.generate_chromosome()
            population.append(chromosome)
        return np.array(population)
    
    def find_best(self, population):
        fitnesses = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            fitnesses[i] = self.fitness(population[i])
        max_id = fitnesses.argmax()
        return population[max_id]

    def cull(self, population, to_cull=.5):
        fitnesses = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            f = self.fitness(population[i])
            fitnesses[i] = f
        
        cull_idx = int(to_cull*population.shape[0])
        parents = population[:cull_idx]
        return parents
    
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, CHROMOSOME_SIZE-1)
        child1 = np.hstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.hstack([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, chromosome):
        mutation_point = random.randint(0, CHROMOSOME_SIZE-1)
        if chromosome[mutation_point] == 0:
            chromosome[mutation_point] = 1
        else:
            chromosome[mutation_point] = 0
        return chromosome

    def algorithm(self, to_cull=.5):
        population = self.generate_population()
        for _ in range(self.generations):
            parents = self.cull(population, to_cull=to_cull)
            len_parent = len(parents)//2
            paired_parents = list(zip(parents[:len_parent], parents[len_parent:]))
            
            children = []
            for pair in paired_parents:
                child1, child2 = self.crossover(*pair)
                if random.uniform(0,1) < self.mutation_probability:
                    child1 = self.mutate(child1)
                if random.uniform(0,1) < self.mutation_probability:
                    child2 = self.mutate(child2)
            children.append(child1)
            children.append(child2)

            population = np.vstack([np.array(parents), np.array(children)])

        best = self.find_best(population)

        total_weight = 0
        total_importance = 0
        for i, b in enumerate(best):
            weight, importance = BOXES[i+1]
            total_weight += b*weight
            total_importance += b*importance

        return total_weight, total_importance




