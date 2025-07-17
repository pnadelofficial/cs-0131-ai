import numpy as np

def read_likelihoods():
    with open("likelihood.txt", "r") as f:
        bird_likelihood, aero_likelihood = f.readlines()
        bird_likelihood = np.array([float(datum) for datum in bird_likelihood.split()])
        aero_likelihood = np.array([float(datum) for datum in aero_likelihood.split()])
    
    return bird_likelihood, aero_likelihood

def read_dataset():
    with open("dataset.txt", "r") as f:
        dataset = f.read()
        dataset = np.array([[float(datum) for datum in row.split()] for row in dataset.strip().split("\n")])
        bird_velocities = dataset[:10]
        aero_velocities = dataset[10:]

    return bird_velocities, aero_velocities

def read_testing():
    with open("testing.txt", "r") as f:
        testing = f.read()
        testing = np.array([[float(datum) for datum in row.split()] for row in testing.strip().split("\n")])
        bird_test = testing[[0, 1, 2, 5, 9]] # from assignment
        aero_test = testing[[3, 4, 6, 7, 8]] # from assignment
   
    return bird_test, aero_test

BIRD_LIKELIHOODS, AERO_LIKELIHOODS = read_likelihoods()
BIRD_VELOCITIES, AERO_VELOCITIES = read_dataset()
BIRD_TEST, AERO_TEST = read_testing()
PROB_BIRD = .5
PROB_AERO = .5
LEN_OBSERVATIONS = BIRD_VELOCITIES.shape[0]
LEN_VELOCITIES = BIRD_VELOCITIES.shape[-1]
LEN_LIKELIHOODS = BIRD_LIKELIHOODS.shape[-1]
MAJOR_TRANSITION = .9
MINOR_TRANSITION = .1
BIRD = 1
AERO = 0

class BayesianNetwork:
    def __init__(self, accel:bool=False):
        self.accel = accel
    
    def velocity_to_likelihood(self, velocity_sequence, bird=True):
        indices = (velocity_sequence * 2).astype(int) # velocity resolution only half of likelihood
        max_likelihood_index = len(BIRD_LIKELIHOODS) - 1
        indices = np.clip(indices, 0, max_likelihood_index) # clip to size of likelihoods

        if bird:
            return BIRD_LIKELIHOODS[indices]
        else:
            return AERO_LIKELIHOODS[indices]

    def accel_to_likelihood(self, accels):
        max_datum = np.max(accels)
        counts, bin_edges = np.histogram(accels, bins=LEN_VELOCITIES, density=True, range=(0,max_datum))
        bin_indices = np.digitize(accels, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, LEN_VELOCITIES - 1)
        likelihoods = counts[bin_indices]
        return likelihoods
    
    def fit(self):
        for i in range(LEN_OBSERVATIONS):
            bird_velocities = np.nan_to_num(BIRD_VELOCITIES[i], nan=0)
            aero_velocities = np.nan_to_num(AERO_VELOCITIES[i], nan=0)

            bird_veloc_likelihoods = self.velocity_to_likelihood(bird_velocities)
            aero_veloc_likelihoods = self.velocity_to_likelihood(aero_velocities, bird=False)

            if self.accel:
                bird_accel = np.gradient(bird_velocities)
                aero_accel = np.gradient(aero_velocities)
            
                bird_accel_likelihoods = self.accel_to_likelihood(bird_accel)
                aero_accel_likelihoods = self.accel_to_likelihood(aero_accel)
            
            for j in range(LEN_LIKELIHOODS):
                bird_likelihood = bird_veloc_likelihoods[j]
                aero_likelihood = aero_veloc_likelihoods[j]

                if self.accel:
                    bird_likelihood *= bird_accel_likelihoods[j]
                    aero_likelihood *= aero_accel_likelihoods[j]
                
                if j != 0:
                    prob_bird = bird_normed * MAJOR_TRANSITION + aero_normed * MINOR_TRANSITION
                    prob_aero = aero_normed * MAJOR_TRANSITION + bird_normed * MINOR_TRANSITION
                else:
                    prob_bird = PROB_BIRD
                    prob_aero = PROB_AERO

                bird_post_update = bird_likelihood * prob_bird
                aero_post_update = aero_likelihood * prob_aero

                total = bird_post_update + aero_post_update
                if total > 0:
                    bird_normed = bird_post_update/total
                    aero_normed = aero_post_update/total
        self.bird_normed = bird_normed
        self.aero_normed = aero_normed
        print(self.bird_normed, self.aero_normed)

    def predict(self, test_sequence):
        preds = []
        probs = []
        
        test_sequence = np.nan_to_num(test_sequence, nan=0)
        bird_veloc_likelihoods = self.velocity_to_likelihood(test_sequence)
        aero_veloc_likelihoods = self.velocity_to_likelihood(test_sequence, bird=False)
        
        if self.accel:
            test_sequence_accel = np.gradient(test_sequence)
            test_accel_likelihoods = self.accel_to_likelihood(test_sequence_accel)
        
        for i in range(LEN_LIKELIHOODS):
            bird_likelihood = bird_veloc_likelihoods[i]
            aero_likelihood = aero_veloc_likelihoods[i]

            if self.accel:
                bird_likelihood *= test_accel_likelihoods[i]
                aero_likelihood *= test_accel_likelihoods[i]

            bird_post_update = bird_likelihood * self.bird_normed
            aero_post_update = aero_likelihood * self.aero_normed

            # if i != 0:
            #     prob_bird = self.bird_normed * MAJOR_TRANSITION + self.aero_normed * MINOR_TRANSITION
            #     prob_aero = self.aero_normed * MAJOR_TRANSITION + self.bird_normed * MINOR_TRANSITION
            # else:
            #     prob_bird = PROB_BIRD
            #     prob_aero = PROB_AERO
            
            # bird_post_update = bird_likelihood * prob_bird
            # aero_post_update = aero_likelihood * prob_aero

            total = bird_post_update + aero_post_update
            if total > 0:
                bird_normed = bird_post_update/total
                aero_normed = aero_post_update/total
            # else:
            #     bird_normed = prob_bird
            #     aero_normed = prob_aero

            pred = BIRD if bird_normed > aero_normed else AERO
            preds.append(pred)
            probs.append(bird_normed)

        return np.array(preds), np.array(probs)         
