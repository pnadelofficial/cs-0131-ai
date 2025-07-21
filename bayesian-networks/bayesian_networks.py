import numpy as np
import scipy.stats as st
import argparse
from typing import Union, Tuple, Dict, List

def read_likelihoods():
    """Reads in likelihood.txt as arrays"""
    with open("likelihood.txt", "r") as f:
        bird_likelihood, aero_likelihood = f.readlines()
        bird_likelihood = np.array([float(datum) for datum in bird_likelihood.split()])
        aero_likelihood = np.array([float(datum) for datum in aero_likelihood.split()])
    
    return bird_likelihood, aero_likelihood

def read_dataset():
    """Reads in dataset.txt as arrays"""
    with open("dataset.txt", "r") as f:
        dataset = f.read()
        dataset = np.array([[float(datum) for datum in row.split()] for row in dataset.strip().split("\n")])
        bird_velocities = dataset[:10]
        aero_velocities = dataset[10:]

    return bird_velocities, aero_velocities

def read_testing():
    """Reads in testing.txt as arrays"""
    with open("testing.txt", "r") as f:
        testing = f.read()
        testing = np.array([[float(datum) for datum in row.split()] for row in testing.strip().split("\n")])
   
    return testing

# constants
## data
BIRD_LIKELIHOODS, AERO_LIKELIHOODS = read_likelihoods()
BIRD_VELOCITIES, AERO_VELOCITIES = read_dataset()
TESTING = read_testing()

## for Bayesian network
PROB_BIRD = .5
PROB_AERO = .5
LEN_OBSERVATIONS = BIRD_VELOCITIES.shape[0]
LEN_VELOCITIES = BIRD_VELOCITIES.shape[-1]
LEN_LIKELIHOODS = BIRD_LIKELIHOODS.shape[-1]
MAJOR_TRANSITION = .9
MINOR_TRANSITION = .1
BIRD = 1
AERO = 0

## evaluation
BIRD_TESTING = [0, 1, 2, 5, 9]
AERO_TESTING = [3, 4, 6, 7, 8]
GOLD = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])

class BayesianNetwork:
    """
    Abstraction for fitting and predicting based on the given data
    """
    def __init__(self, accel:Union[str, None]="nonparam") -> None:
        """
        Initializes the network
        Args:
            accel (str or None): What kind of acceleration you want to use to predict with. "nonparam" will use a nonparametic estimation of the acceleration likelihoods, while "param" will fit a Gaussian distribution to the accelerations and use those likelihoods. None will only used velocity.
        """
        self.accel = accel
    
    def velocity_to_likelihood(self, velocity_sequence:np.array, bird:bool=True) -> np.array:
        """
        Takes a sequence of velocities and determines the likelihood values for them
        Args:
            velocity_sequence (np.array): The sequence to determine the likelihoods for
            bird (bool): Whether the results should be indexed on the bird or plane likelihoods
        Returns:
            np.array: The likelihood array for the given velocities
        """
        indices = (velocity_sequence * 2).astype(int) # velocity resolution only half of likelihood
        indices = np.clip(indices, 0, LEN_LIKELIHOODS - 1) # clip to size of likelihoods

        # index the likelihoods
        if bird:
            return BIRD_LIKELIHOODS[indices] 
        else:
            return AERO_LIKELIHOODS[indices]

    def accel_to_likelihood(self, accels:np.array) -> np.array:
        """
        Takes a sequence of accelerations and determines the likelihood values in a nonparametric manner, i.e. with a histogram
        Args:
            accels (np.array): The sequence to determine the likelihoods for
        Returns:
            np.array: The likelihood array for the given accelerations
        """
        max_datum = np.max(accels) # find max value
        counts, bin_edges = np.histogram(accels, bins=LEN_VELOCITIES, density=True, range=(0,max_datum)) # create a histogram
        bin_indices = np.digitize(accels, bin_edges) - 1 # digitize will assign values of counts to bin edges. I'm not sure why, but it is 1-indexed so I have to subtract 1
        bin_indices = np.clip(bin_indices, 0, LEN_VELOCITIES - 1) # clip to the size of velocities
        likelihoods = counts[bin_indices] # index the likelihoods
        return likelihoods

    def fit_parametric(self) -> None:
        """
        Determines the mean and standard deviation of all of the acceleration data and stores them in self
        """
        all_bird_accels = []
        all_aero_accels = []

        for i in range(LEN_OBSERVATIONS):
            bird_velocities = np.nan_to_num(BIRD_VELOCITIES[i], nan=0)
            aero_velocities = np.nan_to_num(AERO_VELOCITIES[i], nan=0)

            bird_accel = np.gradient(bird_velocities)
            aero_accel = np.gradient(aero_velocities)

            all_bird_accels.extend(bird_accel)
            all_aero_accels.extend(aero_accel)
        
        all_bird_accels = np.array(all_bird_accels)
        all_aero_accels = np.array(all_aero_accels)

        self.bird_accel_mu = np.mean(all_bird_accels)
        self.bird_accel_sigma = np.std(all_bird_accels, ddof=1)
        self.aero_accel_mu = np.mean(all_aero_accels)
        self.aero_accel_sigma = np.std(all_aero_accels, ddof=1)

    def accel_to_likelihood_param(self, accels:np.array, bird:bool=True) -> np.array:
        """
        Fits a normal distribution to the acceleration data with the stored mean and standard deviation data from above
        Args: 
            accel (np.array): The accelerations to fit
            bird (bool): Whether the results should use the statistics of the bird or plane
        Return:
            np.array: The probability distribution for the fit Gaussian
        """
        if bird:
            return st.norm.pdf(accels, loc=self.bird_accel_mu, scale=self.bird_accel_sigma)
        else:
            return st.norm.pdf(accels, loc=self.aero_accel_mu, scale=self.aero_accel_sigma)

    def fit(self) -> None:
        """
        Trains the Bayesian Network so that we can then predict on future data. I borrow the name fit from `scikit-learn` which implements several models with this fit -> predict sequence.
        """
        for i in range(LEN_OBSERVATIONS): # loop through all of the observations
            if self.accel == "param":
                self.fit_parametric() # fit parametic if needed
            
            # deal with nans
            bird_velocities = np.nan_to_num(BIRD_VELOCITIES[i], nan=0)
            aero_velocities = np.nan_to_num(AERO_VELOCITIES[i], nan=0)

            # look up observed likelihoods
            bird_veloc_likelihoods = self.velocity_to_likelihood(bird_velocities)
            aero_veloc_likelihoods = self.velocity_to_likelihood(aero_velocities, bird=False)

            if self.accel == "nonparam": # nonparametric acceleration
                bird_accel = np.gradient(bird_velocities)
                aero_accel = np.gradient(aero_velocities)
            
                bird_accel_likelihoods = self.accel_to_likelihood(bird_accel)
                aero_accel_likelihoods = self.accel_to_likelihood(aero_accel)
            
            if self.accel == "param": # parametic acceleration
                bird_accel = np.gradient(bird_velocities)
                aero_accel = np.gradient(aero_velocities)

                bird_accel_likelihoods = self.accel_to_likelihood_param(bird_accel)
                aero_accel_likelihoods = self.accel_to_likelihood_param(aero_accel, bird=False)
            
            for j in range(LEN_LIKELIHOODS): # loop through all likelihoods
                # individal likelihoods
                bird_likelihood = bird_veloc_likelihoods[j]
                aero_likelihood = aero_veloc_likelihoods[j]

                # multiply velocity likelihoods by acceleration likelihoods if needed
                if self.accel == "nonparam":
                    bird_likelihood *= bird_accel_likelihoods[j]
                    aero_likelihood *= aero_accel_likelihoods[j]
                
                if self.accel == "param":
                    bird_likelihood *= bird_accel_likelihoods[j]
                    aero_likelihood *= aero_accel_likelihoods[j]
                
                if j != 0:
                    # except on first likelihood we use the transition probabilities
                    prob_bird = bird_normed * MAJOR_TRANSITION + aero_normed * MINOR_TRANSITION
                    prob_aero = aero_normed * MAJOR_TRANSITION + bird_normed * MINOR_TRANSITION
                else:
                    # on the first likelihood we use our priors
                    prob_bird = PROB_BIRD
                    prob_aero = PROB_AERO

                # posterior update    
                bird_post_update = bird_likelihood * prob_bird
                aero_post_update = aero_likelihood * prob_aero

                # normalization
                total = bird_post_update + aero_post_update
                if total > 0:
                    bird_normed = bird_post_update/total
                    aero_normed = aero_post_update/total
        
        # store for prediction
        self.bird_normed = bird_normed
        self.aero_normed = aero_normed
        print(f"Final probabilities: Bird = {self.bird_normed:.4f}, Aero = {self.aero_normed:.4f}")

    def predict(self, test_sequence:np.array) -> Tuple[np.array, np.array]:
        """
        Predicts the class given a velocity sequence
        Args:
            test_sequence (np.array): The test sequence to classify
        Returns:
            np.array: The list of predictions at each likelihood step
            np.array: The probability of predictions at each likelihood step
        """
        preds = []
        probs = []
        
        # look up likelihoods
        test_sequence = np.nan_to_num(test_sequence, nan=0)
        bird_veloc_likelihoods = self.velocity_to_likelihood(test_sequence)
        aero_veloc_likelihoods = self.velocity_to_likelihood(test_sequence, bird=False)
        
        if self.accel == "nonparam": # nonparametric acceleration
            test_sequence_accel = np.gradient(test_sequence)
            bird_accel_likelihoods = self.accel_to_likelihood(test_sequence_accel)
            aero_accel_likelihoods = self.accel_to_likelihood(test_sequence_accel)
        
        if self.accel == "param": # parametric acceleration
            test_sequence_accel = np.gradient(test_sequence)
            bird_accel_likelihoods = self.accel_to_likelihood_param(test_sequence_accel)
            aero_accel_likelihoods = self.accel_to_likelihood_param(test_sequence_accel, bird=False)
        
        # initalize the fit priors from training
        bird_normed = self.bird_normed
        aero_normed = self.aero_normed

        for i in range(LEN_LIKELIHOODS): # loop through the likelihoods
            # individual likelihoods
            bird_likelihood = bird_veloc_likelihoods[i]
            aero_likelihood = aero_veloc_likelihoods[i]

            # multiply velocity likelihoods by acceleration likelihoods if needed
            if self.accel == "nonparam":
                bird_likelihood *= bird_accel_likelihoods[i]
                aero_likelihood *= aero_accel_likelihoods[i]
            
            if self.accel == "param":
                bird_likelihood *= bird_accel_likelihoods[i]
                aero_likelihood *= aero_accel_likelihoods[i]

            # posterior update 
            bird_post_update = bird_likelihood * self.bird_normed
            aero_post_update = aero_likelihood * self.aero_normed

            # normalization
            total = bird_post_update + aero_post_update
            if total > 0:
                bird_normed = bird_post_update/total
                aero_normed = aero_post_update/total

            # determine predictions
            pred = BIRD if bird_normed > aero_normed else AERO # if one probability is larger than the other
            preds.append(pred)
            probs.append(bird_normed)

        return np.array(preds), np.array(probs)         

def evaluate(test_ids:List, models:Dict) -> Dict:
    """
    Evaluates the test sequences
    Args:
        test_ids (list): The ids of what test seqeunces to evaluate
        model (dict): A mapping of names and models
    Returns:
        dict: The results of the evaluation
    """
    test_sequences = TESTING[test_ids] # select the asked for testing sequence

    def analyze_preds_probs(preds:np.array, probs:np.array, model_type:str) -> Tuple[np.array, int, float]:
        """
        Analyzes the results from the prediction process by determining the probabilities that agree with the predictions and the taking the last element of the preds array
        Args:
            preds (np.array): The prediction array from predict
            probs (np.array): The probability array from predict
            model_type (str): The name of the model taken from the user input
        Returns:
            np.array: The confidences of the prediction
            int: The final prediction
            float: The confidence of that prediction
        """
        confidences = np.where(preds == BIRD, probs, 1 - probs)
        final_pred = preds[-1]
        final_confidence = confidences[-1]
        print(f"Final prediction {'Bird' if final_pred == BIRD else 'Plane'} | Confidence: {final_confidence:.4f} | Model Type: {model_type}")
        return confidences, final_pred, final_confidence
    
    results = {}
    for model_name in models: # loop through the models
        print(f"Results for {model_name} Model")
        model_results = []
        for test_sequence in test_sequences: # loop through the test sequences
            preds, probs = models[model_name].predict(test_sequence)
            res = analyze_preds_probs(preds, probs, model_name) # analyze
            model_results.append(res)
        results[model_name] = model_results
        final_preds = np.array([tup[1] for tup in model_results]) 
        accuracy = np.sum(final_preds == GOLD)/len(GOLD) # calculate accuracy
        print(f"Final accuracy: {accuracy:.4f}")
        print()    
    return results

def main() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="test_ids", nargs="*", type=int, default=[])
    parser.add_argument("-v", "--velocity", action="store_true")
    parser.add_argument("-a", "--nonparametricacceleration", action="store_true")
    parser.add_argument("-p", "--parametricacceleration", action="store_true")
    args = parser.parse_args()

    test_ids = args.test_ids
    if test_ids == []:
        test_ids = list(range(len(GOLD))) # if no test_ids are given, all are given

    models = {}
    if args.velocity:
        bn_vel = BayesianNetwork(accel=None)
        bn_vel.fit()
        models['Velocity'] = bn_vel
    if args.nonparametricacceleration:
        bn_nonparam = BayesianNetwork(accel="nonparam")
        bn_nonparam.fit()
        models['Nonparametic Acceleration'] = bn_nonparam
    if args.parametricacceleration:
        bn_param = BayesianNetwork(accel="param")
        bn_param.fit()
        models['Parametric Acceleration'] = bn_param
    
    print()
    return evaluate(test_ids, models)

if __name__ == '__main__':
    main()