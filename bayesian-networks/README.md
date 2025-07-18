# Bayesian Networks


To classify the given data, you can use this script: `bayesian_networks.py`. To do so, simply run the script with `python bayesian_networks.py -vap`. The `-v` flag will run the model with only the velocity feature. The `-a` flag will run the model with both velocity and acceleration with likelihoods taken from their histogram. Finally the `-p` flag will run the model with acceleration likelihood values taken from a parametric estimation of the underlying distribution of accelerations.

You may also import the `BayesianNetwork` module into a Python script with `from bayesian_networks import BayesianNetwork`.

See further details on model performance in the report.