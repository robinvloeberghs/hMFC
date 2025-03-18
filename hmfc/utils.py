"""
Utility functions for the HMFC model.
"""

def convert_mean_to_std_ig_params(mu, beta):
    """Convert the mean parameters of an inverse-gamma distribution 
    to the shape and scale parameters.
    """
    alpha = beta / mu + 1
    return alpha, beta