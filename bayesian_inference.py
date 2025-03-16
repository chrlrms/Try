import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

# Define the range of possible values for mu
mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1

# Define a uniform prior distribution
#uniform_dist = np.ones_like(mu)  # Uniform prior (flat distribution)
uniform_dist = uniform_dist / uniform_dist.sum()  # Normalize
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale =0.2)  # Beta prior
beta_dist = beta_dist / beta_dist.sum()  # Normalize
# Plot the prior
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Prior Distribution")
plt.show()

# Define the likelihood function
def likelihood_func(datum, mu_values):
    return sts.norm.pdf(datum, mu_values, scale=0.1)

# Compute likelihood for observed data point (1.7m)
likelihood_out = likelihood_func(1.7, mu)

# Normalize the likelihood
likelihood_out = likelihood_out / likelihood_out.sum()

# Plot the likelihood
plt.plot(mu, likelihood_out, label="Likelihood")
plt.xlabel("Value of $\mu$")
plt.ylabel("Probability Density/Likelihood")
plt.legend()
plt.title("Likelihood Function of $/mu$ given observation 1.7m")
plt.show()

# Compute the posterior (unnormalized)
unnormalized_posterior = likelihood_out * uniform_dist

# Normalize the posterior
#posterior = unnormalized_posterior / unnormalized_posterior.sum()

# Plot the posterior
plt.plot(mu, unnormalized_posterior, label="Posterior")
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.legend()
plt.title("Unnormalized Posterior")
plt.show()
