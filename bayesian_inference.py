import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Defining Prior Beliefs
prior_alpha = 2   # Prior belief about number of heads
prior_beta = 2    # Prior belief about number of tails
# both 2 for setting the probability distribution in 0.5
prior_sigma_alpha = 2   # Prior for variance shape parameter
prior_sigma_beta = 2     # Prior for variance scale parameter
# Beta = alpha / beta

# Simulating Coin Flip Data
np.random.seed(123) #any seed number are pwede
true_p = 0.6  # The true bias of the coin (unknown to us)
n_flips = 100  # Number of coin flips
observed_heads = np.random.binomial(n_flips, true_p)  # Simulated coin flips
#np.random.binomial() since we the outcome is only two or binary/dichotomous (n, p) we count the successes of flipping coins in n times
#np.random.normal() more on like continous taking(mean, sd) 

# Step 3: Compute Posterior Parameters
posterior_alpha = prior_alpha + observed_heads
posterior_beta = prior_beta + (n_flips - observed_heads)

# Step 4: Compute Posterior Standard Deviation (Fixing the Issue)
posterior_sigma_alpha = prior_sigma_alpha + n_flips / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((observed_heads - n_flips * true_p) ** 2) / 2

# Step 5: Sample from Posterior Distributions
posterior_p_samples = np.random.beta(posterior_alpha, posterior_beta, size=10000)
posterior_sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))  # Fixed to estimate SD

# Step 6: Plot the Posterior Distributions
plt.figure(figsize=(10, 4))

# Posterior for Probability of Heads (p) - BLUE
plt.subplot(1, 2, 1)
plt.hist(posterior_p_samples, bins=30, density=True, color='royalblue', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of Coin Bias ($p$)', fontsize=12)
plt.xlabel('Probability of Heads ($p$)', fontsize=10)
plt.ylabel('Density', fontsize=10)

# Posterior for Standard Deviation (sigma) - GREEN
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=30, density=True, color='mediumseagreen', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of Standard Deviation ($\sigma$)', fontsize=12)
plt.xlabel('Standard Deviation ($\sigma$)', fontsize=10)
plt.ylabel('Density', fontsize=10)

plt.tight_layout()
plt.show()

# Step 7: Print Summary Statistics
mean_p = np.mean(posterior_p_samples)
std_p = np.std(posterior_p_samples)
print("Mean estimated probability of heads:", mean_p)
print("Standard deviation of estimated probability:", std_p)

mean_sigma = np.mean(posterior_sigma_samples)
std_sigma = np.std(posterior_sigma_samples)
print("Mean estimated standard deviation:", mean_sigma)
print("Standard deviation of standard deviation:", std_sigma)
