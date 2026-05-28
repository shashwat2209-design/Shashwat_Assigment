#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


# # Question 1

# In[2]:


# Data
admissions = np.array([32, 28, 35, 30, 29, 27, 31, 34, 33, 30])

# 1. Mean, Median, Mode
mean_adm = np.mean(admissions)
median_adm = np.median(admissions)

mode_result = stats.mode(admissions)
mode_adm = mode_result[0][0] if isinstance(mode_result[0], np.ndarray) else mode_result.mode[0]

print(f"Mean: {mean_adm}")
print(f"Median: {median_adm}")
print(f"Mode: {mode_adm}")

print("\nBest Representation:")
print("The mean or median both represent the data well since there are no extreme outliers.")

print("\nImpact of 10% Capacity Increase:")
print("If capacity increases by 10% and actual admissions scale accordingly (scaled by 1.1):")
print(f"New Mean: {mean_adm * 1.1:.2f}")
print(f"New Median: {median_adm * 1.1:.2f}")
print(f"New Mode: {mode_adm * 1.1:.2f}")
print("All central tendency measures will increase by exactly 10%.")


# # Question 2

# In[3]:


# Data
recovery_times = np.array([5, 7, 6, 8, 9, 5, 6, 7, 8, 6])

# 1. Range, Variance, Standard Deviation
r_range = np.ptp(recovery_times)
# ddof=1 is used for sample variance/std dev
r_var = np.var(recovery_times, ddof=1) 
r_std = np.std(recovery_times, ddof=1)

print(f"Range: {r_range} days")
print(f"Sample Variance: {r_var:.4f}")
print(f"Sample Standard Deviation: {r_std:.4f}")

# 2. Interpretation
print("\nInterpretation:")
print(f"The standard deviation of {r_std:.2f} days indicates that most patient recovery times ")
print("deviate from the mean recovery time by roughly ±1.35 days, showing low variability.")

# 3. Impact of new patients (4 and 10 days)
new_recovery_times = np.append(recovery_times, [4, 10])
new_r_std = np.std(new_recovery_times, ddof=1)
print(f"\nNew Standard Deviation (with 4 and 10): {new_r_std:.4f}")
print("The standard deviation increased because 4 and 10 introduce more extreme data points.")


# # Question 3

# In[4]:


# Data
satisfaction = np.array([8, 9, 7, 8, 10, 7, 9, 6, 10, 8, 7, 9])

# 1. Skewness and Kurtosis (bias=False for sample adjustments)
skew = stats.skew(satisfaction, bias=False)
kurt = stats.kurtosis(satisfaction, bias=False) # Fisher's definition (normal = 0)

print(f"Skewness: {skew:.4f}")
print(f"Excess Kurtosis: {kurt:.4f}")

# 2. Distribution interpretation
print("\nDistribution Interpretation:")
if -0.5 <= skew <= 0.5:
    print("The data is approximately symmetric.")
else:
    print("The data deviates from symmetry.")
print("Given the small sample size, it roughly resembles a normal distribution, but kurtosis is slightly flat.")

# 3. Impact of new customer service initiative
print("\nExpected Skewness Change after Initiative:")
print("If scores shift higher, a ceiling effect occurs at the max score (10).")
print("This will create a longer tail on the left, leading to Negative Skewness (Left-skewed).")


# # Question 4

# In[5]:


# Data setup
data = {
    'Nurses': [10, 12, 15, 18, 20, 22],
    'Avg_Recovery_Time': [8, 7, 6, 5, 4, 3]
}
df = pd.DataFrame(data)

# 1. Compute correlation coefficient
corr_coef, _ = stats.pearsonr(df['Nurses'], df['Avg_Recovery_Time'])

print(f"Pearson Correlation Coefficient: {corr_coef:.4f}")
print("This indicates a perfect negative linear relationship.")

# 2. Trend Impact
print("\nImpact of adding 5 nurses per department:")
print("Based on the perfect linear trend (an increase of 2 nurses drops recovery by 1 day):")
print("An increase of 5 nurses will reduce the average recovery time by 2.5 days per department.")


# # Question 5

# In[6]:


# Data
wait_times = np.array([32, 29, 31, 34, 33, 27, 30, 28, 35, 26])
mu_claimed = 30
alpha = 0.05

print("Hypotheses:")
print("H0: mu = 30 (Average wait time is 30 minutes)")
print("H1: mu != 30 (Average wait time is not 30 minutes)\n")

# Perform 1-sample t-test
t_stat, p_val = stats.ttest_1samp(wait_times, mu_claimed)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

if p_val < alpha:
    print("Result: Reject the null hypothesis. The hospital's claim is invalid.")
else:
    print("Result: Fail to reject the null hypothesis. The hospital's claim is valid.")

# Operational Changes
print("\nOperational changes if wait times significantly exceed 30 mins:")
print("- Implement a digital patient triaging system.")
print("- Increase medical and administrative staffing during peak arrival hours.")
print("- Optimize workflow handoffs between registration and clinical care.")


# # Question 6

# In[7]:


# Contingency table configuration
# Columns: [Satisfied, Unsatisfied]
# Rows: High, Medium, Low
contingency_table = np.array([
    [90, 10],  # High
    [60, 40],  # Medium
    [30, 70]   # Low
])

chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-square Statistic: {chi2:.4f}")
print(f"p-value: {p_val}")
print(f"Degrees of Freedom: {dof}")

if p_val < 0.05:
    print("Result: Reject H0. Cleanliness and patient satisfaction are significantly DEPENDENT.")
else:
    print("Result: Fail to reject H0. They are independent.")

# Distribution expectation
print("\nExpected changes if cleanliness improves:")
print("The proportion of 'Satisfied Patients' will shift heavily toward the 'High' cleanliness category,")
print("resulting in an overall decrease in 'Unsatisfied Patients' hospital-wide.")


# # Question 7

# In[8]:


# Data
treatment_A = [5, 6, 7, 5, 6]
treatment_B = [8, 9, 7, 8, 10]
treatment_C = [4, 5, 6, 5, 4]

print("Hypotheses:")
print("H0: Mean recovery times are equal across all treatments (mu_A = mu_B = mu_C)")
print("H1: At least one treatment group has a different mean recovery time.\n")

# Perform ANOVA
f_stat, p_val = stats.f_oneway(treatment_A, treatment_B, treatment_C)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_val}")

if p_val < 0.05:
    print("Result: Reject H0. There is a significant difference in recovery times among treatments.")
else:
    print("Result: Fail to reject H0. No significant difference found.")

# Data to collect for Treatment D
print("\nData required before concluding Treatment D's effectiveness:")
print("- Recovery duration data from a controlled sample of patients utilizing Treatment D.")
print("- Baseline data matching patient demographics (age, health status) to ensure fair comparison.")
print("- Monitoring for confounding factors like side effects or secondary medications.")


# # Question 8

# In[9]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

# Data
admin_times = np.array([12, 15, 14, 16, 18, 13, 14, 17, 15, 19, 16, 14])

# Shapiro-Wilk Test
shapiro_stat, p_val = stats.shapiro(admin_times)
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat:.4f}")
print(f"p-value: {p_val:.4f}")

if p_val > 0.05:
    print("Result: The data follows a normal distribution (Fail to reject H0).")
else:
    print("Result: The data does not follow a normal distribution (Reject H0).")

# Importance statement
print("\nWhy this analysis matters in healthcare:")
print("Knowing if administration time is normally distributed helps managers predict resource allocation,")
print("set standard operating benchmarks, and reliably use parametric predictive modeling.")

# Impact of emergency cases
print("\nImpact of an increase in emergency cases:")
print("The distribution will likely develop a strong right-skew (positive skew) and a longer tail,")
print("as complex emergency registration requirements inflate processing times arbitrarily.")

# Code block to optionally generate diagnostic plots in your notebook
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].hist(admin_times, bins=5, edgecolor='black', alpha=0.7)
# ax[0].set_title('Histogram')
# sm.qqplot(admin_times, line='s', ax=ax[1])
# ax[1].set_title('Q-Q Plot')
# plt.show()


# # Question 9

# In[10]:


import math

# Parameters
lam = 5  # average rate (lambda) per hour
k = 3    # exactly 3 cases

print("Distribution Model: Poisson Distribution")
print("Justification: Used for modeling the number of independent events occurring within a fixed time interval.")

# Probability calculation
prob_3 = stats.poisson.pmf(k, lam)
print(f"\nProbability of exactly 3 cases arriving in the next hour: {prob_3:.4f} ({prob_3*100:.2f}%)")

# Impact of a city accident
print("\nImpact of a major city accident:")
print("The arrival rate (lambda) would spike sharply during that period. The entire probability mass function")
print("would shift to the right, heavily raising the probability of experiencing a larger number of arrivals.")


# # Question 10

# In[11]:


# Data Setup
surgeries = np.array([0, 1, 2, 3, 4, 5])
frequencies = np.array([5, 12, 18, 22, 15, 8])
print("Distribution Model: Empirical Discrete Probability Distribution.")
print("Justification: The data consists of distinct countable outcomes mapping to explicit, recorded historical frequencies.")

# Compute probabilities
total_days = np.sum(frequencies)
probabilities = frequencies / total_days

# Expected Value E(X) = sum(x * P(x))
expected_surgeries = np.sum(surgeries * probabilities)

print("\nProbability Breakdown per Surgery Count:")
for s, p in zip(surgeries, probabilities):
    print(f"Surgeries: {s} | Probability: {p:.4f}")

print(f"\nExpected number of surgeries performed per day: {expected_surgeries:.4f}")

# Impact of a new surgical team
print("\nImpact of hiring a new surgical team:")
print("The hospital's daily capacity will expand. This will shift the higher frequency weightings")
print("toward 4, 5, or more surgeries, increasing both the expected value and changing the shape of the distribution.")


# In[ ]:





# In[ ]:





# In[ ]:




