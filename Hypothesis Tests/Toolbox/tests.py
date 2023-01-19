import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Two-Sample Permuation p-value test

class permut_two_sample_p_test:

  """
  Non-parametric - We are not assuming any underlying distribution.
  Because you have the two samples you will perform and permutation test , 
  and assess the hypothesis that for example: "the forces from Frog B and Frog C come from the same distribution."
  1 - Ho - Define the null hypothesis, example the distribution scores in rj are the same as sp
  2 - Define test statistics that will be performed, example: np.mean(rj_scores)-np.mean(sp_scores)
  3 - Many times of simulating data assuming the null hypothesis is true. For one-sample bootstraping and for two-sample permuting data
  4 - Compute test statistics for simulated data sets.
  5 - p value - % of times that observed test statistics are inside simulated test statistics array.   Less p-values higher statistical difference between
  arrays
  ### Args
  data_1, data_2 - datas to test
  func - test statistics, to create null hypothesis, example np.mean(rj_scores)-np.mean(sp_scores)
  size - 1000 by standard. How many times the simulated test statistics will be made. 
  
  """

  def __init__(self, data_1, data_2, func, size=1000, plot = True):
        
    self.data_1 = data_1
    self.data_2 = data_2
    self.func = func
    self.size = size
    self.plot = plot
    self.empirical_diff_means = self.func(self.data_1, self.data_2)

  #def ecdf(self):
  
  '''
  To include in future, functions to be used in func
  
  Function to A/B test
  
  def diff_frac(data_a,data_b):
  
    frac_a = np.sum(data_a) / len(data_a)
    frac_b = np.sum(data_b)/ len(data_b)
    
    return frac_b-frac_a
  
  '''
    

  def permutation_sample(self):
      """Generate a permutation sample from two data sets."""

      # Concatenate the data sets: data
      data = np.concatenate((self.data_1,self.data_2))

      # Permute the concatenated array: permuted_data
      permuted_data = np.random.permutation(data)

      # Split the permuted array into two: perm_sample_1, perm_sample_2
      perm_sample_1 = permuted_data[:len(self.data_1)]
      perm_sample_2 = permuted_data[len(self.data_1):]

      return perm_sample_1, perm_sample_2

  def draw_perm_reps(self):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(self.size)

    for i in range(self.size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = self.permutation_sample()

        # Compute the test statistic
        perm_replicates[i] = self.func(perm_sample_1, perm_sample_2)

    return perm_replicates

  def p_value(self):

    self.perm_replicates = self.draw_perm_reps()

    p_value = np.sum(self.perm_replicates >= self.empirical_diff_means) / len(self.perm_replicates)

    if self.plot == True:
      _ = sns.histplot(self.perm_replicates)
      _= plt.axvline(self.empirical_diff_means, color='red', linestyle='dashed', linewidth=1)
      plt.show()

    return print('p = ', p_value )


  
#Function Boostraping One-Sample P value

class bs_one_sample_p_test:

  """
  Non-parametric - We are not assuming any underlying distribution.
  It will be used bootstraping once that we just have one of samples, so here we can just test the summary statistics, for example mean, median and etc...
  At the end we could not say that they come from same distribution for example.
  Replicate it's a summary statistics for some bootstraped array.
  1 - Ho - Define the null hypothesis, example the mean scores in rj are the same as sp
  2 - Define test statistics that will be performed, example: np.mean(rj_scores)-np.mean(sp_scores)
  3 - Many times of simulating data assuming the null hypothesis is true. For one-sample bootstraping and for two-sample permuting data
  4 - Compute test statistics for simulated data sets.
  5 - p value - % of times that observed test statistics are inside simulated test statistics array.   Less p-values higher statistical difference between
  arrays
  ### Args
  data_1 - should be and array, observed array values of frog B 
  
  data_2 - should be an unique value, for example mean of frog C forces
  func - test statistics, to create null hypothesis
  size - how many times the simulated test statistics will be made.
  
  """

  def __init__(self,data_1,value_2,func=np.mean,size=10000,plot=True):

    self.data_1 = data_1
    self.value_2 = value_2
    self.func = func
    self.plot = plot
    self.size = size


  def translate_array(self):
      
    # Make an array of translated data_1 with same mean as value_2
    translated_data_1_array = self.data_1 - np.mean(self.data_1) + np.mean(self.value_2) # Fixing the drift in mean
    return translated_data_1_array

  def bootstrap_replicate_1d(self):

    """Generate bootstrap replicate of 1D data."""

    bs_sample = np.random.choice(self.translate_array(), len(self.translate_array()))

    return self.func(bs_sample)


  def draw_bs_reps(self):
    """Draw array of bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(self.size)

    # Generate replicates
    for i in range(self.size):
        bs_replicates[i] = self.bootstrap_replicate_1d()

    return bs_replicates
  

  def p_value(self):

    bs_replicates = self.draw_bs_reps()

    p_value = np.sum(bs_replicates <= self.func(self.data_1)) / self.size

    if self.plot == True:
      _ = sns.histplot(bs_replicates)
      _= plt.axvline(self.func(self.data_1), color='red', linestyle='dashed', linewidth=1)
      plt.show()

    return print('p = ', p_value)
  
    
    
    
#Function Boostraping One-Sample P value
  
class bs_one_sample_p_test:

  """
  Non-parametric - We are not assuming any underlying distribution.
  We will use bootstrap so here we can just test the summary statistics, for example mean, median and etc...
  At the end we could not say that they come from same distribution for example.
  Replicate it's a summary statistics for some bootstraped array.
  1 - Ho - Define the null hypothesis, example the mean scores in rj are the same as sp
  2 - Define test statistics that will be performed, example: np.mean(rj_scores)-np.mean(sp_scores)
  3 - Many times of simulating data assuming the null hypothesis is true. For one-sample bootstraping and for two-sample permuting data
  4 - Compute test statistics for simulated data sets.
  5 - p value - % of times that observed test statistics are inside simulated test statistics array.   Less p-values higher statistical difference between
  arrays
  ### Args
  data_1 - should be and array, observed array values of frog B 
  value - should be an unique value and a summary statistics, for example mean of frog C forces
  size - how many times the simulated test statistics will be made.
  
  """

  def __init__(self,data,value,func=np.mean,size=10000,plot=True):

    self.data = data
    self.value = value
    self.func = func
    self.size = size
    self.plot = plot
    self.overall_mean = self.func(np.append(data,value))


  def translate_array(self):
      
    # Shifiting arrays data_1 and data_2 to represent overall mean
    translated_data_array = self.data - np.mean(self.data) + self.value # Fixing the drift in mean
    
    return translated_data_array


  def bootstrap_replicate_1d(self):

    """Generate bootstrap replicate of 1D data."""

    bs_sample = np.random.choice(self.translate_array(), len(self.translate_array()))
    
    return self.func(bs_sample)


  def draw_bs_reps(self):
    """Draw array of bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(self.size)

    # Generate replicates
    for i in range(self.size):
        bs_replicates[i] = self.bootstrap_replicate_1d()

    return bs_replicates


  def p_value(self):

    bs_replicates = self.draw_bs_reps()

    p_value = np.sum(bs_replicates >= np.mean(self.data)) / self.size

    if self.plot == True:
      _ = sns.histplot(bs_replicates)
      _= plt.axvline(np.mean(self.data), color='red', linestyle='dashed', linewidth=1)
      plt.show()

    return print('p = ', p_value)
    
    
  
#Function Boostraping Two-Sample P value
  
class bs_two_sample_p_test:

  """
  Non-parametric - We are not assuming any underlying distribution.
  We will use bootstrap so here we can just test the summary statistics, for example mean, median and etc...
  At the end we could not say that they come from same distribution for example.
  Replicate it's a summary statistics for some bootstraped array.
  1 - Ho - Define the null hypothesis, example the mean scores in rj are the same as sp
  2 - Define test statistics that will be performed, example: np.mean(rj_scores)-np.mean(sp_scores)
  3 - Many times of simulating data assuming the null hypothesis is true. For one-sample bootstraping and for two-sample permuting data
  4 - Compute test statistics for simulated data sets.
  5 - p value - % of times that observed test statistics are inside simulated test statistics array.   Less p-values higher statistical difference between
  arrays
  ### Args
  data_1 - should be and array, observed array values of frog B 
  
  data_2 - should be an unique value, for example mean of frog C forces
  func - test statistics, to create null hypothesis
  size - how many times the simulated test statistics will be made.
  
  """

  def __init__(self,data_1,data_2,func=np.mean,size=10000,plot=True):

    self.data_1 = data_1
    self.data_2 = data_2
    self.func = func
    self.size = size
    self.plot = plot
    self.overall_mean = self.func(np.concatenate((data_1,data_2)))


  def translate_array(self):
      
    # Shifiting arrays data_1 and data_2 to represent overall mean
    translated_data_1_array = self.data_1 - np.mean(self.data_1) + self.overall_mean # Fixing the drift in mean
    translated_data_2_array = self.data_2 - np.mean(self.data_2) + self.overall_mean # Fixing the drift in mean

    return translated_data_1_array,translated_data_2_array

  def bootstrap_replicate_1d(self):

    """Generate bootstrap replicate of 1D data."""

    bs_sample_1 = np.random.choice(self.translate_array()[0], len(self.translate_array()[0]))
    bs_sample_2 = np.random.choice(self.translate_array()[1], len(self.translate_array()[1]))

    return self.func(bs_sample_1),self.func(bs_sample_2)


  def draw_bs_reps(self):
    """Draw array of bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates_1 = np.empty(self.size)
    bs_replicates_2 = np.empty(self.size)

    # Generate replicates
    for i in range(self.size):
        bs_replicates_1[i] = self.bootstrap_replicate_1d()[0]
        bs_replicates_2[i] = self.bootstrap_replicate_1d()[1]

    # Get replicates of difference of means: bs_replicates
    bs_replicates = bs_replicates_1 - bs_replicates_2

    return bs_replicates
  

  def p_value(self):

    bs_replicates = self.draw_bs_reps()

    mean_diffs = (np.mean(self.data_1)-np.mean(self.data_2))

    p_value = np.sum(bs_replicates >= mean_diffs) / self.size

    if self.plot == True:
      _ = sns.histplot(bs_replicates)
      _= plt.axvline(mean_diffs, color='red', linestyle='dashed', linewidth=1)
      plt.show()

    return print('p = ', p_value)

