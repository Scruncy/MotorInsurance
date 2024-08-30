from tkinter import NORMAL
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.gam.generalized_additive_model import GLMResults
from statsmodels.genmod.families import Gaussian, Poisson, Binomial, Tweedie
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from scipy.stats import nbinom
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
import itertools
from scipy.stats import chi2_contingency, chi2, expon, gaussian_kde, kstest, gamma
import seaborn as sns
from scipy import stats
from scipy.integrate import simps
import pickle
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde
from scipy.stats import (
        norm, expon, gamma, beta, lognorm, weibull_min, 
        uniform, kstest, chi2, t, logistic, pareto, gaussian_kde
    )

def main():


    df = pd.read_csv('data.csv', delimiter=';', low_memory=False)  #add to use the property delimiter since the data are distinguished by ;

    #Some data cleaning since the information needs to be compared. Since the power is very different depending on the type of vehicle
    # datas are split into a class of 3 instead of being numerical

    def classify_values(df, type_column, column_name, max_values, ratio1=1/3, ratio2=2/3):
        """
        Classify values in a DataFrame column based on their ratio to the maximum value for each type.
    
        Parameters:
        - df: The DataFrame containing the data.
        - type_column: The name of the column that specifies the type.
        - column_name: The name of the column to classify.
        - max_values: A dictionary mapping each type to its maximum value.
        - ratio1: The lower ratio threshold for classification.
        - ratio2: The upper ratio threshold for classification.
    
        Returns:
        - A DataFrame with a new column 'Classification' added.
        """
    
        # Define the classification function
        def classify(row):
            type_value = row[type_column]
            max_value = max_values.get(type_value)
            if max_value is None:
                return None  # If type_value is not in max_values
            threshold_1 = max_value * ratio1
            threshold_2 = max_value * ratio2
            value = row[column_name]
            if value < threshold_1:
                return 'easy'
            elif threshold_1 <= value < threshold_2:
                return 'medium'
            else:
                return 'fast'
    
        # Apply the classification function
        df['Classification'] = df.apply(classify, axis=1)
    
        return df

    # Define maximum values for each type (found trough printing the data frame)
    max_values = {
        1: 181,
        2: 340,
        3: 580,
        4: 194
    }
    #the other idea is to simply divide the power into 3 categories: "slow", "medium" and "fast". It's possible to consider 
    #the maximum power a veichile of that type can get and then according to that split the result into 3 categories. since
    # I am not an expert, I would consider the maximum for each category

    print(df[df['Type_risk'] == 1]['Power'].max()) #181
    print(df[df['Type_risk'] == 2]['Power'].max()) #340
    print(df[df['Type_risk'] == 3]['Power'].max()) #580
    print(df[df['Type_risk'] == 4]['Power'].max()) #194

    df = classify_values(df, 'Type_risk', 'Power', max_values)

    df['Date_last_renewal'] = pd.to_datetime(df['Date_last_renewal'], format='%d/%m/%Y')
    df['Date_birth'] = pd.to_datetime(df['Date_birth'], format='%d/%m/%Y')
    # I have to change the format since otherwise it's not possible to use it properly

    print(df.iloc[0:10, 0:4])
    # the porfolio has data from different years, let's check from which year until which one
    print(df['Date_last_renewal'].min())
    print(df['Date_last_renewal'].max())
    # datas are from 2015 to 2018. Datas will be split in this way: all the contracts renewed in 2015 belongs to the first year,
    # in 2016 to the second year and so on
    df['Year'] = df['Date_last_renewal'].dt.year #added new column to represent the year

    print(df.columns)  # New columns added correctly

    '''
    ['ID', 'Date_start_contract', 'Date_last_renewal', 'Date_next_renewal',
           'Date_birth', 'Date_driving_licence', 'Distribution_channel',
           'Seniority', 'Policies_in_force', 'Max_policies', 'Max_products',
           'Lapse', 'Date_lapse', 'Payment', 'Premium', 'Cost_claims_year',
           'N_claims_year', 'N_claims_history', 'R_Claims_history', 'Type_risk',
           'Area', 'Second_driver', 'Year_matriculation', 'Power',
           'Cylinder_capacity', 'Value_vehicle', 'N_doors', 'Type_fuel', 'Length',
           'Weight', 'Classification', 'Year']
    '''
    print(df)
    # Split the DataFrame based on the year
    df_2015 = df[df['Year'] == 2015]
    df_2016 = df[df['Year'] == 2016]
    df_2017 = df[df['Year'] == 2017]
    df_2018 = df[df['Year'] == 2018]

    #here change df_2015 to get different graphs
    plt.figure(figsize=(10, 6))
    plt.hist(df_2016['Premium'], bins=100, edgecolor='black', density=True)  # Adjust number of bins as needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values')
    plt.grid(True)
    plt.show()

    # to check which distribution premium might follow. it looks like a Normal distribution which can only start from some specific values
    # a glm model will be fit based on this information and the fact that for this analysis, a personal decision has been made to consider
    # only ['Type_risk', 'Classification', 'Area', 'Year']

    selected_columns_premia = ['Type_risk', 'Classification', 'Area', 'Year', 'Premium']
    df_selected_premia = df[selected_columns_premia]

    # Convert categorical variables into dummy/indicator variables
    df_dummies_premia = pd.get_dummies(df_selected_premia, columns=['Type_risk', 'Area', 'Classification', 'Year'], drop_first=True)

    # Define the formula for the GLM
    formula = 'Premium ~ Area_1 + Year_2016 + Year_2017 + Year_2018 + Type_risk_2 + Type_risk_3 + Type_risk_4 + Classification_fast + Classification_medium'

    # Fit the GLM with Gaussian family using the formula API
    model_premia = smf.glm(formula=formula, data=df_dummies_premia, family=sm.families.Gaussian()).fit()

    # Print the model summary
    print(model_premia.summary())
    # Histogram of Residuals
    residuals = model_premia.resid_pearson
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.title('Histogram of Residuals')
    plt.show()
    # Q-Q Plot
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    #we can see that the residuals look that they follow a normal distribution except for the tail, since after the tail the fitting is incorrect
    # we always underestimate. so that's more secutirity for future simulation.This could mean that the tail is following another distribution, so a mixture should be considered or other variables should be brought in
    # For semplicity, the Normal assumption is considered valid with only these variables

    #To fit the data to the value you want, you simply have to the changes the values below in new_data
    # Define new data
    new_data = pd.DataFrame({
        'Area_1': [1],
        'Year_2016': [1],
        'Year_2017': [0],
        'Year_2018': [0],
        'Type_risk_2': [0],
        'Type_risk_3': [1],
        'Type_risk_4': [0],
        'Classification_fast': [0],
        'Classification_medium': [1]
    })

    # Predict the expected value (mean)
    mean = model_premia.predict(new_data)[0]

    # Extract variance (dispersion parameter)
    variance = model_premia.scale

    # Calculate standard deviation
    std_dev = np.sqrt(variance)

    def filter_and_plot(df, new_data):
        """
        Filters the DataFrame `df` based on the characteristics in `new_data` and plots the results.

        Parameters:
        - df: pandas DataFrame containing the data.
        - new_data: pandas DataFrame containing the characteristics to filter by.

        Returns:
        - None
        """
        # Ensure new_data has the same columns as df
        new_data = new_data.astype(int)
    
        # Generate a filter mask
        mask = np.all([df[col] == value for col, value in new_data.iloc[0].items()], axis=0)
    
        # Apply the mask to filter the DataFrame
        filtered_df = df[mask]
    
        if filtered_df.empty:
            raise ValueError("No rows match the specified characteristics in new_data.")
        return filtered_df['Premium']
    
    def simulate_values(normal_mean=0, normal_std=1, threshold=40, num_simulations=10000):
        """
        Simulates values from a normal distribution and then generates as many additional values as needed
        to exceed a given threshold.

        Parameters:
        - normal_mean: Mean of the normal distribution.
        - normal_std: Standard deviation of the normal distribution.
        - threshold: The threshold value that must be exceeded.
        - num_simulations: Number of initial values to simulate from the normal distribution.

        Returns:
        - simulated_values: List of arrays where each array contains simulated values exceeding the threshold.
        """
        # List to store the results
        simulated_values = []
    
        # Generate initial values from the normal distribution
        initial_values = np.random.normal(loc=normal_mean, scale=normal_std, size=num_simulations)
    
        initial_value = initial_values[initial_values > threshold]

        return initial_value

    def plot_real_and_simulated_histogram_kde(real_data, simulated_data, bins=30):
        """
        Plots a histogram of real data and KDE of simulated data on a single plot for comparison.

        Parameters:
        - real_data: Array-like, the real dataset for which to plot the histogram.
        - simulated_data: Array-like, the simulated dataset for which to plot the KDE.
        - bins: Number of bins to use in the histogram.

        Returns:
        - None
        """
        plt.figure(figsize=(12, 6))

        # Plot histogram of real data
        sns.histplot(real_data, bins=bins, color='blue', label='Real Data', kde=False, alpha=0.6, stat='density')

        # Plot KDE of simulated data
        sns.kdeplot(simulated_data, color='red', label='Simulated Data KDE', fill=True, bw_adjust=0.5)

        # Customize the plot
        plt.title('Histogram of Real Data and KDE of Simulated Data')
        plt.xlabel('Value')
        plt.ylabel('Density/Frequency')
        plt.legend()

        # Display the plot
        plt.show()
    
    # Save the model to a file so it can be used in the other simulation
    with open('premium.pkl', 'wb') as file:
        pickle.dump(model_premia, file)

    

    data1 = filter_and_plot(df_dummies_premia, new_data)
    data2 = simulate_values(mean, math.sqrt(variance))
    plot_real_and_simulated_histogram_kde(data1, data2, bins=30)


    ###########################################################################################################
    # Number of claims analysis

    def calculate_mean_and_variance(df, type_risk, Area, column_name):
        """
        Calculates the mean and variance of a specified column in a filtered DataFrame.

        Parameters:
        - df: pandas DataFrame containing the data.
        - type_risk: The specific Type_risk to filter on.
        - Area: The specific Area to filter on.
        - column_name: The name of the column to calculate the mean and variance.

        Returns:
        - A tuple containing the mean and variance of the specified column, or (NaN, NaN) if no data meets the criteria.
        """
        # Filter the DataFrame based on the provided conditions
        filtered_df = df[(df['Type_risk'] == type_risk) & 
                         (df['Area'] == Area)]
    
        # Check if the filtered DataFrame is not empty
        if not filtered_df.empty:
            mean_value = np.mean(filtered_df[column_name])
            variance_value = np.var(filtered_df[column_name])
            return mean_value, variance_value
        else:
            return np.nan, np.nan
    
    #in order to check if the datas are overdispersed or not (namely they follow a negative binomial or poisson distribution)

    # Example usage for different years
    meanVariance_2015 = calculate_mean_and_variance(df_2015, 3, 1, 'N_claims_year')
    meanVariance_2016 = calculate_mean_and_variance(df_2016, 3, 1, 'N_claims_year')
    meanVariance_2017 = calculate_mean_and_variance(df_2017, 3, 1, 'N_claims_year')
    meanVariance_2018 = calculate_mean_and_variance(df_2018, 3, 1, 'N_claims_year')

    print("Mean for 2015:", meanVariance_2015)
    print("Mean for 2016:", meanVariance_2016)
    print("Mean for 2017:", meanVariance_2017)
    print("Mean for 2018:", meanVariance_2018)

    #As we can see, they are overdispersed. So they should follow a negative binomial distribution. But first, let's compute 
    # the glm for a poisson, just to 100%. The canonical link, in this case, it's considered

    selected_columns_poisson = ['N_claims_year', 'Type_risk', 'Classification', 'Area', 'Year']
    df_selected_poisson = df[selected_columns_poisson]

    df_dummies_poisson = pd.get_dummies(df_selected_poisson, columns=['Type_risk', 'Area', 'Classification', 'Year'], drop_first=True)

    print(df_dummies_poisson.dtypes)


    # Define the formula for the GLM
    formula = 'N_claims_year ~ Area_1 + Year_2016 + Year_2017 + Year_2018 + Type_risk_2 + Type_risk_3 + Type_risk_4 + Classification_fast+Classification_medium'

    # Fit the GLM with Poisson family
    model_poisson = glm(formula=formula, data=df_dummies_poisson, family=Poisson()).fit()

    # Print the summary of the model
    print(model_poisson.summary())

    #Let's graphically see if it's feasible or not

    def plot_relative_frequencies_only(df, column_name):
        """
        Plots the relative frequencies of categories in the specified column of the DataFrame.

        Parameters:
        - df: pandas DataFrame containing the data.
        - column_name: The name of the column to analyze.

        Returns:
        - None
        """
        # Calculate relative frequencies
        relative_frequencies = df[column_name].value_counts(normalize=True)
        relative_frequencies = relative_frequencies.sort_index()

        # Plotting the relative frequencies
        relative_frequencies.plot(kind='bar', color='skyblue', edgecolor='black')

        # Customize the plot
        plt.title(f'Relative Frequencies of {column_name}')
        plt.xlabel('Number of claims')
        plt.ylabel('Relative Frequency')

        # Display the plot
        plt.show()
    
    plot_relative_frequencies_only(df_2015[(df_2015['Type_risk'] == 3) & 
                          (df_2015['Area'] == 1) & 
                          (df_2015['Year'] == 2015)], 'N_claims_year')

    # it's possible to change the values to display different data, but it doesn't look like a good poisson

    def plot_relative_frequencies(df, column_name, Lambda):
        """
        Plots the relative frequencies of categories in the specified column of the DataFrame,
        and overlays the Poisson distribution.

        Parameters:
        - df: pandas DataFrame containing the data.
        - column_name: The name of the column to analyze.

        Returns:
        - None
        """
        # Calculate relative frequencies
        relative_frequencies = df[column_name].value_counts(normalize=True)
        relative_frequencies = relative_frequencies.sort_index()

        # Plotting the relative frequencies
        ax = relative_frequencies.plot(kind='bar', color='skyblue', edgecolor='black')

        # Poisson distribution parameters
        mean_value = Lambda
        x_values = np.arange(relative_frequencies.index.min(), relative_frequencies.index.max() + 1)
        poisson_pmf = poisson.pmf(x_values, mean_value)

        # Overlaying the Poisson distribution
        ax.plot(x_values, poisson_pmf, 'r-', marker='o', label='Poisson Distribution')

        # Customize the plot
        plt.title(f'Relative Frequencies of {column_name} with Poisson Distribution Overlay')
        plt.xlabel(column_name)
        plt.ylabel('Relative Frequency / Poisson PMF')
        plt.legend()

        # Display the plot
        plt.show()
    
    example_row = {
                'conts' : 1, 
                'Area_1': 0, 
                'Type_risk_2': 1, 
                'Type_risk_3': 0,
                'Type_risk_4': 0,
                'Classification_fast': 1,
                'Classification_medium': 0,
                'Year_2016': 0, 
                'Year_2017': 0,
                'Year_2018': 0,
            }
        
    spec = pd.DataFrame([example_row])    
    # Compute mu using the fitted model
    mu = model_poisson.predict(spec)

    plot_relative_frequencies(df_2015[(df_2015['Type_risk'] == 2) & 
                          (df_2015['Area'] == 0) & 
                          (df_2015['Year'] == 2015)], 'N_claims_year', mu)

    # if change correctly the values between example_row and the data_frame given in plot_relative_frequencies, it's possible to see
    # that the big problem is due to the fact that the probability of claims is always wrongly estimated. As we can see, thanks to plot_relative_frequencies_comparison
    # we can see in a single window this big problem 

    def plot_relative_frequencies_comparison(dfs, column_name, labels):
        """
        Plots the relative frequencies of categories for multiple DataFrames in a single window for comparison.

        Parameters:
        - dfs: List of pandas DataFrames containing the data.
        - column_name: The name of the column to analyze.
        - labels: List of labels for each DataFrame to use in the title.

        Returns:
        - None
        """
        # Number of DataFrames
        num_dfs = len(dfs)
    
        # Determine the number of rows and columns for the subplot grid
        ncols = 2  # Number of columns in the subplot grid
        nrows = (num_dfs + 1) // ncols  # Calculate the number of rows needed

        # Create a subplot grid
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), squeeze=False)
    
        # Flatten axes array for easier iteration
        axes = axes.flatten()

        for i, df in enumerate(dfs):
            # Calculate relative frequencies
            relative_frequencies = df[column_name].value_counts(normalize=True)
            relative_frequencies = relative_frequencies.sort_index()
        
            # Plotting the relative frequencies
            axes[i].bar(relative_frequencies.index, relative_frequencies.values, color='skyblue', edgecolor='black')
        
            # Customize the plot
            axes[i].set_title(f'Relative Frequencies of {column_name} ({labels[i]})')
            axes[i].set_xlabel('Number of claims')
            axes[i].set_ylabel('Relative Frequency')

        # Hide any unused subplots
        for j in range(num_dfs, len(axes)):
            axes[j].axis('off')
    
        # Adjust layout
        plt.tight_layout()
    
        # Display the plot
        plt.show()
    
    plot_relative_frequencies_comparison(               # graphical comparison of the 4 graphs
        dfs=[df_2015, df_2016, df_2017, df_2018],
        column_name='N_claims_year',
        labels=['2015', '2016', '2017', '2018']
    )

    # Once the poisson distribution has been excluded, let's considerate the negative binomial. Problem of the negative binomial:
    # one of the two parameters need to be determined a priori (in this case alpha), so first we need to compute and find the best one
    # to use in the glm analysis. We start with some ideas: we know that the value are pretty focused on 0, so we know that alpha needs to be very
    # small, between 0.1 and 0.3. the closer to 0, the higher the probability of being zero. It's been chosen to use the method of moments

    selected_columns_nbinom = ['N_claims_year', 'Type_risk', 'Classification', 'Area', 'Year']
    df_selected_nbinom = df[selected_columns_nbinom]

    df_dummies_nbinom = pd.get_dummies(df_selected_nbinom, columns=['Type_risk', 'Area', 'Classification', 'Year'], drop_first=True)

    r_definitive = np.mean(df_selected_nbinom['N_claims_year'])**2 / (np.var(df_selected_nbinom['N_claims_year']) - np.mean((df_selected_nbinom['N_claims_year'])))
    print(r_definitive)
    # Prepare your data
    for col in df_dummies_nbinom.columns:
        if df_dummies_nbinom[col].dtype == 'object':
            df_dummies_nbinom[col] = pd.to_numeric(df_dummies_nbinom[col], errors='coerce')

    # Check again to confirm the conversion
    print(df_dummies_nbinom.dtypes)

    for col in ['Area_1', 'Type_risk_2', 'Type_risk_3', 'Type_risk_4',  'Classification_fast','Classification_medium', 'Year_2016' , 'Year_2017',  'Year_2018']:
        df_dummies_nbinom[col] = df_dummies_nbinom[col].astype(int)

    # Define the Negative Binomial model
    negbinom_model = sm.GLM(
        df_dummies_nbinom['N_claims_year'], 
        sm.add_constant(df_dummies_nbinom[['Area_1', 'Type_risk_2', 'Type_risk_3', 'Type_risk_4',  'Classification_fast','Classification_medium', 'Year_2016' , 'Year_2017',  'Year_2018']]), 
        family=family.NegativeBinomial(alpha=r_definitive)
    )

    # Fit the model
    negbinom_result = negbinom_model.fit()

    print(negbinom_result.summary())
    print(negbinom_result.model.family.alpha)

    def plot_relative_frequencies_nbinom(df, column_name, r, p):
        """
        Plots the relative frequencies of categories in the specified column of the DataFrame,
        and overlays the Negative Binomial distribution.

        Parameters:
        - df: pandas DataFrame containing the data.
        - column_name: The name of the column to analyze.
        - r: The number of successes (dispersion parameter) for the Negative Binomial distribution.
        - p: The probability of success in each trial for the Negative Binomial distribution.

        Returns:
        - None
        """
        # Calculate relative frequencies
        relative_frequencies = df[column_name].value_counts(normalize=True)
        relative_frequencies = relative_frequencies.sort_index()

        # Plotting the relative frequencies
        ax = relative_frequencies.plot(kind='bar', color='skyblue', edgecolor='black')

        # Negative Binomial distribution parameters
        x_values = np.arange(relative_frequencies.index.min(), relative_frequencies.index.max() + 1)
        nbinom_pmf = nbinom.pmf(x_values, r, p)

        # Overlaying the Negative Binomial distribution
        ax.plot(x_values, nbinom_pmf, 'r-', marker='o', label='Negative Binomial Distribution')

        # Customize the plot
        plt.title(f'Relative Frequencies of {column_name} with Negative Binomial Distribution Overlay')
        plt.xlabel(column_name)
        plt.ylabel('Relative Frequency / Negative Binomial PMF')
        plt.legend()

        # Display the plot
        plt.show()
    
    #To graphically check the graphs



        
    # Save the GLM model with pickle
    with open('negbinom_model.pkl', 'wb') as f:
        pickle.dump(negbinom_result, f)

    # Save additional information to a text file
    with open('negbinom_model_info.txt', 'w') as f:
        # Save model coefficients
        f.write('Coefficients:\n')
        coefficients = negbinom_result.params
        coefficients.to_csv(f, header=True)
    
        # Save model summary (optional)
        f.write('\nModel Summary:\n')
        f.write(negbinom_result.summary().as_text())
    
        # Save other relevant information
        f.write('\nScale: {}\n'.format(negbinom_result.scale))
        f.write('Deviance: {}\n'.format(negbinom_result.deviance))
        f.write('AIC: {}\n'.format(negbinom_result.aic))
        f.write('Alpha: {}\n'.format(negbinom_result.model.family.alpha))
        f.write('Link function: {}\n'.format(negbinom_result.model.family.link.__class__.__name__))
    
    '''
    =========================================================================================
                                coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    const                    -3.2023      0.033    -97.571      0.000      -3.267      -3.138
    Area_1                    0.1453      0.010     14.450      0.000       0.126       0.165
    Type_risk_2               1.2172      0.030     40.376      0.000       1.158       1.276
    Type_risk_3               0.9367      0.029     32.735      0.000       0.881       0.993
    Type_risk_4              -4.9109      1.000     -4.909      0.000      -6.872      -2.950
    Classification_fast      -0.5481      0.238     -2.299      0.022      -1.015      -0.081
    Classification_medium     0.2059      0.022      9.513      0.000       0.163       0.248
    Year_2016                -0.0384      0.018     -2.161      0.031      -0.073      -0.004
    Year_2017                -0.5711      0.019    -30.599      0.000      -0.608      -0.534
    Year_2018                -1.4639      0.021    -68.099      0.000      -1.506      -1.422
    =========================================================================================
    '''

    def transform_array(arr):
        # Check if the length of the array is more than 4
        if len(arr) > 7:
            # Create a new array of length 7
            new_arr = arr[:7]  # Copy the first 4 elements
        
            # Calculate the sum of the values beyond the 4th element
            out_sum = sum(arr[7:])  # Sum elements from index 4 onward
        
            # Add the sum to the last element of the new array
            new_arr[-1] += out_sum
        
            return new_arr
        else:
            # If the array length is 7 or less, return it unchanged
            return arr
    
    # it's used for the chi-squared analysis


    def chi_squared_test_negative_binomial(df, column_name, r, p):
        """
        Perform a Chi-Squared test to validate the model against a Negative Binomial distribution.

        Parameters:
        - df: pandas DataFrame containing the data.
        - column_name: The name of the column to analyze.
        - r: The number of successes (dispersion parameter) for the Negative Binomial distribution.
        - p: The probability of success in each trial for the Negative Binomial distribution.
        - min_count_threshold: Minimum count to avoid small sample issues in Chi-Squared test.
        """
    
        # Calculate observed frequencies

        # Calculate observed frequencies
        observed_frequencies = df[column_name].value_counts().sort_index()
        tot = observed_frequencies.sum()

        # Determine the maximum value
        maximum = observed_frequencies.index.max()

        # Create a range from 0 to maximum
        possible_values = range(0, maximum + 1)

        # Reindex to include all possible values, filling missing values with 0
        observed_frequencies = observed_frequencies.reindex(possible_values, fill_value=0)

        # Convert to a list
        observed_frequencies_list = list(observed_frequencies)

        # Output the result
        print(observed_frequencies_list)

        

        # Define x-values for the Negative Binomial distribution
        x_values = np.arange(observed_frequencies.index.min(), maximum + 1)
        observed_frequencies = transform_array(observed_frequencies_list)
        print(observed_frequencies)
    
        # Compute expected frequencies using the Negative Binomial PMF
        expected_nb = nbinom.pmf(x_values, r, p)
        expected_nb_count = expected_nb * (tot)
    
        accumulated_sum = 0
    
        for index in range(len(expected_nb_count)):
            accumulated_sum += expected_nb_count[index]
            if accumulated_sum > (tot):
                expected_nb_count[index] -= (accumulated_sum - tot)
                break
        expected_nb_count_list = list(expected_nb_count)
        expected_nb_count = transform_array(expected_nb_count)
    
        print(expected_nb_count)
    
        # Compute the Chi-Squared statistic
        chi2_stat = 0
        for obs, exp in zip(observed_frequencies, expected_nb_count):
            if obs < 0.05 or exp < 0.05:
                break
            chi2_stat += (obs - exp) ** 2 / exp

    
        df = len(observed_frequencies) - 1
        if df ==0:
            df =1
        p_value = 1 - chi2.cdf(chi2_stat, df)
  
        print(f"Chi-Squared Statistic: {chi2_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
    
        return p_value

    # Define the different categories for each variable
    type_risk_values = df_2015['Type_risk'].unique()
    area_values = df_2015['Area'].unique()
    classification_values = df_2015['Classification'].unique()

    #it'important so later it's possible to iterate trough each value
    counter1 = 0
    counter = 0

    for type_risk, area, classification in itertools.product(type_risk_values, area_values, classification_values):
    
    
    
        # Filter the DataFrame for each combination
        subset_df = df_2017[(df_2017['Type_risk'] == type_risk) & 
                            (df_2017['Area'] == area) & 
                            (df_2017['Classification'] == classification)]
        # watch out, if you change the value of a year here, you have to do it as well below (where it's written here)

        counter1 += 1
    
        # Check if the subset is not empty
        if not subset_df.empty:
            print(f"Plotting for Type_risk={type_risk}, Area={area}, Classification={classification}")
        
    # Prepare the row for computing mu
            example_row = {
                'conts' : 1, 
                'Area_1': 1 if area == 1 else 0, 
                'Type_risk_2': 1 if type_risk == 2 else 0, 
                'Type_risk_3': 1 if type_risk == 3 else 0, 
                'Type_risk_4': 1 if type_risk == 4 else 0,
                'Classification_fast': 1 if classification == 'fast' else 0,
                'Classification_medium': 1 if classification == 'medium' else 0,
                'Year_2016': 0, 
                'Year_2017': 1,  #here
                'Year_2018': 0,
            }
        
            spec = pd.DataFrame([example_row])
            print(spec)
            eta = np.dot(spec, negbinom_result.params)
            cov_matrix = negbinom_result.cov_params()
            # Calculate the standard error of the prediction
            var_eta = np.dot(np.dot(spec, cov_matrix), spec.T).diagonal()
            se_eta = np.sqrt(var_eta)
            # Compute the critical value for the specified confidence level
            z_score = norm.ppf((1 + 0.95) / 2)
    
            # Compute the confidence intervals for the linear predictor (eta)
            eta_lower = eta - z_score * se_eta
            eta_upper = eta + z_score * se_eta
        
            print(eta)
    
            # Transform back to the original scale (mu)
            mu = np.exp((eta))
            mu_lower = np.exp((eta_lower))
            mu_upper = np.exp((eta_upper))
            print(mu_lower)
            print(mu_upper)
            # Compute mu using the fitted model
            mu = negbinom_result.predict(spec)
            print("Predicted mean (mu) for the specific values:", mu[0])
        
    
            # Compute p based on mu and the dispersion parameter alpha
            alpha = r_definitive # (from the model fitting)
            p = alpha / (alpha + mu)
        
        
            # Plot the relative frequencies
            plot_relative_frequencies_nbinom(subset_df, 'N_claims_year', alpha, p)
        
            # Define a range of mu values within the confidence interval
            mu_values = np.linspace(mu_lower, mu_upper, 100)
            alpha_values = np. linspace(0.01, 1)
            max_p_value = 0
       
        
            if chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p) < 0.05:
                for alpha in alpha_values:
                    for mu in mu_values:
                        # Compute p based on mu and the dispersion parameter alpha
                        p = alpha / (alpha + mu)
        
                        # Run the chi-squared test for this specific value of mu
                        p_value = chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p)
        
                        # Check if this p-value is the highest found so far
                        if p_value > max_p_value:
                            max_p_value = p_value
                        if max_p_value > 0.05:
                            break
                    if max_p_value > 0.05:                    
                        counter += 1
                        break
                
            else:
                counter += 1
                
    print(counter / counter1)    #How many times, for each year, the p-value is greater than 0.05 (when it's lower, it's mostly because the size
                                 # is huge, but graphically they fit properly)
                    
    #so far so good, I have that N is following NBinom where alpha is constant and p varies over the time. The grapich response
    # is pretty positive, since we have to think that these are real datas, so ofc it's normal that sometimes they are not 
    # perfectly fitted.

    ##########################################################################################à
    # Now let's analyze the Y distribution

    selected_columns = ['N_claims_year', 'Type_risk', 'Classification', 'Area', 'Year', 'Cost_claims_year']
    df_selected = df[selected_columns]
    df_selected = df_selected[df_selected['N_claims_year'] == 1]

    df_dummies = pd.get_dummies(df_selected, columns=['Type_risk', 'Area', 'Classification', 'Year'], drop_first=True)

    # Define the formula for the GLM
    formula = 'Cost_claims_year ~ Area_1 + Year_2016 + Year_2017 + Year_2018 + Type_risk_2 + Type_risk_3 + Classification_fast+Classification_medium'

    # Fit the GLM with Poisson family
    model = glm(formula=formula, data=df_dummies, family=Tweedie(var_power=1)).fit()

    # Print the summary of the model
    print(model.summary())

    # So we can see that Y stays the same and it doesn't really matter any other variable. Since this result, an analysis will
    # be made on the aggregate and then considered equal for everything. 

    def mixture_fitting(df, number_claims, thresholds):

        # Create a range of values for plotting
        if number_claims < 3:
            data = df[(df['N_claims_year'] == number_claims)]['Cost_claims_year']
        elif number_claims >=3:
            data = df[(df['N_claims_year'] >= number_claims)]['Cost_claims_year']
        x = np.linspace(min(data), max(data), 1000)

        # Plotting the histogram as an empirical PDF
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=500, density=True, alpha=0.6, color='k', edgecolor='k', label='Empirical PDF (Histogram)')

        plt.title('Empirical PDF of the Data (Histogram)')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()


        # Split data into regions
        data1 = data[data < thresholds[0]]
        print(len(data1))
        data2 = data[(data >= thresholds[0]) & (data <= thresholds[1])]
        print(len(data2))
        data3 = data[data > thresholds[1]] 
        # Define the threshold
        # Filter values greater than 850 and subtract 850

        sns.kdeplot(data, bw_adjust=0.5)

        def fit_and_plot_distributions(data):
            # List of distributions to fit and their corresponding names for kstest
            distributions = {
                'Normal': (norm, 'norm'),
                'Exponential': (expon, 'expon'),
                'Gamma': (gamma, 'gamma'),
                'Beta': (beta, 'beta'),
                'Log-Normal': (lognorm, 'lognorm'),
                'Weibull': (weibull_min, 'weibull_min'),
                'Uniform': (uniform, 'uniform'),
                'Chi-Square': (chi2, 'chi2'),
                'Logistic': (logistic, 'logistic'),
                'Pareto': (pareto, 'pareto')
            }

            # Create a range of values for plotting the PDFs
            x = np.linspace(min(data), max(data), 100)

            # Kernel Density Estimate (KDE) for the empirical PDF
            kde = gaussian_kde(data)
            pdf_empirical = kde(x)

            # Plot each fitted distribution on a separate graph
            for name, (dist, _) in distributions.items():
                plt.figure(figsize=(10, 6))
        
                try:
                    # Fit the distribution to the data
                    params = dist.fit(data)  #mle
            
                    # Compute the PDF for the fitted distribution
                    pdf_fitted = dist.pdf(x, *params)
            
                    # Plot the empirical PDF and the fitted PDF
                    plt.plot(x, pdf_empirical, 'k--', label='Empirical PDF (KDE)')
                    plt.plot(x, pdf_fitted, label=f'Fitted {name}')
                    plt.title(f'Fitted {name} Distribution')
                    plt.xlabel('Data')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
            
                except Exception as e:
                    print(f"Could not fit {name}: {e}")
            
        fit_and_plot_distributions(data1)
        fit_and_plot_distributions(data2)
        fit_and_plot_distributions(data3)


    # mixture_fitting(df, 2, [700,1600])

    ###################################################
    # Mixture chosen: exponential, gamma, log-normal with [250,850] when Y=1
    # When Y=2 [700,1600] gamma log-normal, logistic
    # When Y>=3 [800,2000] exponential, gamma, logistic

    def generate_and_plot_synthetic_data(data, thresholds, rv1, rv2, rv3, change1 = 1, change2 = 1, size=10000):
        """
        Fits the given data to specified distributions, generates synthetic data based on these fits,
        and plots histograms of both empirical and synthetic data.

        Parameters:
        - data: The original data to fit.
        - thresholds: A list containing the thresholds to split the data.
        - rv1, rv2, rv3: Random variables from scipy.stats (e.g., expon, gamma, lognorm).
        - size: The size of the synthetic data to generate (default is 10000).
        """

        # Split data into regions
        data1 = data[data < thresholds[0]]
        data2 = data[(data >= thresholds[0]) & (data <= thresholds[1])]
        data3 = data[data > thresholds[1]]

        # Estimate the distribution parameters for each segment
        params1 = rv1.fit(data1)
        params2 = rv2.fit(data2)
        params3 = rv3.fit(data3)

        # Calculate proportions for each segment
        total_length = len(data)
        prop1 = len(data1) / total_length 
        prop2 = len(data2) / total_length
        prop3 = len(data3) / total_length

        # Adjust proportions
        prop1 *= change1
        prop2 *= change2
        prop3 /= (prop1 + prop2 + prop3)

        # Function to generate synthetic data
        def generate_synthetic_data(size):
            synthetic_data = []
            for _ in range(size):
                u = np.random.uniform()
                if u < prop1:
                    # Generate from rv1 and ensure it is within the range
                    value = rv1.rvs(*params1)
                    while value < 40 or value >= thresholds[0]:
                        value = rv1.rvs(*params1)
                    synthetic_data.append(value)
                elif u < prop2 + prop1:
                    # Generate from rv2 and ensure it is within the range
                    value = rv2.rvs(*params2)
                    while value < thresholds[0] or value > thresholds[1]:
                        value = rv2.rvs(*params2)
                    synthetic_data.append(value)
                else:
                    # Generate from rv3 and ensure it is within the range
                    value = rv3.rvs(*params3)
                    while value <= thresholds[1]:
                        value = rv3.rvs(*params3)
                    synthetic_data.append(value)
        
            return np.array(synthetic_data)

        # Generate synthetic data
        synthetic_data = generate_synthetic_data(size)

        # Print statistics
        print("Estimated Parameters:")
        print(f"Params1 ({rv1.name}): {params1}")
        print(f"Params2 ({rv2.name}): {params2}")
        print(f"Params3 ({rv3.name}): {params3}")
        print(prop1)
        print(prop2)
        print(prop3)
    
        print("\nSynthetic Data Mean:", np.mean(synthetic_data))
        print("Original Data Mean:", np.mean(data))
        print("Synthetic Data Variance:", np.var(synthetic_data))
        print("Original Data Variance:", np.var(data))

        # Plot histograms
        plt.figure(figsize=(12, 6))

        # Empirical data histogram
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=515, density=True, alpha=0.6, color='k', edgecolor='k', label='Empirical Data')
        plt.title('Histogram of Empirical Data')
        plt.xlabel('Data')
        plt.xlim(0, 30000)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Synthetic data histogram
        plt.subplot(1, 2, 2)
        plt.hist(synthetic_data, bins=515, density=True, alpha=0.6, color='r', edgecolor='r', label='Synthetic Data')
        plt.title('Histogram of Synthetic Data')
        plt.xlim(0, 30000)
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        plt.show()

    # Example usage:
    data = df[df['N_claims_year'] >= 3]['Cost_claims_year']
    thresholds = [800,2000]

    # Call the function with the data and the random variables to use for each segment
    generate_and_plot_synthetic_data(data, thresholds, expon, gamma, logistic, 0.98, 1.04)


