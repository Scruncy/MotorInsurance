

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import statsmodels.api as sm
from statsmodels.formula.api import glm
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




df = pd.read_csv('data.csv', delimiter=';')  #add to use the property delimiter since the data are distinguished by ;


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



# Define maximum values for each type
max_values = {
    1: 181,
    2: 340,
    3: 580,
    4: 194
}

df = classify_values(df, 'Type_risk', 'Power', max_values)

print(df['Classification'])
print(df.columns)

"""
 Index(['ID', 'Date_start_contract', 'Date_last_renewal', 'Date_next_renewal',
       'Date_birth', 'Date_driving_licence', 'Distribution_channel',
       'Seniority', 'Policies_in_force', 'Max_policies', 'Max_products',
       'Lapse', 'Date_lapse', 'Payment', 'Premium', 'Cost_claims_year',
       'N_claims_year', 'N_claims_history', 'R_Claims_history', 'Type_risk',
       'Area', 'Second_driver', 'Year_matriculation', 'Power',
       'Cylinder_capacity', 'Value_vehicle', 'N_doors', 'Type_fuel', 'Length',
       'Weight', 'Classification'],
      dtype='object')

"""

print(df.iloc[0:10, 0:4])
#as we can see, the porfolio has data from different years, let's check from which year until which one

df['Date_last_renewal'] = pd.to_datetime(df['Date_last_renewal'], format='%d/%m/%Y')
df['Date_birth'] = pd.to_datetime(df['Date_birth'], format='%d/%m/%Y')
# I have to change the format since otherwise it's not possible to use it properly
print(df['Date_last_renewal'].min())
print(df['Date_last_renewal'].max())



#as we can see, there are 4 years of data in the portofolio, from 2015 to 2018.

#since the analysis should be made yearly, I split the portofolio into 4 different years and work singularly: the idea
# behind is that even if a contract starts at the end of decemeber 2015, it still belongs to the studio of the first year

df['Year'] = df['Date_last_renewal'].dt.year

print(df)

# Split the DataFrame based on the year
df_2015 = df[df['Year'] == 2015]
df_2016 = df[df['Year'] == 2016]
df_2017 = df[df['Year'] == 2017]
df_2018 = df[df['Year'] == 2018]

print(df_2015[ (df_2015['Classification'] == 'easy') & (df_2015['Cost_claims_year'] > 0)]['Cost_claims_year'])

# Calculate the mean
mean_cost = np.mean(df_2015[(df_2015['N_claims_year'] == 1) & (df_2015['Type_risk'] == 3)]['Cost_claims_year'])
print(mean_cost)

# Calculate the variance
variance_cost = np.var(df_2015[(df_2015['N_claims_year'] == 1) & (df_2015['Type_risk'] == 3)]['Cost_claims_year'])
print(variance_cost)

# Calculate the skewness
skewness_cost = df_2015[(df_2015['N_claims_year'] == 1) & (df_2015['Type_risk'] == 3)]['Cost_claims_year'].skew()
print(skewness_cost)

# Assuming df_2015 is your DataFrame and 'Cost_claims_year' is the column of interest
# sns.kdeplot(df_2015[(df_2015['N_claims_year'] == 3) & (df_2015['Type_risk'] == 3)]['Cost_claims_year'], bw_adjust=0.5)

plt.figure(figsize=(10, 6))
plt.hist(df_2015['Cost_claims_year'], bins=100, edgecolor='black', density=True)  # Adjust number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_2016[(df_2016['Type_risk'] == 3) & (df_2016['Value_vehicle'] < 50000)]['Value_vehicle'], bins=30, edgecolor='black', density=True)  # Adjust number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_2017[(df_2017['Type_risk'] == 3) & (df_2017['Value_vehicle'] < 50000)]['Value_vehicle'], bins=30, edgecolor='black', density=True)  # Adjust number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_2018[(df_2018['Type_risk'] == 3) & (df_2018['Value_vehicle'] < 50000)]['Value_vehicle'], bins=30, edgecolor='black', density=True)  # Adjust number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.grid(True)
plt.show()


print("DataFrame for 2015:")
print(df_2015)

print("\nDataFrame for 2016:")
print(df_2016)

print("\nDataFrame for 2017:")
print(df_2017)

print("\nDataFrame for 2018:")
print(df_2018)

'''def plot_relative_frequencies(df, column_name):
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
'''
'''
Intercept                        -0.3798
Type_risk_2[T.True]               1.3114
Type_risk_3[T.True]               0.9585
Type_risk_4[T.True]              -4.8585
Area                              0.1625
Year                             -0.5803
'''
    
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
    
specific_date = datetime(1995, 8, 18)

plot_relative_frequencies(df_2015[(df_2015['Type_risk'] == 3) & 
                      (df_2015['Area'] == 1) & 
                      (df_2015['Year'] == 2015)], 'N_claims_year', 0.5 )  # Use == for comparison, 'N_claims_year')
plot_relative_frequencies(df_2016[(df_2016['Type_risk'] == 3) & 
                      (df_2016['Area'] == 1) & 
                      (df_2016['Year'] == 2016)], 'N_claims_year', 0.64)  # Use == for comparison, 'N_claims_year')
plot_relative_frequencies(df_2017[(df_2017['Type_risk'] == 3) & 
                                  (df_2017['Area'] == 1)], 
                          'N_claims_year', 0.36)  # Use == for comparison, 'N_claims_year')
plot_relative_frequencies(df_2018[(df_2018['Type_risk'] == 3) & 
                                  (df_2018['Area'] == 1)], 
                          'N_claims_year', 0.19)  # Use == for comparison, 'N_claims_year')  # Use == for comparison, 'N_claims_year')
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
# Example usage for different years
mean_2015 = calculate_mean_and_variance(df_2015, 3, 1, 'N_claims_year')
mean_2016 = calculate_mean_and_variance(df_2016, 3, 1, 'N_claims_year')
mean_2017 = calculate_mean_and_variance(df_2017, 3, 1, 'N_claims_year')
mean_2018 = calculate_mean_and_variance(df_2018, 3, 1, 'N_claims_year')

print("Mean for 2015:", mean_2015)
print(np.mean(df_2015[df_2015['Type_risk'] == 3]['N_claims_year']))
print("Mean for 2016:", mean_2016)
print("Mean for 2017:", mean_2017)
print("Mean for 2018:", mean_2018)

    
plot_relative_frequencies_comparison(               # graphical comparison of the 4 graphs
    dfs=[df_2015, df_2016, df_2017, df_2018],
    column_name='N_claims_year',
    labels=['2015', '2016', '2017', '2018']
)



# consider the fucking canonical link function otherwise of course lambda doesn't make sense


# To decide which values will be included in the simulation:
# at the moment the idea is to consider of course the type of insurance (type risk), the year since there is a continuos
# development which could only be explicated by an increase of safety, "Area" and "power": 

# idea: to compare better (the problem would be that the horse power of bikes, for example, is way less than cars) between
# different vehicles, we consider the ratio horse power per weight (to consider the same type of veichle) (NVM)


df['PP'] = df['Power'] / df['Weight']

car = df[df['Type_risk'] == 3]

# Scatter plot between Variable1 and Variable2
plt.figure(figsize=(8, 6))
plt.scatter(car['PP'], car['Value_vehicle'], alpha=0.7)
plt.xlabel('Variable1')
plt.ylabel('Variable2')
plt.title('Scatter Plot of Variable1 vs Variable2')
plt.grid(True)
plt.show()

#the other idea is to simply divide the power into 3 categories: "slow", "medium" and "fast". It's possible to consider 
#the maximum power a veichile of that type can get and then according to that split the result into 3 categories. since
# I am not an expert, I would consider the maximum for each category

print(df[df['Type_risk'] == 1]['Power'].max()) #181
print(df[df['Type_risk'] == 2]['Power'].max()) #340
print(df[df['Type_risk'] == 3]['Power'].max()) #580
print(df[df['Type_risk'] == 4]['Power'].max()) #194




# as graphical visualisation suggests, they look like a possible poisson/binomial distribution splitting values
# into 4 different years and considering all of them together
# to take in consideration also the other variables which might influence the distribution of the claims


selected_columns = ['N_claims_year', 'Type_risk', 'Classification', 'Area', 'Year']
df_selected = df[selected_columns]
df_selected['Year'] = df_selected['Year']


# Assuming df is your DataFrame and var2 is the categorical variable
df_dummies = pd.get_dummies(df_selected, columns=['Type_risk', 'Area', 'Classification', 'Year'], drop_first=True)

print(df_dummies.dtypes)


# Define the formula for the GLM
formula = 'N_claims_year ~ Area_1 + Year_2016 + Year_2017 + Year_2018 + Type_risk_2 + Type_risk_3 + Type_risk_4 + Classification_fast+Classification_medium'

# Fit the GLM with Poisson family
model = glm(formula=formula, data=df_dummies, family=Poisson()).fit()

# Print the summary of the model
print(model.summary())


# Correctly apply conditions and compute the mean
mean_value = np.mean(df_selected[(df_selected['Classification'] == 'easy') & 
                                (df_selected['Area'] == 1) & 
                                (df_selected['Year'] == 4) &
                                (df_selected['Type_risk'] == 3)
                                ]['N_claims_year'])

print(mean_value)



import statsmodels.api as sm
import statsmodels.formula.api as smf

def neg_llf(alpha):
    try:
        model = sm.GLM(
            df_dummies['N_claims_year'], 
            sm.add_constant(df_dummies[['Area_1', 'Type_risk_2', 'Type_risk_3', 'Type_risk_4',  'Classification_fast','Classification_medium', 'Year_2016' , 'Year_2017',  'Year_2018']]), 
            family=family.NegativeBinomial(alpha, link=links.NegativeBinomial(alpha)))
        return -model.llf
    except:
        return np.inf
    

intercept = np.ones(len(df_dummies['N_claims_year']))

# Initial guess for alpha
alpha_init = 1.0

# Find the optimal alpha
result = minimize(neg_llf, alpha_init, bounds=[(0, 10)])

# Get the optimal alpha and dispersion
alpha_opt = result.x[0]

print(alpha_opt)

for col in df_dummies.columns:
    if df_dummies[col].dtype == 'object':
        df_dummies[col] = pd.to_numeric(df_dummies[col], errors='coerce')

# Check again to confirm the conversion
print(df_dummies.dtypes)

for col in ['Area_1', 'Type_risk_2', 'Type_risk_3', 'Type_risk_4',  'Classification_fast','Classification_medium', 'Year_2016' , 'Year_2017',  'Year_2018']:
    df_dummies[col] = df_dummies[col].astype(int)

# Define the Negative Binomial model
negbinom_model = sm.GLM(
    df_dummies['N_claims_year'], 
    sm.add_constant(df_dummies[['Area_1', 'Type_risk_2', 'Type_risk_3', 'Type_risk_4',  'Classification_fast','Classification_medium', 'Year_2016' , 'Year_2017',  'Year_2018']]), 
    family=family.NegativeBinomial(alpha=0.1760965105877899, link=links.NegativeBinomial(0.1760965105877899))
)

# Fit the model
negbinom_result = negbinom_model.fit()

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


# Print the summary of the model
print(negbinom_result.summary())

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

# Extract the dispersion parameter alpha from the fitted model
alpha_estimated = negbinom_result.model.family.alpha

# Calculate mean and variance of the fitted values
fitted_mean = negbinom_result.fittedvalues.mean()
print(fitted_mean)
fitted_variance = negbinom_result.fittedvalues.var()

# Estimating r using the formula: r = (mean^2) / (variance - mean)
r_estimated = (fitted_mean ** 2) / (fitted_variance - fitted_mean)

print(f"Estimated alpha (dispersion parameter): {alpha_estimated}")
print(f"Estimated r (size parameter): {r_estimated}")

def plot_relative_frequencies(df, column_name, r, p):
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



    
# Define the different categories for each variable
type_risk_values = df_2015['Type_risk'].unique()
area_values = df_2015['Area'].unique()
classification_values = df_2015['Classification'].unique()

# Calculate p for the Negative Binomial distribution

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
    

''' 
    
counter = 0

for type_risk, area, classification in itertools.product(type_risk_values, area_values, classification_values):
    
    
    
    # Filter the DataFrame for each combination
    subset_df = df_2015[(df_2015['Type_risk'] == type_risk) & 
                        (df_2015['Area'] == area) & 
                        (df_2015['Classification'] == classification)]
    
    # Check if the subset is not empty
    if not subset_df.empty:
        print(f"Plotting for Type_risk={type_risk}, Area={area}, Classification={classification}")
        counter = counter +1
        
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
            'Year_2017': 0,
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
        mu = links.NegativeBinomial(0.1760965105877899).inverse(eta)
        mu_lower = links.NegativeBinomial(0.1760965105877899).inverse(eta_lower)
        mu_upper = links.NegativeBinomial(0.1760965105877899).inverse(eta_upper)
        print(mu_lower)
        print(mu_upper)
        # Compute mu using the fitted model
        mu = negbinom_result.predict(spec)
        print("Predicted mean (mu) for the specific values:", mu[0])
        
    
        # Compute p based on mu and the dispersion parameter alpha
        alpha = 0.1760965105877899  # This is your dispersion parameter (from the model fitting)
        p = alpha / (alpha + mu)
        
        
        # Plot the relative frequencies
        plot_relative_frequencies(subset_df, 'N_claims_year', alpha, p)
        
    # Define a range of mu values within the confidence interval
        mu_values = np.linspace(mu_lower, mu_upper, 100)
        alpha_values = np. linspace(0.05, 2)
        max_p_value = 0
        
        if chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p) < 0.05:
            for alpha in alpha_values:
                for mu in mu_values:
                    # Compute p based on mu and the dispersion parameter alpha
                    p = alpha / (alpha + mu)
                    print(p)
        
                    # Run the chi-squared test for this specific value of mu
                    p_value = chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p)
        
                    # Check if this p-value is the highest found so far
                    if p_value > max_p_value:
                        max_p_value = p_value
                        best_mu = mu
                    if p_value > 0.05:
                        print("ok")
                        break
                if p_value > 0.05:
                    break
            
        
print(counter)

        

for type_risk, area, classification in itertools.product(type_risk_values, area_values, classification_values):
    
    # Filter the DataFrame for each combination
    subset_df = df_2016[(df_2016['Type_risk'] == type_risk) & 
                        (df_2016['Area'] == area) & 
                        (df_2016['Classification'] == classification)]

    
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
            'Year_2016': 1, 
            'Year_2017': 0,
            'Year_2018': 0,
        }
        
        spec = pd.DataFrame([example_row])    
        # Compute mu using the fitted model
        mu = negbinom_result.predict(spec)
        print("Predicted mean (mu) for the specific values:", mu[0])
    
        # Compute p based on mu and the dispersion parameter alpha
        alpha = 0.1760965105877899 # This is your dispersion parameter (from the model fitting)
        p = alpha / (alpha + mu)
        
        # Call the plotting function
        plot_relative_frequencies(subset_df, 'N_claims_year', alpha, p)
        # Define a range of mu values within the confidence interval
        mu_values = np.linspace(mu_lower, mu_upper, 100)
        alpha_values = np. linspace(0.05, 2)
        max_p_value = 0
        
        if chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p) < 0.05:
            for alpha in alpha_values:
                for mu in mu_values:
                    # Compute p based on mu and the dispersion parameter alpha
                    p = alpha / (alpha + mu)
                    print(p)
        
                    # Run the chi-squared test for this specific value of mu
                    p_value = chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p)
        
                    # Check if this p-value is the highest found so far
                    if p_value > max_p_value:
                        max_p_value = p_value
                        best_mu = mu
                    if p_value > 0.05:
                        print("ok")
                        break
                if p_value > 0.05:
                    break
        

        
for type_risk, area, classification in itertools.product(type_risk_values, area_values, classification_values):
    
    # Filter the DataFrame for each combination
    subset_df = df_2017[(df_2017['Type_risk'] == type_risk) & 
                        (df_2017['Area'] == area) & 
                        (df_2017['Classification'] == classification)]
    
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
            'Year_2017': 1,
            'Year_2018': 0,
        }
        
        spec = pd.DataFrame([example_row])    
        # Compute mu using the fitted model
        mu = negbinom_result.predict(spec)
        print("Predicted mean (mu) for the specific values:", mu[0])
    
        # Compute p based on mu and the dispersion parameter alpha
        alpha = 0.1760965105877899 # This is your dispersion parameter (from the model fitting)
        p = alpha / (alpha + mu)
        
        # Call the plotting function
        plot_relative_frequencies(subset_df, 'N_claims_year', alpha, p)
        chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p)
        
print("done")
        
for type_risk, area, classification in itertools.product(type_risk_values, area_values, classification_values):
    
    # Filter the DataFrame for each combination
    subset_df = df_2018[(df_2018['Type_risk'] == type_risk) & 
                        (df_2018['Area'] == area) & 
                        (df_2018['Classification'] == classification)]
    
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
            'Year_2017': 0,
            'Year_2018': 1,
        }
        
        spec = pd.DataFrame([example_row])    
        # Compute mu using the fitted model
        mu = negbinom_result.predict(spec)
        print("Predicted mean (mu) for the specific values:", mu[0])
    
        # Compute p based on mu and the dispersion parameter alpha
        alpha = 0.1760965105877899 # This is your dispersion parameter (from the model fitting)
        p = alpha / (alpha + mu)
        
        # Call the plotting function
        plot_relative_frequencies(subset_df, 'N_claims_year', alpha, p)
        chi_squared_test_negative_binomial(subset_df, 'N_claims_year', alpha, p)

#so far so good, I have that N is following NBinom where alpha is constant and p varies over the time. The grapich response
# is pretty positive, since we have to think that these are real datas, so ofc it's normal that sometimes they are not 
# perfectly fitted.


'''


selected_columns = ['N_claims_year', 'Type_risk', 'Classification', 'Area', 'Year', 'Cost_claims_year']
df_selected = df[selected_columns]
df_selected['Year'] = df_selected['Year']
df_selected = df_selected[df_selected['N_claims_year'] == 1]



# Assuming df is your DataFrame and var2 is the categorical variable
df_dummies = pd.get_dummies(df_selected, columns=['Type_risk', 'Area', 'Classification', 'Year'], drop_first=True)

print(df_dummies.dtypes)


# Define the formula for the GLM
formula = 'Cost_claims_year ~ Area_1 + Year_2016 + Year_2017 + Year_2018 + Type_risk_2 + Type_risk_3 + Classification_fast+Classification_medium'

# Fit the GLM with Poisson family
model = glm(formula=formula, data=df_dummies, family=Tweedie(var_power=1)).fit()

# Print the summary of the model
print(model.summary())

# So we can see that Y stays the same and it doesn't really matter any other variable. Since this result, an analysis will
# be made on the aggregate and then considered equal for everything. To consider the fact that the assumption is that
# in case of two claims, the distribution is simply the sum of the previous two.

# 300, 850 and so on

# Create a range of values for plotting
data = df[(df['N_claims_year'] == 1)]['Cost_claims_year']
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

# Kernel Density Estimate (KDE) for the empirical PDF
kde = gaussian_kde(data)
pdf_empirical = kde(x)
integral = simps(pdf_empirical, x)
print(pdf_empirical.sum())

# Plotting the empirical PDF
plt.figure(figsize=(8, 6))
plt.plot(x, pdf_empirical, 'k-', label='Empirical PDF (KDE)')
plt.title('Empirical PDF of the Data')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()


data = df[(df['N_claims_year'] == 1) & (df['Cost_claims_year'] > 800)]['Cost_claims_year'] 
# Define the threshold
# Filter values greater than 850 and subtract 850

sns.kdeplot(data, bw_adjust=0.5)

from scipy.stats import (
    norm, expon, gamma, beta, lognorm, weibull_min, 
    uniform, kstest, chi2, t, logistic, pareto, gaussian_kde
)

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
            params = dist.fit(data)
            
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
            

fit_and_plot_distributions(data)


# log-normal over 850. gamma under 250, weibull the other one


data = df[(df['N_claims_year'] == 1)]['Cost_claims_year']

# Define thresholds
thresholds = [250, 825]

np.random.seed(42)

# Split data into regions
data1 = data[data < thresholds[0]]
print(len(data1))
data2 = data[(data >= thresholds[0]) & (data <= thresholds[1])]
print(len(data2))
data3 = data[data > thresholds[1]]

# Fit distributions
params_gamma = gamma.fit(data1)
params_weibull = weibull_min.fit(data2)
params_lognorm = lognorm.fit(data3)

# Calculate proportions for each segment
total_length = len(data)
prop_gamma = len(data1) / total_length 
prop_weibull = len(data2) / total_length
prop_lognorm = len(data3) / total_length

prop_gamma = prop_gamma * 0.96
prop_weibull = prop_weibull * 0.97
prop_lognorm = prop_lognorm / (prop_gamma + prop_weibull + prop_lognorm)

# Generate synthetic data using the proportions
def generate_synthetic_data(size):
    synthetic_data = []
    for _ in range(size):
        u = np.random.uniform()
        if u < prop_gamma:
            # Generate from gamma and ensure it is within the range
            value = gamma.rvs(*params_gamma) if params_gamma is not None else np.nan
            while value < 40 or value >= thresholds[0]:
                value = gamma.rvs(*params_gamma) if params_gamma is not None else np.nan
            synthetic_data.append(value)
        elif u < prop_gamma + prop_weibull:
            # Generate from Weibull and ensure it is within the range
            value = weibull_min.rvs(*params_weibull) if params_weibull is not None else np.nan
            while value < thresholds[0] or value > thresholds[1]:
                value = weibull_min.rvs(*params_weibull) if params_weibull is not None else np.nan
            synthetic_data.append(value)
        else:
            # Generate from log-normal and ensure it is within the range
            value = lognorm.rvs(*params_lognorm) if params_lognorm is not None else np.nan
            while value <= thresholds[1]:
                value = lognorm.rvs(*params_lognorm) if params_lognorm is not None else np.nan
            synthetic_data.append(value)
    
    return np.array(synthetic_data)

# Generate synthetic data with the same length as the original data
synthetic_data = generate_synthetic_data(10000)

print(np.mean(synthetic_data))
print(np.mean(data))
print(np.var(synthetic_data))
print(np.var(data))

# Plotting the histograms of the empirical and synthetic data
plt.figure(figsize=(12, 6))

# Empirical data histogram
plt.subplot(1, 2, 1)
plt.hist(df[(df['N_claims_year'] == 1) ]['Cost_claims_year'], bins=515, density=True, alpha=0.6, color='k', edgecolor='k', label='Empirical Data')
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


data2 = df[(df['N_claims_year'] == 1) & (df['Area'] == 0) & (df['Cost_claims_year'] > 750) & (df['Cost_claims_year'] < 10000)]['Cost_claims_year']
# Fit the logistic distribution to the data
params_logistic = logistic.fit(data2)

print(params_logistic)
print(np.mean(data2))

# Perform the Kolmogorov-Smirnov test
d_statistic, p_value = kstest(data2, 'logistic', args=params_logistic)

# Print the results
print(f"KS Statistic: {d_statistic}")
print(f"P-value: {p_value}")

fit_and_plot_distributions(data2)

#exp until 750







    