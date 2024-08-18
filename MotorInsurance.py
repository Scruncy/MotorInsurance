

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian, Poisson, Binomial
from datetime import datetime


df = pd.read_csv('data.csv', delimiter=';')  #add to use the property delimiter since the data are distinguished by ;
print(df.columns)

"""
 Index(['ID', 'Date_start_contract', 'Date_last_renewal', 'Date_next_renewal',
       'Date_birth', 'Date_driving_licence', 'Distribution_channel',
       'Seniority', 'Policies_in_force', 'Max_policies', 'Max_products',
       'Lapse', 'Date_lapse', 'Payment', 'Premium', 'Cost_claims_year',
       'N_claims_year', 'N_claims_history', 'R_Claims_history', 'Type_risk',
       'Area', 'Second_driver', 'Year_matriculation', 'Power',
       'Cylinder_capacity', 'Value_vehicle', 'N_doors', 'Type_fuel', 'Length',
       'Weight'],
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


plt.figure(figsize=(10, 6))
plt.hist(df_2015[(df_2015['Type_risk'] == 3) & (df_2015['Value_vehicle'] < 50000)]['Value_vehicle'], bins=30, edgecolor='black', density=True)  # Adjust number of bins as needed
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

def plot_relative_frequencies(df, column_name):
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
                      (df_2015['Value_vehicle'] > 10000) & 
                      (df_2015['Date_birth'] < specific_date)], 'N_claims_year')  # Use == for comparison, 'N_claims_year')
plot_relative_frequencies(df_2016[(df_2016['Type_risk'] == 3) & 
                      (df_2016['Value_vehicle'] > 10000) & 
                      (df_2016['Date_birth'] < specific_date)], 'N_claims_year')  # Use == for comparison, 'N_claims_year')
plot_relative_frequencies(df_2017[(df_2017['Type_risk'] == 3) & 
                      (df_2017['Value_vehicle'] > 10000) & 
                      (df_2017['Date_birth'] < specific_date)], 'N_claims_year')  # Use == for comparison, 'N_claims_year')
plot_relative_frequencies(df_2018[(df_2018['Type_risk'] == 3) & 
                      (df_2018['Value_vehicle'] > 10000) & 
                      (df_2018['Date_birth'] < specific_date)], 'N_claims_year')  # Use == for comparison, 'N_claims_year')
def calculate_mean(df, type_risk, value_vehicle, year, column_name):
    filtered_df = df[(df['Type_risk'] == type_risk) & 
                     (df['Value_vehicle'] < value_vehicle) &
                     (df['Area'] == 0) &
                     (df['Year_matriculation'] < year)]
    if not filtered_df.empty:
        return np.mean(filtered_df[column_name])
    else:
        return np.nan  # Return NaN if no data meets the criteria

# Example usage for different years
mean_2015 = calculate_mean(df_2015, 3, 100000, 2016, 'N_claims_year')
mean_2016 = calculate_mean(df_2016, 3, 100000, 2016, 'N_claims_year')
mean_2017 = calculate_mean(df_2017, 3, 100000, 2016, 'N_claims_year')
mean_2018 = calculate_mean(df_2018, 3, 100000, 2016, 'N_claims_year')

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


# as graphical visualisation suggests, they look like a possible poisson/binomial distribution splitting values
# into 4 different years and considering all of them together
# to take in consideration also the other variables which might influence the distribution of the claims


selected_columns = ['N_claims_year', 'Type_risk', 'Classification', 'Area', 'Year']
df_selected = df[selected_columns]
df_selected['Year'] = df_selected['Year'] - 2014


# Assuming df is your DataFrame and var2 is the categorical variable
df_dummies = pd.get_dummies(df_selected, columns=['Classification', 'Type_risk'], drop_first=True)

print(df_dummies.dtypes)


# Define the formula for the GLM
formula = 'N_claims_year ~ Area + Year + Classification_fast + Classification_medium + Type_risk_2 + Type_risk_3 + Type_risk_4'

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








    