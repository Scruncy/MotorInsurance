

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

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
    plt.xlabel('Category')
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
        axes[i].set_xlabel('Category')
        axes[i].set_ylabel('Relative Frequency')

    # Hide any unused subplots
    for j in range(num_dfs, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    

plot_relative_frequencies(df, 'N_claims_year')
    
plot_relative_frequencies_comparison(               # graphical comparison of the 4 graphs
    dfs=[df_2015, df_2016, df_2017, df_2018],
    column_name='N_claims_year',
    labels=['2015', '2016', '2017', '2018']
)

# as graphical visualisation suggests, they look like a possible poisson/binomial distribution splitting values
# into 4 different years and considering all of them together

print(np.mean(df_2018['N_claims_year']))


    