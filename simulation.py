from calendar import c
from cgitb import text
from queue import Empty
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
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
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



def generate_sample(alpha, p, size):
    """Generate samples based on the Negative Binomial distribution."""
    # Generate samples from the Negative Binomial distribution
    negative_binomial_samples = np.random.negative_binomial(alpha, p, size)

    # List to hold the final samples
    final_samples = []

    # Generate samples based on the value from the Negative Binomial distribution
    for sample in negative_binomial_samples:
        if sample == 0:
            # Do nothing
            continue
        elif sample == 1:
            # Sample from Exponential(2)
            final_sample = expon.rvs(scale=2)
            final_samples.append(final_sample)
        elif sample == 2:
            # Sample from Uniform(0, 1) and split into 2 parts
            uniform_value = np.random.uniform(0, 1)
            final_samples.append(alpha * uniform_value)
            final_samples.append((1 - alpha) * uniform_value)
        else:
            # Sample from Uniform(0, 1) and split into 'sample' parts
            uniform_value = np.random.uniform(0, 1)
            split_values = [uniform_value / sample] * sample
            for i in range(sample):
                # Randomly distribute parts
                random_index = np.random.randint(sample)
                split_values[random_index] += uniform_value / sample
            final_samples.extend(split_values)
    
    return (final_samples)

def plot_histogram(data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plot a histogram of the given data.

    Parameters:
    - data: A list or numpy array of numerical data
    - bins: Number of bins for the histogram
    - title: Title of the histogram
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    
def mean_excess(data, threshold):
    """
    Calculate the mean excess function over a given threshold.
    
    Parameters:
    - data: A list or numpy array of numerical data
    - threshold: The threshold value
    
    Returns:
    - mean_excess_value: The mean excess over the given threshold
    """
    data = np.array(data)
    # Filter data points that are greater than the threshold
    excesses = data[data > threshold] - threshold
    
    # Compute the mean of the excesses
    if len(excesses) == 0:
        return np.nan  # Return NaN if there are no values greater than the threshold
    mean_excess_value = np.mean(excesses)
    return mean_excess_value

def plot_mean_excess(data, thresholds):
    """
    Plot the mean excess function for a range of thresholds.
    
    Parameters:
    - data: A list or numpy array of numerical data
    - thresholds: A list or numpy array of threshold values
    """
    mean_excess_values = [mean_excess(data, t) for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mean_excess_values, marker='o', linestyle='-', color='b')
    plt.title('Mean Excess Function')
    plt.xlabel('Threshold')
    plt.ylabel('Mean Excess')
    plt.grid(True)
    plt.show()
    
def first_page(negbinom_result):
    def save_integer():
        nonlocal saved_integer_value
        try:
            saved_integer_value = int(integer_value.get())
        except:
            saved_integer_value = 100 #if the input is wrong, it gets value 100 as default
            print("hello world")
        finally:
            print(f"Entered Integer Value: {saved_integer_value}")

    def save_and_update(section, value):
        saved_selection[section] = value
        update_dict()
        highlight_button(section, get_button(section, value))

    def update_dict():
        example_row['Area_1'] = 1 if saved_selection['area'] == "urban" else 0
        example_row['Type_risk_2'] = 1 if saved_selection['type_risk'] == 2 else 0
        example_row['Type_risk_3'] = 1 if saved_selection['type_risk'] == 3 else 0
        example_row['Type_risk_4'] = 1 if saved_selection['type_risk'] == 4 else 0
        example_row['Classification_fast'] = 1 if saved_selection['classification'] == 'fast' else 0
        example_row['Classification_medium'] = 1 if saved_selection['classification'] == 'medium' else 0
        example_row['Year_2016'] = 1 if saved_selection['year'] == 2016 else 0
        example_row['Year_2017'] = 1 if saved_selection['year'] == 2017 else 0
        example_row['Year_2018'] = 1 if saved_selection['year'] == 2018 else 0

        print(example_row)  # For debugging purposes

    def highlight_button(section, button):
        if last_selected[section]:
            last_selected[section].config(bg='SystemButtonFace')
        button.config(bg='lightblue')
        last_selected[section] = button

    def get_button(section, value):
        if section == 'year':
            return next(b for b in year_buttons if int(b.cget('text')) == value)
        elif section == 'area':
            return next(b for b in area_buttons if b.cget('text').lower() == value)
        elif section == 'type_risk':
            return next(b for b in type_risk_buttons if int(b.cget('text').split()[-1]) == value)
        elif section == 'classification':
            return next(b for b in classification_buttons if b.cget('text').lower() == value)

    def initialize_default_values():
        for section, value in saved_selection.items():
            save_and_update(section, value)
        highlight_button('year', get_button('year', saved_selection['year']))
        highlight_button('area', get_button('area', saved_selection['area']))
        highlight_button('type_risk', get_button('type_risk', saved_selection['type_risk']))
        highlight_button('classification', get_button('classification', saved_selection['classification']))
        
        
        
    def open_input_portfolio():
        input_window = tk.Toplevel(root)
        input_window.title("Input Window")

        # Create and pack the percentage entry
        tk.Label(input_window, text="Enter the VaR you want to compute (0-100):").pack(pady=5)
        percentage_entry = tk.Entry(input_window)
        percentage_entry.pack(pady=5)
    
        # Create and pack the integer entry
        tk.Label(input_window, text="Enter the amount of money you would like to reserve for this pool:").pack(pady=5)
        integer_entry = tk.Entry(input_window)
        integer_entry.pack(pady=5)
        
        def display_information():
            global analysis_frame
    
            # Delete existing analysis_frame if it exists
            if analysis_frame is not None:
                analysis_frame.destroy()

            # Create a new analysis frame inside the main window
            analysis_frame = tk.Frame(main_frame)
            analysis_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Create a text widget to display the information
            text_widget = tk.Text(analysis_frame, wrap=tk.NONE)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Insert information into the text widget
            for idx, item in enumerate(information, start=1):
                # Insert the pool number
                text_widget.insert(tk.END, f"Pool {idx}\n", "bold")
                # Insert each label and value
                for label, value in item:
                    text_widget.insert(tk.END, f"{label}: {value}\n\n")
                # Insert a red line after each entry in the array
                text_widget.insert(tk.END, "\n", "red_line")
                text_widget.insert(tk.END, "-" * 50 + "\n", "red_line")

            # Apply the tag for the red line and bold text
            text_widget.tag_configure("red_line", foreground="red")
            text_widget.tag_configure("bold", font=("Helvetica", 12, "bold"))
            # Scroll to the bottom
            text_widget.yview_moveto(1.0)
            # Disable the text widget to make it read-only
            text_widget.config(state=tk.DISABLED)


            # Create a vertical scrollbar and link it to the text widget
            scrollbar = tk.Scrollbar(analysis_frame, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)

        def get_values():
            # Retrieve slider values
            global yearly_sim

            # Retrieve percentage value
            try:
                percentage = float(percentage_entry.get())
                if 0 <= percentage <= 100:
                    print(f"Entered Percentage: {percentage}")
                else:
                    print("Percentage must be between 0 and 100")
                    percentage = 99.5
            except:
                print("Invalid percentage input")
                percentage = 99.5
        
            # Retrieve integer value
            try:
                integer_value = int(integer_entry.get())
                print(f"Entered Integer Value: {integer_value}")
            except:
                print("Invalid integer input. Setting value to 0.")
                integer_value = 0

            input_window.destroy()
            claims_portfolio = []
            premia_portfolio = []
            for i in range(1000):
                test = 0
                test1 = 0
                for s in range(len(yearly_sim)):             
                    test += ((yearly_sim[s][0][i][0]))
                    test1 += ((yearly_sim[s][0][i][1]))
                claims_portfolio.append(test)
                premia_portfolio.append(test1)
            total_clients = np.sum([np.sum(item[1]) for item in yearly_sim])
            summm = integer_value
            counter = 0
            for s in range(1000):
                for i in range(365):
                    if premia_portfolio[s][i] + summm - claims_portfolio[s][i] < 0:
                        counter += 1
                        break
                # Update progress bar)
                percent_complete = (s + 1) / 1000 * 100
                print(percent_complete)
                progress_var.set(percent_complete)
                progress_label.config(text=f"{int(percent_complete)}% complete")
                root.update_idletasks()
                    
            ruin = counter / 1000
                
                

            sums = np.sum(claims_portfolio, axis=1)
            sums.sort()
            premias = np.sum(premia_portfolio, axis=1)
            info = [
                ("Characteristics of the pool", "This is the whole portfolio with all the previous pools (not portfolios) included"),
                ("Total people in the pool", total_clients),
                ("Percentile analyzed for VaR", percentage),
                ("VaR value", sums[int(percentage * 10) -1]), #important to substract 1, otherwise it would be out of index and wrong by one value
                ("Starting amount of money of the pool", integer_value),
                ("Ruin probability", ruin),
                ("Amount of claims to be paid", np.sum(sums) /1000),
                ("Amount of premia received", np.sum(premias) /1000),
                ("Variance", np.var(sums))
            ]
            information.append(info)
            
            display_information()
            
            thresholds = np.linspace(0, np.max(sums), num=50)  # Define thresholds from 0 to max(data)
                    # Plot histograms
            plot_histogram(sums, bins=30, title="Histogram of Normally Distributed Data", xlabel="Value", ylabel="Frequency")

            plot_mean_excess(sums, thresholds)       

        # Create and pack the submit button
        tk.Button(input_window, text="Submit", command=get_values).pack(pady=10)
        
    def open_input_window(lower, upper , predict, alpha):
        input_window = tk.Toplevel(root)
        input_window.title("Input Window")
    
        # Create and pack the sliders with initial values
        tk.Label(input_window, text="Value of p").pack(pady=5)
        slider1 = tk.Scale(input_window, from_=float(lower[0]), to=float(upper[0]), orient='horizontal', resolution=0.01)
        slider1.set(predict)  # Set the initial value of slider1
        slider1.pack(pady=5)

        tk.Label(input_window, text="Value of alpha:").pack(pady=5)
        slider2 = tk.Scale(input_window, from_=0.1, to=1, orient='horizontal', resolution=0.01)
        slider2.set(alpha)  # Set the initial value of slider2
        slider2.pack(pady=5)

        # Create and pack the percentage entry
        tk.Label(input_window, text="Enter the VaR you want to compute (0-100):").pack(pady=5)
        percentage_entry = tk.Entry(input_window)
        percentage_entry.pack(pady=5)
    
        # Create and pack the integer entry
        tk.Label(input_window, text="Enter the amount of money you would like to reserve for this pool:").pack(pady=5)
        integer_entry = tk.Entry(input_window)
        integer_entry.pack(pady=5)
        
        def display_information():
            global analysis_frame
    
            # Delete existing analysis_frame if it exists
            if analysis_frame is not None:
                analysis_frame.destroy()

            # Create a new analysis frame inside the main window
            analysis_frame = tk.Frame(main_frame)
            analysis_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Create a text widget to display the information
            text_widget = tk.Text(analysis_frame, wrap=tk.NONE)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Insert information into the text widget
            for idx, item in enumerate(information, start=1):
                # Insert the pool number
                text_widget.insert(tk.END, f"Pool {idx}\n", "bold")
                # Insert each label and value
                for label, value in item:
                    text_widget.insert(tk.END, f"{label}: {value}\n\n")
                # Insert a red line after each entry in the array
                text_widget.insert(tk.END, "\n", "red_line")
                text_widget.insert(tk.END, "-" * 50 + "\n", "red_line")

            # Apply the tag for the red line and bold text
            text_widget.tag_configure("red_line", foreground="red")
            text_widget.tag_configure("bold", font=("Helvetica", 12, "bold"))
            # Scroll to the bottom
            text_widget.yview_moveto(1.0)

            # Disable the text widget to make it read-only
            text_widget.config(state=tk.DISABLED)

            # Create a vertical scrollbar and link it to the text widget
            scrollbar = tk.Scrollbar(analysis_frame, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)

        def get_values():
            # Retrieve slider values
            global data
            global information
            slider1_value = slider1.get()
            slider2_value = slider2.get()
            print(f"Slider 1 Value: {slider1_value}")
            print(f"Slider 2 Value: {slider2_value}")

            # Retrieve percentage value
            try:
                percentage = float(percentage_entry.get())
                if 0 <= percentage <= 100:
                    print(f"Entered Percentage: {percentage}")
                else:
                    print("Percentage must be between 0 and 100")
                    percentage = 99.5
            except:
                print("Invalid percentage input")
                percentage = 99.5
        
            # Retrieve integer value
            try:
                integer_value = int(integer_entry.get())
                print(f"Entered Integer Value: {integer_value}")
            except:
                print("Invalid integer input. Setting value to 0.")
                integer_value = 0

            input_window.destroy()
                # Generate samples
            data = []
            counter = 0
            sum_claims = 0
            sum_premia = 0
            yearly_simulation = []
            for i in range(1000):   

                simulate_result = generate_sample(slider2_value, slider1_value, saved_integer_value)
                sum_claims += np.sum(simulate_result)
                print(np.sum(simulate_result))
                # Initialize the year vector with zeros
                year_claims = np.zeros(365) #365 days

                # Randomly choose positions and place the simulate_result values in the year vector
                for value in simulate_result:
                    pos = np.random.randint(0, 365)  # Random position between 0 and 364
                    year_claims[pos] += value  # Add the value to the chosen position
                print(np.sum(year_claims))
                year_premia = np.zeros(365)
                premia = np.random.exponential(1, saved_integer_value)
                sum_premia += np.sum(premia)
                # Randomly choose positions and place the simulate_result values in the year vector
                for value in premia:
                    pos = np.random.randint(0, 365)  # Random position between 0 and 365
                    year_premia[pos] += value  # Add the value to the chosen position
                yearly_simulation.append([year_claims, year_premia])
                summ = integer_value   
                for s in range(365):
                    if integer_value + year_premia[s] - year_claims[s] < 0:
                        counter += 1
                        break
                    
                data.append(np.sum(simulate_result))
                # Update progress bar
                percent_complete = (i + 1) / 1000 * 100
                progress_var.set(percent_complete)
                progress_label.config(text=f"{int(percent_complete)}% complete")
                root.update_idletasks()

            progress_label.config(text="Simulation complete")
            yearly_sim.append([yearly_simulation, saved_integer_value])
            data.sort()
            ruin = counter / 1000
            print(sum_claims)
            info = [
                ("Characteristics of the pool", example_row),
                ("Amount of people in the pool", saved_integer_value),
                ("Value of p", slider1_value),
                ("Value of alpha", slider2_value),
                ("Percentile analyzed for VaR", percentage),
                ("VaR value", data[int(percentage * 10) -1]), #important to substract 1, otherwise it would be out of index and wrong by one value
                ("Starting amount of money of the pool", integer_value),
                ("Ruin probability", ruin),
                ("Amount of premia recevied", sum_premia /1000),
                ("Amount of claims to be paid", sum_claims /1000),
                ("Variance", np.var(data))
            ]
            information.append(info)
            
            display_information()
            
            thresholds = np.linspace(0, np.max(data), num=50)  # Define thresholds from 0 to max(data)
                    # Plot histograms
            plot_histogram(data, bins=30, title="Histogram of Normally Distributed Data", xlabel="Value", ylabel="Frequency")

            plot_mean_excess(data, thresholds)       

        # Create and pack the submit button
        tk.Button(input_window, text="Submit", command=get_values).pack(pady=10)

    # Initialize Tkinter variables and setup
    
    root = tk.Tk()
    root.title("Model Summary and Example Row Update")
    root.geometry('900x600')
    root.resizable(True, True)
    # Set minimum window size
    root.minsize(800, 800)
    integer_value = tk.IntVar(value=0)

    # Initialize dictionaries
    example_row = {
        'conts': 1,
        'Area_1': 0,
        'Type_risk_2': 0,
        'Type_risk_3': 0,
        'Type_risk_4': 0,
        'Classification_fast': 0,
        'Classification_medium': 0,
        'Year_2016': 0,
        'Year_2017': 0,
        'Year_2018': 0,
    }

    last_selected = {'year': None, 'area': None, 'type_risk': None, 'classification': None}
    saved_selection = {'year': 2018, 'area': 'urban', 'type_risk': 3, 'classification': 'fast'}
    saved_integer_value = 0

    # Create main frame


    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create text widget to display the summary
    text_frame = tk.Frame(main_frame)
    text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    summary_text = negbinom_result.summary().as_text()
    text_widget = tk.Text(text_frame, wrap=tk.NONE)
    text_widget.insert(tk.END, summary_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.config(yscrollcommand=scrollbar.set)

    # Create frame for buttons and integer input
    button_frame = tk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # Create category labels, integer entry, and buttons
    tk.Label(button_frame, text="Year:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    tk.Label(button_frame, text="Area:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
    tk.Label(button_frame, text="Type Risk:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
    tk.Label(button_frame, text="Classification:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
    tk.Label(button_frame, text="Enter how many insured people you want in this specific pool:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
    integer_entry = tk.Entry(button_frame, textvariable=integer_value)
    integer_entry.grid(row=4, column=1, padx=5, pady=5)

    save_integer_button = tk.Button(button_frame, text="Save Integer", command=save_integer)
    save_integer_button.grid(row=4, column=2, padx=5, pady=5)

    year_buttons = [tk.Button(button_frame, text=str(year), command=lambda y=year: save_and_update('year', y)) for year in [2015, 2016, 2017, 2018]]
    area_buttons = [tk.Button(button_frame, text=area, command=lambda a=area.lower(): save_and_update('area', a)) for area in ["Urban", "Rural"]]
    type_risk_buttons = [tk.Button(button_frame, text=f"Type Risk {risk}", command=lambda r=risk: save_and_update('type_risk', r)) for risk in [1, 2, 3, 4]]
    classification_buttons = [tk.Button(button_frame, text=classification, command=lambda c=classification.lower(): save_and_update('classification', c)) for classification in ["Slow", "Medium", "Fast"]]

    # Grid buttons
    for i, button in enumerate(year_buttons): button.grid(row=0, column=i + 1, padx=5, pady=5)
    for i, button in enumerate(area_buttons): button.grid(row=1, column=i + 1, padx=5, pady=5)
    for i, button in enumerate(type_risk_buttons): button.grid(row=2, column=i + 1, padx=5, pady=5)
    for i, button in enumerate(classification_buttons): button.grid(row=3, column=i + 1, padx=5, pady=5)
   

       


    def simulate():
        global simulate_result
        global data
            
        # Store the result in a global variable
        simulate_result = (example_row, saved_integer_value)
        
        text_frame.pack_forget()
        hist_frame = None
        spec = pd.DataFrame([example_row])
        mu = negbinom_result.predict(spec).iloc[0]
        alpha = 0.1760965105877899  # This is your dispersion parameter (from the model fitting)
        p = alpha / (alpha + mu)
        eta = np.dot(spec, negbinom_result.params)
        cov_matrix = negbinom_result.cov_params()
        var_eta = np.dot(np.dot(spec, cov_matrix), spec.T).diagonal()
        se_eta = np.sqrt(var_eta)
        z_score = norm.ppf((1 + 0.95) / 2)
        eta_lower = eta - z_score * se_eta
        eta_upper = eta + z_score * se_eta
        mu_lower = links.NegativeBinomial(0.1760965105877899).inverse(eta_lower)
        mu_upper = links.NegativeBinomial(0.1760965105877899).inverse(eta_upper)
        p_lower = alpha / (alpha + mu_lower)
        p_upper = alpha / (alpha + mu_upper)
        open_input_window(p_lower, p_upper, p, alpha )
        


        
    simulate_button = tk.Button(button_frame, text="Simulate", command=simulate)
    simulate_button.grid(row=4, column=3, padx=5, pady=5)
    # Create progress bar next to the "Simulate" button
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(button_frame, variable=progress_var, maximum=100)

    # Create label to show the percentage complete next to the progress bar
    progress_label = tk.Label(button_frame, text="0% complete")
    progress_label.grid(row=4, column=5, padx=5, pady=5)
    
    progress_bar.grid(row=4, column=4, padx=5, pady=5, sticky='w')
    simulate_button = tk.Button(button_frame, text="Portfolio", command=open_input_portfolio)
    simulate_button.grid(row=5, column=2, padx=5, pady=5)
    # Initialize default values
    initialize_default_values()

    # Start the Tkinter main loop
    root.mainloop()

    
# Load the GLM model from the pickle file
with open('negbinom_model.pkl', 'rb') as f:
    negbinom_result = pickle.load(f)
    
analysis_frame = None
data = []
yearly_sim = []
information = []
first_page(negbinom_result)















