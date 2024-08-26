import tkinter as tk
import subprocess

def run_analysis():
    root.destroy()  # Close the window
    subprocess.run(['python', 'analysis.py'])

def run_simulation():
    root.destroy()  # Close the window
    subprocess.run(['python', 'simulation.py'])

# Create the main window
root = tk.Tk()
root.title("Choose an Action")

# Set initial window size (e.g., fullscreen)
root.geometry('1920x1080')  # Set initial size
root.resizable(True, True)  # Allow resizing in both directions

# Define a style for buttons
button_style = {
    'bg': '#4CAF50',  # Green background
    'fg': '#ffffff',  # White text
    'font': ('Arial', 16, 'bold'),  # Font style
    'bd': 5,          # Border width
    'relief': 'raised',  # Raised effect
    'width': 20,
    'height': 2
}

# Create the Analysis button
analysis_button = tk.Button(root, text="Analysis", command=run_analysis, **button_style)
analysis_button.pack(pady=20, fill=tk.BOTH, expand=True)

# Create the Simulation button
simulation_button = tk.Button(root, text="Simulation", command=run_simulation, **button_style)
simulation_button.pack(pady=20, fill=tk.BOTH, expand=True)

# Start the Tkinter event loop
root.mainloop()

