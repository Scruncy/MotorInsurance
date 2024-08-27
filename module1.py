import tkinter as tk
from tkinter import ttk
import time

def start_simulation():
    total_work = 100  # The total amount of work (e.g., 100 iterations)
    
    for i in range(total_work):
        time.sleep(0.05)  # Simulating work (sleep for 50ms)
        
        # Calculate the percentage of work done
        percent_complete = (i + 1) / total_work * 100
        
        # Update the progress bar
        progress_var.set(percent_complete)
        
        # Update the label showing the percentage
        progress_label.config(text=f"{int(percent_complete)}% complete")
        
        # Refresh the GUI
        root.update_idletasks()

# Set up the main window
root = tk.Tk()
root.title("Simulation Progress")

# Set up the progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=20)

# Set up the label to show the percentage
progress_label = tk.Label(root, text="0% complete")
progress_label.pack()

# Set up the button to start the simulation
start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()
