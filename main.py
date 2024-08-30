import importlib

def run_analysis():
    print("Running Analysis...")
    # Place the code from analysis.py here or import and call its main function
    import analysis
    importlib.reload(analysis)
    analysis.main()

def run_simulation():
    print("Running Simulation...")
    # Place the code from simulation.py here or import and call its main function
    import simulation
    importlib.reload(simulation)
    simulation.main()

def main():
    while True:
        print("\nChoose an Action:")
        print("1. Analysis")
        print("2. Simulation")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            run_analysis()
        elif choice == '2':
            run_simulation()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

