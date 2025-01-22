
"""
This script defines a command-line interface for running and managing TSLAMM (Sea Level Affecting Marshes Model) simulations.
Classes:
    SimulationShell: A class that provides methods to load, save, set parameters, and run TSLAMM simulations.
Functions:
    __init__: Initializes the SimulationShell instance and sets up signal handlers.
    setup_signal_handler: Sets up signal handlers for graceful termination.
    signal_handler: Handles interruptions and exits the program cleanly.
    load: Loads a simulation from a specified file.
    show: Displays the value of a specified parameter or all parameters.
    sethelp: Displays help information for the set command.
    showhelp: Displays help information for the show command.
    set: Sets the value of a specified parameter.
    get_parameter_value: Retrieves and displays the value of a specified parameter.
    save: Saves the current simulation to a file.
    saveas: Saves the current simulation to a specified file.
    new_model: Creates a new simulation model.
    run_model: Runs the simulation model.
    run: Runs the command-line shell.
    process_command: Processes a command entered by the user.
Usage:
    Run the script and follow the command-line prompts to manage TSLAMM simulations.
"""

import sys
import os
import signal
import shlex
from LatinHypercubeRun import uncert_run
from SensitivityRun import sens_run
from SLR6 import TSLAMM_Simulation
from app_global import VERSION_NUM, IPCCScenarios, IPCCEstimates, ProtectScenario

# Define parameter types and their expected data types
parm_types = [
    "file_name",            "{string}",
    "sim_name",             "{string}",
    "time_step",            "{int}",
    "max_year",             "{int}",
    "run_specific_years",   "{true/false}",
    "years_string",         "{\"comma separated ints\"}",
    "run_uncertainty",      "{true/false}",
    "run_sensitivity",      "{true/false}",
    "gis_years",            "{\"comma separated ints\"}",
    "gis_each_year",        "{true/false}",
    "elev_file_name",       "{string}",
    "nwi_file_name",        "{string}",
    "slp_file_name",        "{string}",
    "imp_file_name",        "{string}",
    "ros_file_name",        "{string}",
    "dik_file_name",        "{string}",
    "vd_file_name",         "{string}",
    "uplift_file_name",     "{string}",
    "sal_file_name",        "{string}",
    "d2m_file_name",        "{string}",
    "storm_file_name",      "{string}",
    "output_file_name",     "{string}",
    "Scen_A1B",             "{true/false}",
    "Scen_A1T",             "{true/false}",
    "Scen_A1F1",            "{true/false}",
    "Scen_A2",              "{true/false}",
    "Scen_B1",              "{true/false}",
    "Scen_B2",              "{true/false}",
    "Est_Min",              "{true/false}",
    "Est_Mean",             "{true/false}",
    "Est_Max",              "{true/false}",
    "1_meter",              "{true/false}",
    "1.5_meter",            "{true/false}",
    "2_meter",              "{true/false}",
    "Protect_NoProtect",    "{true/false}",
    "Protect_ProtDeveloped","{true/false}",
    "Protect_ProtAll",      "{true/false}"
]

# Separate parameter names and types into two lists
parameters = parm_types[::2]  # Take every second element starting from index 0
ptypes = parm_types[1::2]     # Take every second element starting from index 1

class SimulationShell:
    def __init__(self):
        self.simulation: TSLAMM_Simulation = None
        self.running = True
        self.setup_signal_handler()

    # Setup signal handlers for graceful termination
    def setup_signal_handler(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    # Signal handler to handle interruptions
    def signal_handler(self, sig, frame):
        print("\nProcess interrupted. Exiting...")
        self.running = False
        sys.exit(0)  # This will ensure the program exits cleanly

    # Load a simulation from a file
    def load(self, file_path):
        try:
            file_path = file_path.strip('\'"')

            # Resolve relative paths or use the current working directory if no directory is specified
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)

            file_dir = os.path.dirname(file_path)
            os.chdir(file_dir)

            self.simulation = TSLAMM_Simulation()
            with open(file_path, 'r') as file:
                self.simulation.load_store(file, file_path, VERSION_NUM, True)
            print(f'Simulation Loaded: "{self.simulation.sim_name}"')
        except Exception as e:
            print(f'Failed to load simulation: {e}')

    # Show the value of a parameter or all parameters
    def show(self, parameter):
        if self.simulation is None:
            print("No simulation loaded")
            return

        if parameter == "?":
            self.showhelp(parameter)
        elif parameter == "all":
            for param in parameters:
                self.get_parameter_value(param)
        else:
            self.get_parameter_value(parameter)

    # Display help for the set command
    def sethelp(self, parameter= None):
        if parameter == "?":
            print("input is 'set {parameter} {value}'.  Parameters listed below\n")
            for i in range(len(parameters)):
                print(f"     {parameters[i]} {ptypes[i]}")
        else:
            print("Type 'set ?' for a list of available parameters")

    # Display help for the show command
    def showhelp(self, parameter= None):
        if parameter == "?":
            print("input is 'show {parameter}' or 'show all'.  Parameters listed below\n")
            for param in parameters:
                print(f"     {param}")
        else:
            print("Type 'show ?' for a list of available parameters")

    # Set the value of a parameter
    def set(self, parameter, value):
        if self.simulation is None:
            print("No simulation loaded")
            return
        try:
            if parameter.startswith('Scen_'):
                param_dict = self.simulation.ipcc_scenarios
                key = IPCCScenarios[parameter]
                value = value.lower() in ['true', '1', 'yes']
                param_dict[key] = value
                print(f'{parameter} set to {value}')

            elif parameter.startswith('Est_'):
                param_dict = self.simulation.ipcc_estimates
                key = IPCCEstimates[parameter]
                value = value.lower() in ['true', '1', 'yes']
                param_dict[key] = value
                print(f'{parameter} set to {value}')

            elif parameter.startswith('Protect_'):
                param_dict = self.simulation.prot_to_run
                key = ProtectScenario[parameter.split('_')[1]]
                value = value.lower() in ['true', '1', 'yes']
                param_dict[key] = value
                print(f'{parameter} set to {value}')

            elif parameter in ['1_meter', '1.5_meter', '2_meter']:
                index_map = {'1_meter': 0, '1.5_meter': 1, '2_meter': 2}
                index = index_map[parameter]
                value = value.lower() in ['true', '1', 'yes']
                self.simulation.fixed_scenarios[index] = value
                print(f'{parameter} set to {value}')

            else:
                if hasattr(self.simulation, parameter):
                    current_value = getattr(self.simulation, parameter)
                    if isinstance(current_value, bool):
                        value = value.lower() in ['true', '1', 'yes']
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    setattr(self.simulation, parameter, value)
                    print(f'{parameter} set to {value}')
                else:
                    print(f'"{parameter}" is not a model setting.  Type set ? for a list of available parameters')

        except ValueError as e:
            print(f'"{value}" is not a valid setting for "{parameter}"')

    # Get the value of a parameter
    def get_parameter_value(self, parameter):
        if hasattr(self.simulation, parameter):
            value = getattr(self.simulation, parameter)
            print(f'{parameter}: {value}')
        else:
            # New logic for the extended parameters
            if parameter.startswith("Scen_"):
                scenario = IPCCScenarios[parameter]
                value = self.simulation.ipcc_scenarios[scenario]
            elif parameter.startswith("Est_"):
                estimate = IPCCEstimates[parameter]
                value = self.simulation.ipcc_estimates[estimate]
            elif parameter in ["1_meter", "1.5_meter", "2_meter"]:
                index = {"1_meter": 0, "1.5_meter": 1, "2_meter": 2}[parameter]
                value = self.simulation.fixed_scenarios[index]
            elif parameter.startswith("Protect_"):
                protect_scenario = ProtectScenario[parameter.split('_')[1]]
                value = self.simulation.prot_to_run[protect_scenario]
            else:
                print(f'"{parameter}" is not a model setting. Type show ? for a list of available parameters')
                return
            print(f'{parameter}: {value}')

    # Save the current simulation to a file
    def save(self, split_files =False):
        if self.simulation is None:
            print("No simulation loaded")
            return
        try:
            # Check if the file exists
            if os.path.exists(self.simulation.file_name):
                # Prompt the user for overwrite permission
                response = input(f'Overwrite {self.simulation.file_name}? (y/n): ').strip().lower()
                if response != 'y':
                    return

            # Open the file and save the simulation
            with open(self.simulation.file_name, 'w') as savefile:
                self.simulation.load_store(savefile, self.simulation.file_name, VERSION_NUM, False, split_files)

            if split_files:
                print("Model saved to multiple files, base file: " + self.simulation.file_name)
            else: print("Model saved to file: " + self.simulation.file_name)

        except Exception as e:
            print(f'Failed to save model: {e}')

    # Save the current simulation to a specified file
    def saveas(self, filen, split_files =False):
        if self.simulation is None:
            print("No simulation loaded")
            return

        filen = filen.strip('\'"')

        # Resolve relative paths or use the current working directory if no directory is specified
        if not os.path.isabs(filen):
            filen = os.path.abspath(filen)

        file_path = os.path.dirname(filen)
        os.chdir(file_path)

        try:
            # Check if the file exists
            if os.path.exists(filen):
                # Prompt the user for overwrite permission
                response = input(f'Overwrite {filen}? (y/n): ').strip().lower()
                if response != 'y':
                    return

            # Open the file and save the simulation
            with open(filen, 'w') as savefile:
                self.simulation.load_store(savefile, filen, VERSION_NUM, False, split_files)

            if split_files:
                print("Model saved to multiple files, base file: " + filen)
            else: print("Model saved to file: " + filen)

        except Exception as e:
            print(f'Failed to save model: {e}')

    # Create a new simulation model
    def new_model(self, CA_Categories=False):
        self.simulation = TSLAMM_Simulation(CA_Categories)
        print("New simulation created.")

    # Run the simulation model
    def run_model(self, cpu_count=None):
        if self.simulation is None:
            print("No simulation loaded")
            return
        try:
            if cpu_count:
                self.simulation.cpu_count = cpu_count
            if self.simulation.run_sensitivity:
                sens_run(self.simulation)
            if self.simulation.run_uncertainty:
                uncert_run(self.simulation)
            else:
                self.simulation.execute_run()

            print(f'Execution completed, model run log saved to: {self.simulation.run_record_file_name}')
        except Exception as e:
            print(f'Failed to run model: {e}')

        finally:
            self.simulation.dispose_mem()

    # Run the command line shell itself
    def run(self):
        print('type ? for a command list')
        while self.running:
            try:
                command = input('> ').strip()
                if command.lower() in ['exit', 'quit']:
                    break
                self.process_command(command)
            except KeyboardInterrupt:
                self.signal_handler(None, None)
            except Exception as e:
                print(f'Error processing command: {e}')

    # Process a command entered by the user
    def process_command(self, command):
        parts = shlex.split(command)  # Use shlex to split the command line correctly
        if not parts:
            return
        cmd = parts[0].lower()
        args = parts[1:]
        cmdlist = "    new\n\r    load {file name}\n\r    save {split_files(blank=False)}\n\r    saveas {file name, split_files(blank=False)} " \
                  "\n\r    set {parameter} {value}\n\r    show {parameter or all}\n\r    run_model {cpu_count(blank=all)} \n\r    quit"

        if cmd == 'load':
            if len(args) == 0:
                print('file_name is a required parameter')
            else:
                self.load(args[0])
        elif cmd == 'set' and len(args) ==0:
            self.sethelp()
        elif cmd == 'set' and len(args) ==1:
            self.sethelp(args[0])
        elif cmd == 'set' and len(args) == 2:
            self.set(args[0], args[1])
        elif cmd == 'show' and len(args) ==0:
            self.showhelp()
        elif cmd == 'show' and len(args) ==1:
            self.show(args[0])
        elif cmd == 'save':
            if len(args) == 1:
                self.save(split_files=bool(args[0]))
            else:
                self.save()
        elif cmd == 'saveas':
            if len(args) == 0:
                print('file_name is a required parameter')
            elif len(args) == 1:
                self.saveas(args[0])
            else:
                self.saveas(args[0], split_files=bool(args[1]))
        elif cmd == 'run_model':
            if len(args) == 1:
                self.run_model(cpu_count=int(args[0]))
            else:
                self.run_model()
        elif cmd == 'new':
            if len(args) == 1:
                self.new_model(CA_Categories=bool(args[0]))
            else:
                self.new_model(False)
        elif cmd == '?':
            print(cmdlist)
        else:
            print("Unknown command, type ? for a command list")

# Main entry point for the script
if __name__ == '__main__':
    shell = SimulationShell()
    shell.run()