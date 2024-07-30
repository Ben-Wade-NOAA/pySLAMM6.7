import sys
import time
from SLR6 import TSLAMM_Simulation
from app_global import VERSION_NUM
from LatinHypercubeRun import uncert_run
from SensitivityRun import sens_run

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <file>")
        sys.exit(1)

    file_path = sys.argv[1]

    start = time.time()
    # Create an instance of TSLAMM_Simulation
    simulation = TSLAMM_Simulation()

    # Open the file and call the load_store method
    with open(file_path, 'r') as file:
        simulation.load_store(file, file_path, VERSION_NUM, True)

    # if drive_externally:
    if simulation.run_sensitivity:
        sens_run(simulation)
    if simulation.run_uncertainty:
        uncert_run(simulation)
    else:
        simulation.execute_run()

    simulation.dispose_mem()

    end = time.time()
    elapsed = round(end - start)
    print(f"Elapsed = {elapsed} seconds")


if __name__ == "__main__":
    #  multiprocessing.set_start_method('spawn')
    main()
