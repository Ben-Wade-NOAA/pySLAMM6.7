# PySLAMM 6.7 Installation and Test Instructions

**Version:** Initial release on 08/19/2024

---

![slamm6](https://github.com/user-attachments/assets/5990dfd2-0c43-456f-be7c-3d07a4c5fb52)

## Introduction

**PySLAMM 6.7** is a Python translation of the SLAMM 6.7 code originally written in Delphi (object-oriented Pascal for Windows). This project was funded by the Coastal Resilience branch of NCCOS (NOAA) under contract to Consolidated Safety Services Incorporated (CSS), with oversight from Christine Addison Buckel, Ramin Familkhalili, and Rebecca Atkins.

All model process code has been translated to Python and tested across multiple operating systems. While the graphical user interface (GUI) is not included, PySLAMM 6.7 can read SLAMM text files produced by SLAMM 6.7 Delphi and has been tested to machine accuracy against model results from the original version. This allows models to be created using the Windows GUI and then executed in Python. Additionally, PySLAMM 6.7 can read inputs from and write outputs to GeoTIFF format, simplifying access to model data via GIS.

### Model Access Methods

PySLAMM 6.7 can be accessed in Python through several methods:

- **Command Line Execution:**
  - Run the `SLAMM_Run.py` script from the command line, passing the model input file as a parameter. The model run will complete automatically.

- **Simplified Command-Line Interface:**
  - Run the `pySLAMM6_7.py` file to access a set of commands (e.g., `Load`, `Run_model`). Enter `?` to view available commands.

- **Python Scripting:**
  - Import SLAMM6.7 objects into your Python script and modify them directly for customized model runs.

### Documentation

For detailed guidance, please refer to the following documents:

- [PySLAMM 6.7 Technical Documentation](https://github.com/WarrenPinnacle/pySLAMM6.7/blob/main/docs/pySLAMM_6.7_Technical_Documentation.pdf)

- [PySLAMM 6.7 User's Manual](https://github.com/WarrenPinnacle/pySLAMM6.7/blob/main/docs/pySLAMM_6.7_Users_Manual.pdf)

---
## Linux Installation

1. **Transfer the Installation File**
   - Use `scp` to transfer the file to the user directory.
   - Update `sudo` if required: `sudo apt update`.
   - Install `pip3` if required: `sudo apt install python3-pip`.
   
   > **Note:** If `sudo apt` is already running daily maintenance tasks, you may need to wait a few minutes.

2. **Install Python 3.11 if required**
   - Install Python 3.11: `sudo apt install python3.11`.
   - For older versions of Ubuntu (e.g., 18.04), you may need to manually install Python 3.11:
     [How to Install Python 3.11 on Ubuntu 18.04](https://medium.com/@elysiumceleste/how-to-install-python-3-11-3-on-ubuntu-18-04-e1cb4d404ef3).

3. **Install a Virtual Environment if required**
   - Install the virtual environment tools: `sudo apt install python3-venv`.
   - Create the virtual environment: `python3 -m venv test_env`.
   - Activate the virtual environment: `source ~/test_env/bin/activate`.

4. **Install the PySLAMM Package**
   - Use `pip3` to install the wheel file:  
     ```bash
     pip3 install pySLAMM-6.7.0-py3-none-any.whl
     ```
   - OR use `pip3` to install the tar.gz file:
     ```bash
     pip3 install pyslamm-6.7.0.tar.gz
     ```

5. **Navigate to the Installation Location and Run the Command Line**
   - Run the command:
     ```bash
     python3.11 pySLAMM6_7.py
     ```

6. **Determine the Installation Location**
   - The installation location will vary depending on the use of a virtual environment, etc. In a virtual environment, it may be installed to `/virtual/lib/python3.11/site-packages/`. To find the installation location, type:
     ```bash
     pip3 show pySLAMM
     ```

   - The example installation may be in the `.local/Kakahaia` folder. To verify this, you can use the Linux command:
     ```bash
     sudo find / -iname "Kakahaia.txt"
     ```

---

## Windows Installation

1. **Install Python 3.11**
   - Download Python 3.11 from [python.org](https://www.python.org/downloads/).

2. **Install a Virtual Environment**
   - From a Windows CMD prompt:
     ```cmd
     python -m venv test_env
     test_env\Scripts\activate
     ```

3. **Install the PySLAMM Package**
   - Copy the installer file to a directory on the server and navigate to that folder.
   - Use `pip3` to install the wheel file:
     ```cmd
     pip3 install pySLAMM-6.7.0-py3-none-any.whl
     ```
   - OR use `pip3` to install the tar.gz file:
     ```cmd
     pip3 install pyslamm-6.7.0.tar.gz
     ```

4. **Navigate to the Location of `pySLAMM6_7.py` and Run the Command Line**
   - Example:
     ```cmd
     cd test_env\Lib\site-packages\
     python pySLAMM6_7.py
     ```

---

For any questions, please contact [jclough@warrenpinnacle.com](mailto:jclough@warrenpinnacle.com).
