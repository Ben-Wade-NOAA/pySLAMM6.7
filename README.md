# PySLAMM 6.7 Installation and Test Instructions

**Version:** Initial release on 08/19/2024

---
![slamm6](https://github.com/user-attachments/assets/5990dfd2-0c43-456f-be7c-3d07a4c5fb52)

## Linux

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

## Windows

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
