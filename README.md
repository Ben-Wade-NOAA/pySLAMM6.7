PySLAMM 6.7 Installation and Test Instructions
8/13/2024--  initial release instructions

Linux 

    1. Transfer the Installation File
        Use scp to transfer the file to the user directory
        update sudo if required "sudo apt update"
        install pip3 if required "sudo apt install python3-pip"
        
        note, if sudo apt is already running daily maintenance
        tasks you may need to wait a few minutes.
        
    2. install Python 3.11 if required
        "sudo install python3.11"
        
        (older versions of ubuntu such as 18.0.4 may require a manual install  
        https://medium.com/@elysiumceleste/how-to-install-python-3-11-3-on-ubuntu-18-04-e1cb4d404ef3 )
    
    3. install a virtual environment if required, 
        install the tools: "sudo apt install python3-venv"
        create the virtual environment:  "python -m venv test_env"
        activate the virtual environment: "source ~/venv/bin/activate"

    4. use pip3 to install the wheel file:  
        pip3 install pySLAMM-6.7.0-py3-none-any.whl  
    OR 
        pip3 install pyslamm-6.7.0.tar.gz

    5. navigate to the installation location to run the command line
        
        python3.11 pySLAMM6_7.py
    

Windows

    1. Install Python 3.11, download from https://www.python.org/downloads/
    
    2. install a virtual environment from a Windows CMD prompt
        python -m venv test_env
        test_env\Scripts\activate
        
    3. copy the installer file to a directory on the server and navigate to that folder
        pip3 install pySLAMM-6.7.0-py3-none-any.whl  
    OR 
        pip3 install pyslamm-6.7.0.tar.gz
        
    4. navigate to the location of pySLAMM6_7.py to run the command line

        e.g. cd test_env\Lib\site-packages\
    
        python pySLAMM6_7.py
        
Please send any questions to jclough@warrenpinnacle.com