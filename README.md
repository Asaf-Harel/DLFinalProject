# Asaf Harel's Deep Learning Final Project

## Installation:

<a href="https://drive.google.com/file/d/1KIRV340Bi-DOfOut-ZmCHHkVHLTh9Jzj/view?usp=sharing" target="_blank">--> Click
Here <--</a>

After Downloading, unzip the folder and run "Antivirus.exe"

**Note:** if the program crashes, follow the steps below

## Make the user experience easier

Copy the folder into `C:\Program Files\` and then right-click on the `Antivirus.exe` file and select `Create shortcut`
and save it to wherever you want.

## Fixing Crashing Issue

### Download source code

Clone the repository with ```git clone https://github.com/Asaf-Harel/DLFinalProject.git```


### Create only the necessary file

Create a file called `requirements.txt` and copy the content of `requirements.txt` from the GitHub repo into the file
you just created


### Install dependencies

Run the command ```pip install -r requirements.txt```

Now try running the program again, if it's still crashing try the following solution:

### Install PyInstaller

Run the command ```pip install pyinstaller```


### Compile app.py

Run the command ```pyinstaller app.spec```
**Note:** Make sure you followed the steps above and downloaded all the necessary dependencies


## Open app
Go to the `dist/app/` folder and run the program called `Antivirus.exe` 