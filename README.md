# P6
Repository for the bachelor project with the data necessary for the challenge GutBrainIE @ CLEF 2026 task 6.


# Depedencies
This project has been tested on Python 3.12, but it should also work for 3.11. Newer version might not work since some packages have not been updated yet. The depedencies for this project can be found in requirement.txt file or pyproject.toml. To install the depedencies the following commands can be used: 
```pip install -r requirements.txt```
pyproject.toml,
```pip install .```


# Data
The data for this project is from the [GutBrain@CLEF2026](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) challenge. After downloading the data, the two files `Annotations` and `Articles` should be placed inside of `./Data/raw/`. 

# Preprocessing
The preprocessing file ´preprocessing.py` will generate the necessary files by running it just makes sure the data is downloaded and is in the correct place. 

# Trainning and Evaluating models

