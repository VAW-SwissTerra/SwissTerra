# terra — SwissTerra archival image processing

## Requirements

* An **Unix based OS** (tested on Ubuntu and Arch Linux).

* **[Agisoft Metashape 1.6.5](https://www.agisoft.com/downloads/installer/)** (for another version, change accordingly in `environment.yml`).

* **[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)** to install the rest of the dependencies.

## Installation
1. 	
	Installation through conda is currently the only supported method.

	```bash
	git clone https://github.com/VAW-SwissTerra/SwissTerra.git
	cd SwissTerra
	# This will create an environment called swissterra and download all the packages
	conda env create --file environment.yml
	conda activate swissterra
	```
2. 	To install the `terra` cli interface, perform one of either actions:
	```bash
	python setup.py 
	```
	or one of the two to enable editing of the source code:
	```bash
	python setup.py develop --user
	```
	or
	```bash
	echo "alias terra='$(which python) $(pwd)/terra/__main__.py'" >> ~/.bashrc
	source ~/.bashrc
	```

	Make sure that the correct conda environment is activated before running this step.

3. 	Make sure to set up the `agisoft_LICENSE` environment variable to point to `metashape.lic` in the Metashape installation folder.

### Data
The data should either be in a folder/symlink called `input/` in the working directory, or be defined by the environment variable `SWISSTERRA_INPUT_DIR` (which takes precedence over the former).

Run `terra data check` to validate that all files can be located.

## Usage
Run `terra -h` to see the how to run the program.

1. Set up a dataset (TODO: ADD INSTRUCTIONS ON HOW TO DO THIS)

2. `terra files check`: See that all files can be found

3. `terra processing DATASET generate-inputs`: Generate all inputs with default settings.

4. `terra processing DATASET run`: Start the main pipeline
