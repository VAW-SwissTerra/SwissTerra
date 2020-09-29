# terra --- SwissTerra archival image processing

## Installation
Installation through conda is currently the only supported method.

```bash
# This will create an environment called swissterra and download all the packages
conda create --file environment.yml

# Install Metashape
pip install http://download.agisoft.com/Metashape-1.6.4-cp35.cp36.cp37-abi3-linux_x86_64.whl
```
then
```bash
python setup.py 
```
OR
```bash
python setup.py develop --user
```
to allow modification of the source code in place.

Also make sure to set up the `agisoft_LICENSE` environment variable to point to `metashape.lic` in the Metashape installation folder.

### Data
The data should either be in a folder/symlink called `input/` in the working directory, or be defined by the environment variable `SWISSTERRA\_INPUT\_DIR` (which takes precedence).

Run `terra data check` to validate that all files can be located.

## Usage
Run `terra` to see the how to run the program.
