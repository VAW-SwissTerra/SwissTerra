# SwissTerra — SwissTerra archival image processing

## Requirements

* An **Unix based OS** (tested on Ubuntu and Arch Linux).

* **[Agisoft Metashape 1.7.1](https://www.agisoft.com/downloads/installer/)** (for another version, change accordingly in `environment.yml`).

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
	pip install .
	```
	or one of the two to enable editing of the source code:
	```bash
	pip install --editable .
	```

	Make sure that the correct conda environment is activated before running this step.

3. 	Set up the `agisoft_LICENSE` environment variable to point to `metashape.lic` in the Metashape installation directory.

### Required data
The data should either be in a directory/symlink called `input/` in the working directory, or be defined by the environment variable `SWISSTERRA_INPUT_DIR` (which takes precedence over the former).

* `input/images/*.tif`: The directory/symlink with the input images.
* `input/image_metadata/*.txt`: The directory/symlink with the image metadata text files.
* `input/basedata/swissALTI3D_pr2019_LV95.tif`: A non-free DEM mosaic from 2011--2019 ([swisstopo, 2019](https://shop.swisstopo.admin.ch/de/products/height_models/alti3D)).
* `input/shapefiles/swissALTI3D_Metatatenlayer_032019/Metadata_SwissALTI3D_preRelease2019_LV95.shp`: Metadata on the approximate capture dates of the DEM mosaic.
* `input/shapefiles/swissTLM3D_lakes.shp`: Lakes from a non-free land-use map ([swisstopo, 2020](https://shop.swisstopo.admin.ch/en/products/landscape/tlm3D)).
* `input/shapefiles/Glacierarea_1935_split.shp`: Glacier outlines from ~1935 ([Freudiger et al., 2018](https://doi.org/10.6094/UNIFR/15008)).
* `input/shapefiles/inventory_sgi2016_r2020/SGI_2016_glaciers.shp`: Modern glacier outlines ([GLAMOS, 2020](https://doi.glamos.ch/data/inventory/inventory_sgi2016_r2020.html)).

The files below are only needed for the `terra overview` functionality:

* `input/shapefiles/SGI_1973.shp`: Glacier outlines from 1973 ([Müller et al., 1976](https://doi.glamos.ch/data/inventory/inventory_sgi1973_r1976.zip)).
* `input/shapefiles/V_TERRA_BGDI.shp`: Image viewshed polygons (swisstopo).
* `input/shapefiles/V_TERRA_VIEWSHED_PARAMS.shp`: Image location points (swisstopo). (TODO: could be replaced with the metadata files)

Run `terra files check-inputs` to validate that all files can be located.

## Usage
Run `terra -h` to see the how to run the program.

`DATASET` can either be a specific instrument at a specific year (e.g. `Wild13_1924`. Type `terra processing .` to see every available choice), or `full` for all the data.

1. `terra files check-inputs`: See that all files can be found.

2. `terra preprocessing train-fiducials`: Manual intervention. Train the fiducial mark matcher.

3. `terra processing DATASET generate-inputs`: Generate all inputs with default settings.

4. `terra processing DATASET run`: Start the main photogrammetric pipeline.

TODO: Make a CLI entry point for the script below.

5. `python terra/evaluation.py`: Coregister the photogrammetrically created DEMs

6. `python terra/evaluation.py`: Generate dDEMs and analyse the result.


## Credit
The work was performed at [@VAW_glaciology](https://twitter.com/VAW_glaciology) in the frame of the work presented in Mannerfelt et al., ([2022; in discussion](https://doi.org/10.5194/tc-2022-14)).
It was co-financed by the Swiss Federal Office of Meteorology and Climatology ([MeteoSwiss](https://www.meteoswiss.admin.ch/)) in the frame of [GCOS Swizerland](https://www.meteoswiss.admin.ch/home/research-and-cooperation/international-cooperation/gcos/gcos-switzerland-projects.html).
