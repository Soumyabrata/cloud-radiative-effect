# Cloud Radiative Effect Study Using Sky Camera

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript: S. Dev, S. Manandhar, F. Yuan, Y. H. Lee and S. Winkler, Cloud Radiative Effect Study Using Sky Camera, *Proc. IEEE AP-S Symposium on Antennas and Propagation and USNC-URSI Radio Science Meeting*, 2017. 

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

## Manuscript
The author version of this manuscript is `manuscript.PDF`. 

## Code Organization
All codes are written in python. Thanks to [Florian Savoy](https://github.com/FSavoy) for contributing the camera calibration code used in undistorting a sky/cloud image. 

### Dataset
All input dataset can be found in the folder `./input`.

### Core functionality
* `color16mask.py` Generates the red and blue ratio channel.
* `import_WS_CI.py` Imports the weather station data and also calculates the clearness index.
* `internal_calibration.py` Provides the internal calibration model of our sky camera.
* `make_cluster_mask.py` Generates the output binary sky/cloud image and computes the cloud coverage. 
* `nearest.py` Finds the nearest datapoint according to a criterion. 
* `normalize_array.py` Normalizes an input array. 
* `SG_solarmodel.py` Computes the total solar irradiance for Singapore clear sky model. 
* `showasImage.py` Normalizes an array/matrix in the range [0,255]. 
* `undistortImg.py` Undistorts a sky/cloud image based on the camera calibration model. 

### Reproducibility 
In addition to all the related codes, we have also shared the generated results. These files are contained in the folder `./results`.

The program `./Cloud Radiative Effect Study Using Sky Camera.ipynb` is the main script, that reproduces all the results. It uses different helper scripts stored in the folder `./helperFunctions`. It also reproduces the figures and tables in this associated paper.
