## Environment Setup
### First, install the environment: Create a Python environment with version equal to 3.10 using conda env create -f environment.yml. This should be compatible with both Windows and Linux in theory.
### conda activate demo
### openslide frequently reporting errors is normal. Please try:
### conda install -c conda-forge openslide

## After the environment is ready, prepare to run. Only final.py is needed for the entire process.
### Open final.py
<img width="448" alt="457e63572b85d84ab02b21018d19098" src="https://github.com/user-attachments/assets/6c7a03fd-ab3a-4b77-a0c1-40117b1140db" />


### Specify an arbitrary root directory.
### ### Download the paired tif and xml files to any directory (continuously decompress from Baidu Netdisk. All xml files are in the annotation archive. The tif files need to be obtained by further decompressing within the center and patient folders). However, note that the tif and xml files should be paired (i.e., one patient corresponds to one annotation).

### Enter the patient number followed by _level6, such as 008_level6, 004_node_4_level6. Please ensure to include _level6
### After setting the root directory, the tif/xml directory, and the patient number, simply run final.py.
### To process the next tif-xml pair, run it again.

