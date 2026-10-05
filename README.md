# Mitochondria Tracking in Microscopy Videos

[[DOI: 10.5281/zenodo.23166373](https://zenodo.org/records/23166373?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjNhODliZGNlLTIyZWMtNGRlZi04OTczLTRhMWMyNGQ4YzUzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI0YmE4MmY4YmJhMTJjNjQ0Y2I0MThkNGE1ZGE3ODYxNCJ9.B-m3EsRnJM7SY6fLk121ZGiZp6e8S-TJ7Y0isbmrLTXnh16rLKNBzbaQaj4DFmu0qqAD2BHLpnm-teW_cMCJUw)

Python pipeline for detecting and tracking mitochondria in fluorescence microscopy videos, with a machine-learning–based track association model.

# Entering the folder
cd mito-tracking

# Creating the environment
conda env create -f environment.yml
conda activate mito-tracking

# Instaling the package
pip install -e .

# Running the pipeline
python scripts/reproduce_paper.py
