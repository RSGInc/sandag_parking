# sandag-parking

This script prepares expected parking cost data for SANDAG MGRA series 15 zones. 

The processing includes the following steps organized into separate python modules:
1. `reductions.py`: Reduce/organize dataset and estimate model fit
2. `imputation.py`: Impute missing values
3. `districts.py`: Find the parking districts
4. `estimate_spaces.py`: Estimate spaces
5. `expected_cost.py`: Calculate expected costs

These modules inherit a few helper functions from `base.py`, which are all then inherited and run with `processing.py`to provide a single point of entry to the program. The script can be controlled with `settings.yaml`, where users specify inputs, outputs, parameters, and which models to run. 

The processing script can be run by either running `processing.py` or through command line by navigating to the directory containing the "parking" folder and executing the line:

```python -m parking```



