# High Frequency TRF
code for computing high frequency TRFs from MEG data.  
based on eelbrain https://eelbrain.readthedocs.io/en/stable/

**boostTRF.py**
Compute TRFs from MEG sqd files
Ensure that the sqd filepath, stimuli filepath and source localization filepaths (forward solutions, mri directory) are valid
Run the script to compute high frequency TRFs for each subject in the sqd filepath and save it as a pickle file

**testTRF.py**
Module to perform statistical tests using permutation tests and TFCE (https://doi.org/10.1016/j.neuroimage.2008.03.061)
````
import testTRF as T`
ds = T.load_smooth(T.pklFv)
tests = T.run_tests_vol(ds) 
````

Tests can be customized inside the ``run_tests_vol()`` function