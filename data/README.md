# Data Directory

This directory has both the observations we used in the measurement (`observations`) as well as the mocks we used for comparison (`mocks`). 

All files are in the same simplified format, one line per star, with Cartesian coordinates followed by observational coordinates. In the mocks, we also provide a flag in the last column (`sdss?`) for if the star would have been observed in SDSS (`1`) or not (`0`).

Note that this generation of mocks _does not_ put the LMC in the present-day location. Refer to the manuscript for the 'true' values derived from the parent simulations.