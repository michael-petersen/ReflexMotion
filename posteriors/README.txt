This folder contains various posteriors that were calculated for the paper. The posteriors themselves are an uncommented mess, so we provided a reader:

from reflexmotion import posteriorreader as pr
ALL     = pr.read_posterior('posteriors/all_d040150_sgr20_L3000_cov.posteriors2')
print(ALL.keys())
# dict_keys(['vtravel', 'phi', 'theta', 'sigmar', 'sigmap', 'sigmat', 'vra', 'vth', 'vphi'])


For clarity, the columns are as follows:
0   vtravel
1   phi, the azimuthal apex location (in radians)
2   theta, the cosine of the colatitude apex location (in radians). In spherical coordinates, so b=90-arccos(theta.deg)
3   1/sigmar^2, the hyperparameter in the radial direction
4   1/sigmap^2, the hyperparameter in the azimuthal direction
5   1/sigmat^2, the hyperparameter in the colatitude direction
6   v_radial, the mean velocity in the radial direction
7   v_theta, the mean velocity in the colatitude direction
8   v_phi, the mean velocity in the azimuthal direction
9   log likelihood

Each row of the posterior file is an output from a chain. The reader will generate the dictionary above.
