import numpy as np
from SIMILE.simulate_milli_lens import simlens



simlens = simlens()
simlens.obs_date_string = '2024-09-24 00:00:00'
simlens.output_dir = './sim_data/'

### Parameters to be varied for the simulations ###
# This will simulate 38880 datasets (194400 files) #
F_A_arr = [30, 50, 100]    # primary component  total flux
flux_ratio_arr = [1./7, 1./35]    # flux ratios
R_B_arr = [1.0, 3.0, 10.0, 100.0]    # distance between components in mas
angles = ['maj', 'min']    # placing secondary component along clean beam
    # major or minor axis
scan_length_arr = [1, 3, 10]    # scan length in minutes
ra_arr = np.array([   # RA in fractional hours
                   (0 + 0 / 60 + 0 / 3600),
                   (4 + 0 / 60 + 0 / 3600),
                   (8 + 0 / 60 + 0 / 3600),
                   (12 + 0 / 60 + 0 / 3600),
                   (16 + 0 / 60 + 0 / 3600),
                   (20 + 0 / 60 + 0 / 3600),
                   ])
dec_arr = np.array([    # Dec in degrees
                    # (-20 + 0 / 60 + 0 / 3600),        
                    (0 + 0 / 60 + 0 / 3600),
                    (20 + 0 / 60 + 0 / 3600),
                    (40 + 0 / 60 + 0 / 3600),
                    (60 + 0 / 60 + 0 / 3600),
                    (80 + 0 / 60 + 0 / 3600),
                    ])
n_ant_arr = [10, 8, 6]    # number of antennas observing
gain_noise_arr = ['A', 'B', 'C']    # different data quality, 'A' is thermal
    # noise only, 'B' is moderate and 'C' more severe gain errors

### Simulate all cases with run_all() ###
simlens.run_all(
    F_A_arr,
    flux_ratio_arr,
    R_B_arr,
    angles,
    scan_length_arr,
    ra_arr,
    dec_arr,
    n_ant_arr,
    gain_noise_arr,
    parallelize=False,
    overwrite=False,
    files_exist='files_exist.txt'    # can provide file here, otherwise comment
    )

### Simulate single case... ###

# ...with simulate_case()... #
# simlens.simulate_case(
    # (F_A_arr[0],
     # flux_ratio_arr[0]*F_A_arr[0],
     # simlens.FWHM_A,
     # simlens.FWHM_A*np.sqrt(flux_ratio_arr[0]),
     # R_B_arr[0],
     # angles[0],
     # scan_length_arr[0],
     # ra_arr[0],
     # dec_arr[0],
     # n_ant_arr[0],
     # gain_noise_arr[0])
    # )

# ...and with run_all() #
# simlens.run_all(
    # F_A_arr[0],
    # flux_ratio_arr[0],
    # R_B_arr[0],
    # angles[0],
    # scan_length_arr[0],
    # ra_arr[0],
    # dec_arr[0],
    # n_ant_arr[0],
    # gain_noise_arr[0],
    # parallelize=False
    # )
