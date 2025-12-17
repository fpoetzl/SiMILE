from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.io import ascii
from astropy.io import fits
from astropy.time import Time
import datetime
import ehtim as eh
import json
import numpy as np
from multiprocessing import Pool
import os
import psutil
import time
import threading
import sys
# from multiprocessing import set_start_method
# set_start_method("spawn")
# from multiprocessing import get_context

print(f'numpy version: {np.__version__}')
# print(eh.__file__)
# input()



def get_LST_max(obs_date_string, centre_lat, centre_lon, ra, dec,
                array_lats, array_lons, elevmin, elevmax, scan_length):
    '''
    # Purpose: helper function to get the local sidereal time at the array
    centre at the time of maximum source elevation.
    
    # Args:
        obs_date_string (str): String containing the observation date. Format
          needs to be 'YYYY-MM-DD HH:MM:SS'. This is needed to generate the
          test utc times to find the time of maximum elevation.
        centre_lat (float): geographical latitude of array centre in degrees.
        centre_lon (float): geographical longitude of array centre in degrees.
        ra (float): Right ascension of the observation in hours.
        dec (float): Declination of the observation in degrees.
        array_lats (list): array latitudes.
        elevmin (float): minimum telescope elevation.
        elevmax (float): maximum telescope elevation.
        scan_length (float): scan length in minutes.
    
    # Returns:
        LST_max (float): local sidereal time at array centre with latitude and
          longitude given by centre_lat, centre_lon at maximum source elevation
          in hours.
    '''
    if elevmax != 90:
        elevmax = elevmax - 0.25    # safety for slightly uncertain station
        # coordinates causing missing data again when station elevation > elexmax
    
    check_int = 1   # minutes
    
    from datetime import datetime, timedelta
    start_time = datetime.strptime(obs_date_string, '%Y-%m-%d %H:%M:%S')
    # test utc_times in steps of 5 minutes
    n_steps = int(24 * 60 / check_int)
    utc_times = [ (start_time + timedelta(minutes=check_int*i)).strftime('%Y-%m-%d %H:%M:%S')
                  for i in range(n_steps) ]  # 24 hours * 6 intervals per hour
    # test utc_times in steps of one hour
    # utc_times = [obs_date_string[:11]+hour+obs_date_string[13:] for hour in 
                 # ['01','02','03','04','05','06','07','08','09','10','11','12',
                 #  '13','14','15','16','17','18','19','20','21','22','23']]
    
    station_LSTs = np.empty((len(utc_times), len(array_lats)), dtype=object)

    for j, (lat, lon) in enumerate(zip(array_lats, array_lons)):
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)
        t_loc = Time(utc_times, scale='utc', location=loc)
        lsts = t_loc.sidereal_time('mean')        # Angle array
        for i in range(len(utc_times)):
            station_LSTs[i,j] = lsts[i]
    
    # determine LST at array centre
    obs_loc = EarthLocation(lat=centre_lat*u.deg, lon=centre_lon*u.deg)
    obs_times = Time(utc_times, scale='utc', location=obs_loc)
    LSTs = obs_times.sidereal_time('mean')
    
    # determine hour angle to calculate source elevation
    hour_angles = LSTs - ra*u.hourangle

    alt_centre = np.arcsin(
          np.sin(centre_lat*np.pi/180)*np.sin(dec*np.pi/180)
        + np.cos(centre_lat*np.pi/180)*np.cos(dec*np.pi/180)
        * np.cos(hour_angles.deg*np.pi/180) 
        )    # in radians
    alt_centre_deg = alt_centre*180/np.pi    # in degrees
    
    
    
    # ### OLD VERSION ###
    # # calculate LST at maximum elevation
    # hs = LSTs.hms[0]
    # ms = hour_angles.hms[1]
    # ss = hour_angles.hms[2]
    # LSTs_h = np.array([h+m/60+s/3600 for (h,m,s) in zip(hs,ms,ss)])
    
    # LST_max = LSTs_h[np.where(alt_centre==max(alt_centre))]
    
    # h = int(LST_max[0])
    # m = int((LST_max[0] - h)*60)
    # s = (LST_max[0] - h - m/60)*3600
    # print(f'!OLD!: LST for maximum source elevation: {h:02d}:{m:02d}:{s:05.2f}')
    
    
    
    # Calculate all candidate times where every telescope is below elevmax
    candidate_indxs = []
    alts_tel_list = []
    for i, alt0 in enumerate(alt_centre):
        ha = np.deg2rad(hour_angles.deg[i])
        has = [np.deg2rad((lst - ra*u.hourangle).deg) for lst in station_LSTs[i,:]]
        alts_tel = [
            np.rad2deg(
                np.arcsin(
                    np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(dec)) + 
                    np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(dec))*np.cos(has[k])
                    )
                )
            for k, lat in enumerate(array_lats)
            ]
        alts_tel_list.append(alts_tel)
        if np.all(np.array(alts_tel) <= elevmax) and np.all(np.array(alts_tel) >= elevmin):
            candidate_indxs.append(i)
    
    if not candidate_indxs:
        raise ValueError("No time found where all telescopes stay below elevmax")

    # Now check scan window requirement
    halfwin = int(np.round((scan_length/check_int)/2))  # half window in steps of check_int
    valid_idx = []
    for idx in candidate_indxs:
        # Define bounds for checking
        lo = max(0, idx-halfwin)
        hi = min(len(alt_centre_deg) - 1, idx + halfwin)
        ok = True
        for j in range(lo, hi+1):
            has = [np.deg2rad((lst - ra*u.hourangle).deg) for lst in station_LSTs[j,:]]
            alts_tel = [
                np.rad2deg(
                    np.arcsin(
                        np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(dec)) + 
                        np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(dec))*np.cos(has[k])
                        )
                    )
                for k, lat in enumerate(array_lats)
                ]
            if np.any(np.array(alts_tel) > elevmax):
                ok = False
                break
        if ok == True:
            valid_idx.append(idx)
    
    if not valid_idx:
        raise ValueError("No time window of length scan_length found where all telescopes stay below elevmax")

    # Pick the index with maximum centre altitude among valid times
    idx = max(valid_idx, key=lambda i: alt_centre_deg[i])
    
    # Convert LST to H:M:S
    LSTs_h = LSTs.hour
    h = int(LSTs_h[idx])
    m = int((LSTs_h[idx] - h)*60)
    s = (LSTs_h[idx] - h - m/60)*3600
    
    print(f'LST for maximum source elevation: {h:02d}:{m:02d}:{s:05.2f}')
    LST_max = LSTs_h[idx]
    
    # for testing: manual input of starting time
    # start_time = datetime.strptime('2024-09-24 10:00:00', '%Y-%m-%d %H:%M:%S')
    # utc_time = Time(start_time, scale='utc')
    # obs_time = Time(utc_time, scale='utc', location=obs_loc)
    # LST = obs_time.sidereal_time('mean')
    # LST_h = LST.hour
    # _h = int(LST_h)
    # _m = int((LST_h - _h)*60)
    # _s = (LST_h - _h - _m/60)*3600
    # print(f'LST TEST: {_h:02d}:{_m:02d}:{_s:05.2f}')
    # LST_max = LST_h
    
    return LST_max



def get_clean_beam(npix, fov, ra, dec, freq, mjd, F_A, X_A, Y_A, FWHM_A, array,
                   tint, tadv, bw, tstart, tstop, timetype, elevmin, elevmax,
                   ampcal, phasecal, seed, output_dir, uvfits_filename):
    '''
    # Purpose: small helper function to determine the clean beam from an
    observation.
    
    # Args:
        npix (int): Number of pixels to be used for the mock image.
        fov (int): field of view in mas to be used for the mock image.
        ra (float): Right ascension of the observation in hours.
        dec (float): Declination of the observation in degrees.
        freq (float): observing frequency in Hz.
        mjd (float): modified Julian Date for the observation.
        F_A (float): flux density of the primary Gaussian component in mJy.
        X_A (float): right ascension of the primary Gaussian component in mas.
        Y_A (float): declination of the primary Gaussian component in mas.
        FWHM_A (float): FWHM size of the primary Gaussian component in mas.
        array (array Object): array object after reading specific file into
          eht-imaging.
        tint (float): data Integration time in seconds.
        tadv (float): 
        bw (float): observing bandwidth in Hz.
        tstart (float): start time of scan in hours.
        tstop (float): end time of scan in hours.
        timetype (str): determines how to interpret starting and stopping time.
        elevmin (float): minimum telescope elevation in degrees.
        elevmax (float): maximum telescope elevation in degrees.
        ampcal (bool): determines if dataset will have amplitude gain noise
          applied.
        phasecal (bool): determines if dataset will have phase gain noise
          applied.
        seed (int): seed for numpy.random.
        output_dir (str): output folder to be used for the scratch file.
        uvfits_filename (str): file name to be used for the scratch file.
    
    # Returns:
        clean_beam (numpy.ndarray): list containing clean beam major axis and
          minor axis in mas and the position angle in degrees defined positive
          North to East.
    '''
    ### Create a new image ###
    img = eh.image.make_empty(npix, fov*1e3*eh.RADPERUAS, ra, dec, freq)
    img.source = 'SMILE TEST'
    img.mjd = mjd
    
    ### Add primary component (A) to the image ###
    if F_A != 0:
        img = img.add_gauss(F_A*1e-3, [FWHM_A*1e3*eh.RADPERUAS,
                                       FWHM_A*1e3*eh.RADPERUAS,
                                       0,
                                       X_A*1e3*eh.RADPERUAS,
                                       Y_A*1e3*eh.RADPERUAS])
    
    ### Create dummy uvfits file ###
    obs_sim_test = img.observe(
        array,
        tint=tint,
        tadv=tadv,
        bw=bw,
        tstart=tstart,
        tstop=tstop,
        mjd=mjd,
        timetype=timetype,
        elevmin=elevmin,
        elevmax=elevmax,
        ampcal=ampcal,
        phasecal=phasecal,
        seed=seed,
        ttype='direct'
        )
    
    ### Save uvfits file ###
    # obs_sim_test.save_uvfits(output_dir + uvfits_filename[:-7] + '_test.uvfits')
    # obs_sim_test = eh.obsdata.load_uvfits(output_dir + uvfits_filename[:-7] + '_test.uvfits')
    
    ### Determine clean beam parameters by fitting a Gaussian to the clean beam ###
    clean_beam = obs_sim_test.cleanbeam(fov=fov*eh.RADPERUAS*1000, npix=npix).fit_gauss(units='natural')
    
    clean_beam[2] = clean_beam[2] - 180    # consistent with angle defined positive North to East
    
    return clean_beam



def monitor_resources():
    import getpass
    user = getpass.getuser()
    prev_io = {}
    prev_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        total_cpu = 0.0
        total_ram_gb = 0.0
        total_ram_percent = 0.0
        total_read = 0
        total_write = 0

        # Average CPU usage across all cores
        per_core_cpu = psutil.cpu_percent(interval=None, percpu=True)
        avg_cpu_all_cores = np.mean(per_core_cpu)

        print("\nResource usage for Python processes and threads:")
        print(f"{'PID':>6} {'Name':20} {'CPU%':>6} {'RAM GB':>8} {'RAM%':>6} {'Read MB/s':>10} {'Write MB/s':>10} {'Threads (CPU time s)':>20}")
        print("-"*110)

        for proc in psutil.process_iter(['pid', 'name', 'username']):
            try:
                # Filter for your username and Python processes
                if proc.info['username'] != user or 'python' not in proc.info['name'].lower():
                    continue

                # Process CPU and memory
                cpu = proc.cpu_percent(interval=None)
                mem_info = proc.memory_info()
                ram_gb = mem_info.rss / (1024**3)
                ram_percent = proc.memory_percent()

                # Disk I/O
                io_counters = proc.io_counters()
                pid = proc.pid
                if pid in prev_io:
                    read_bytes = (io_counters.read_bytes - prev_io[pid][0]) / dt
                    write_bytes = (io_counters.write_bytes - prev_io[pid][1]) / dt
                else:
                    read_bytes = 0.0
                    write_bytes = 0.0
                prev_io[pid] = (io_counters.read_bytes, io_counters.write_bytes)

                # Thread CPU times
                thread_info = [f"{th.id}:{th.user_time + th.system_time:.1f}" for th in proc.threads()]

                total_cpu += cpu
                total_ram_gb += ram_gb
                total_ram_percent += ram_percent
                total_read += read_bytes
                total_write += write_bytes

                threads_str = ','.join(thread_info)
                print(f"{pid:6} {proc.info['name'][:20]:20} {cpu:6.1f} {ram_gb:8.2f} {ram_percent:6.1f} {read_bytes/1e6:10.2f} {write_bytes/1e6:10.2f} {threads_str:>20}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        print("-"*110)
        print(f"TOTAL{'':22} {total_cpu:6.1f} {total_ram_gb:8.2f} {total_ram_percent:6.1f} {total_read/1e6:10.2f} {total_write/1e6:10.2f}")
        print(f"Average CPU usage across all cores: {avg_cpu_all_cores:.1f}%")
        time.sleep(3)

# def monitor_resources():
    # while True:
        
        # # Get CPU usage
        # cpu_usage = psutil.cpu_percent(interval=1)
        
        # # Get RAM usage
        # ram = psutil.virtual_memory()
        # ram_usage = ram.percent
        # ram_usage_gb = ram.used / (1024**3)    # Convert bytes to gigabytes
        
        # # Print the results
        # print(f"CPU Usage: {cpu_usage}%")
        # print(f"RAM Usage: {round(ram_usage_gb,1)} GB / {ram_usage} %")
        
        # # Wait for interval seconds before the next check
        # time.sleep(1)



def write_difmap_mod(F_A, F_B, X_A, Y_A, X_B, Y_B, FWHM_A, FWHM_B, freq, out,
                     names=False):
    '''
    # Purpose: small helper function to print a difmap-readable modelfit file.
    
    # Args:
        F_A (float): Flux density of the primary Gaussian component in mJy.
        F_B (float): Flux density of the secondary Gaussian component in mJy.
        X_A (float): Right ascension of the primary Gaussian component in mas.
        Y_A (float): Declination of the primary Gaussian component in mas.
        X_B (float): Right ascension of the secondary Gaussian component in mas.
        Y_B (float): Declination of the secondary Gaussian component in mas.
        FWHM_A (float): FWHM size of the primary Gaussian component in mas.
        FWHM_B (float): FWHM size of the primary Gaussian component in mas.
        freq (float): observing frequency in Hz.
        out (str): output file name.
    
    # Returns:
        Creates two files, a .mfit file readable by difmap and a .mfitid file
        where the last column has entries according to the component
        designation (not readable by difmap but good for other purposes).
    '''
    F = [F_A*1e-3, F_B*1e-3]
    r = [np.sqrt(X_A**2 + Y_A**2), np.sqrt(X_B**2 + Y_B**2)]
    theta = [np.arctan2(X_A, Y_A)*180/np.pi, np.arctan2(X_B, Y_B)*180/np.pi]
    FWHM = [FWHM_A, FWHM_B]
    ax_ratio = [1.0, 1.0]
    phi = [1.0, 1.0]
    T = [1, 1]
    freqs = [freq, freq]
    if names == True:
        names = ['A', 'B']
        form_str = '%s'
    else:
        names = [0, 0]
        form_str = '%d'
    
    ascii.write([F, r, theta, FWHM, ax_ratio, phi, T, freqs, names],
                out,
                names=['Flux (Jy)',
                       'Radius (mas)',
                       'Theta (deg)',
                       'Major FWHM (mas)',
                       'Axial ratio',
                       'Phi (deg)',
                       'T',
                       'Freq (Hz)',
                       'SpecIndex',
                       ],
                formats={'Flux (Jy)':'.6f',
                         'Radius (mas)':'.6f',
                         'Theta (deg)':'.4f',
                         'Major FWHM (mas)':'.6f',
                         'Axial ratio':'.5f',
                         'Phi (deg)':'.4f',
                         'T':'%d',
                         'Freq (Hz)':'.5e',
                         'SpecIndex':form_str,               
                         },
                overwrite=True, format='fixed_width', delimiter=' ')
    
    with open(out, 'r') as file:
        lines = file.readlines()
    # Add an exclamation mark at the beginning for difmap readability
    for i, line in enumerate(lines):
        if i == 0:
            lines[i] = '!' + lines[0]
        else:
            lines[i] = ' ' + lines[i]
    with open(out, 'w') as file:
        file.writelines(lines)



def detect_pol(uvfits_file, difmap_path='/usr/local/difmap/difmap'):
    
    # This is a temporary workaround because ideally, one would not need to
    # use difmap here. But reading uvfits with astropy is a nightmare...
    import pexpect
    from pexpect import replwrap
    
    # Add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ':{0}'.format(difmap_path) 
    
    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False)
    child.expect_exact('0>', None, 2)

    def send_difmap_command(command,prompt='0>'):
        child.sendline(command)
        child.expect_exact(prompt, None, 2)
    
    send_difmap_command('observe ' + uvfits_file)
    
    if 'RR' and 'LL' in child.before:
        print('Dual pol. detected in input uvfits file.')
        os.system('rm -rf difmap.log*')
        return 'RL'
    elif 'RR' in child.before:
        print('Single pol. (R) detected in input uvfits file.')
        os.system('rm -rf difmap.log*')
        return 'R'
    elif 'LL' in child.before:
        print('Single pol. (L) detected in input uvfits file.')
        os.system('rm -rf difmap.log*')
        return 'L'
    else:
        print('Could not detect pol. in input uvfits file.')
        os.system('rm -rf difmap.log*')
        return None
    
    # This does not work (yet)!
    # hdul = fits.open(uvfits_file)
    # for i, hdu in enumerate(hdul):
        # print(i, hdu.name, type(hdu.data))
    # print(hdul[0].header.get('GROUPS'))
    # data = hdul[0].data['DATA']  # Access the visibility data
    # datatest = hdul[0].data.data[0, 0, 0, 0, 0, :, :]
    # # print(np.shape(data))
    # # print(data[0])
    # print(datatest)
    
    # print(hdul[0].data.parnames)
    
    # print(hdul[0].header['BITPIX'])
    # print(hdul[0].header.get('BSCALE'))
    # print(hdul[0].header.get('BZERO'))
    
    # input()
    # hdul.close()

    # # Extract visibility part (shape: nvis, 1, 1, 1, 1, npol = 4, Re x Im x Weight = 3)
    # vis_array = data.squeeze()  # Remove singular dimensions (shape becomes nvis, 4, 3)
    
    # nonzero_rows_list = []
    
    # # Assume only one visibility for simplicity, otherwise loop
    # for n_vis in range(len(vis_array)):
        # vis = vis_array[n_vis]
    
        # row_sums = np.linalg.norm(vis, axis=1)  # Compute magnitude per row
        # # each entry now stands for 1 pol: rr, ll, rl, lr
        
        # # Check which row is non-zero
        # nonzero_rows = np.where(row_sums > 0)[0]
        # nonzero_rows_list.append(nonzero_rows)
        # if n_vis == 100:
            # print(vis)
            # print(row_sums)
            # print(nonzero_rows)
        # if len(nonzero_rows) > 0:
            # print(vis)
    
    # if all(len(sublist) == 0 for sublist in nonzero_rows_list):
        # print('No visibilities found at all!')
        # return None
    # elif all(len(sublist) == 1 for sublist in nonzero_rows_list):
        # if all(sublist[0] == 0 for sublist in nonzero_rows_list):
            # return 'r'
        # elif all(sublist[0] == 1 for sublist in nonzero_rows_list):
            # return 'l'
    # else:
        # return 'rl'



def remove_pol(uvfits_file, pol='R'):
    # TODO: this is a workaround that just copies one polarization to the other
    # and adjusts the weights so they are the same when averaging R and L pol.
    # for Stokes I when imaging in difmap. It would be best to remove the
    # polarization axis completely; but this does not work yet.    

    # input_file = uvfits_file
    # output_file = "./output_copy.uvfits"
    # print(uvfits_file)
    # hdul = fits.open(input_file)
    # data = hdul[0].data['DATA']  # Access the visibility data
    # hdul.close()
    # print(data.shape)
    # print(data[0])
    
    # # Step 1: Open the original file
    # with fits.open(input_file, mode='readonly') as hdul:
        # # Step 2: Copy the HDU list
        # hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])

    # # Step 3: Write out a new UVFITS file (verifies structure automatically)
    # hdul_copy.writeto(output_file, overwrite=True, output_verify='exception')
    
    with fits.open(uvfits_file, mode='update') as hdul:
        data = hdul[0].data
        vis_array = data['DATA']

        if pol == 'R':
            vis_array[:, :, :, :, :, 1, :] = vis_array[:, :, :, :, :, 0, :]
        elif pol == 'L':
            vis_array[:, :, :, :, :, 0, :] = vis_array[:, :, :, :, :, 1, :]
        vis_array[:, :, :, :, :, :, 2] *= 0.5

        new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
        new_hdul.writeto(uvfits_file, overwrite=True, output_verify='exception')
    
    # hdul = fits.open(output_file)
    # data = hdul[0].data['DATA']  # Access the visibility data
    # hdul.close()
    # print(data.shape)
    # print(data[0])



class simlens:
    '''
    simlens is an object that takes some standard observational parameters to
    define an observation simulation. Then some functions are defined on it to
    create the simulated data.
    '''
    def __init__(self):
        # Parameters for the simulation that are not varied and are set to
            # some standard values
        self.case_name = 'Test'
        self._output_dir_ = './sim_data/'
        if not os.path.exists(self._output_dir_):
            os.mkdir(self._output_dir_)
        self.array_file = './SMILE_array.txt'
        self.source_name = 'SMILE TEST'
        self.tint = 10    # in seconds
        self.tadv = self.tint
        self.bw = 2.5e7 * 8    # in Hz, 8 IF with 25 MHz bandwidth each
        self.obs_date_string = '2024-09-24 00:00:00'
        self.obs_date = Time(self.obs_date_string)
        self.mjd = self.obs_date.mjd
        self.tstart = None
        self.timetype = 'UTC'
        self.elevmin = 10    # in degrees
        self.elevmax = 85    # in degrees
        self.seed = 17
        self.ttype = 'direct'
        
        # Parameters for image creation that will be 'observed'
        self.freq = 4.9e9    # in Hz
        self.npix = 1024
        self.fov = 320    # in mas
        
        # Parameters for primary Gaussian component (fixed)
        self.FWHM_A = 1.0    # in mas
        self.X_A = 0.0    # in mas
        self.Y_A = 0.0    # in mas
        
        # Telescope positions (fixed for now, will include other arrays later)
        self.array = 'VLBA'
        if self.array == 'VLBA':
            # Define station coordinates #
            self.array_lats = [
                48.13117,    # BR
                30.63521,    # FD
                42.93362,    # HN
                31.95625,    # KP
                35.77529,    # LA
                19.80159,    # MK
                41.77165,    # NL
                37.23176,    # OV
                34.30107,    # PT
                17.75652,    # SC
                ]
            self.array_lons = [
                -119.68325,    # BR
                -103.94483,    # FD
                 -71.98681,    # HN
                -111.61236,    # KP
                -106.24559,    # LA
                -155.45581,    # MK
                 -91.57413,    # NL
                -118.27714,    # OV
                -108.11912,    # PT
                 -64.58376,    # SC
                ]
            # Define station SEFDs for R and L pol. #
            sefds_dict = {
                '1.7GHz': np.tile([314.,314.], (10, 1)),
                '2.2GHz': np.tile([347.,347.], (10, 1)),
                '5.0GHz': np.tile([210.,210.], (10, 1)),
                '8.0GHz': np.tile([327.,327.], (10, 1)),
                '15.0GHz': np.tile([543.,543.], (10, 1)),
                '22.0GHz': np.tile([640.,640.], (10, 1)),
                }
            sefds_default_dict = {
                '1.7GHz':[314.,314.],
                '2.2GHz':[347.,347.],
                '5.0GHz':[210.,210.],
                '8.0GHz':[327.,327.],
                '15.0GHz':[543.,543.],
                '22.0GHz':[640.,640.],
                }
            
            # Choose SEFDs closest to chosen observing frequency #
            freqs = []
            for key in sefds_dict.keys():
                freqs.append(float(key.rstrip('GHz'))*1e9)
            freqs = np.array(freqs)
            idx = np.argmin(np.abs(freqs - self.freq))
            freq = round(freqs[idx]/1e9,1)
            self.sefds = sefds_dict[str(freq)+'GHz']
            self.sefds_default = sefds_default_dict[str(freq)+'GHz']
            
            # Load array file with SEFDs if applicable #
            if os.path.exists('./SMILE_array_' + str(freq) + 'GHz.txt'):
                self.array_file = './SMILE_array_' + str(freq) + 'GHz.txt'
            else:
                print('Could not find frequency-specific array file, trying default one.')
                if os.path.exists('./SMILE_array.txt'):
                    self.array_file = './SMILE_array.txt'
                else:
                    print('! Warning: Could not find default array file!'
                          ' Please provide by setting the path manually.')
            # self.sefds_default = [
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # [310.,310.],
                # ]
    
    @property
    def output_dir(self):
        return self._output_dir_
    
    @output_dir.setter
    def output_dir(self, value):
        self._output_dir_ = value    # Internal storage
        if not os.path.exists(self._output_dir_):
            os.mkdir(self._output_dir_)
    
    def simulate_case(self, params, use_dataset=''):
        '''
        # Purpose: Function to create a single set of simulated data for an
          observation.
        
        # Args:
        params (tuple): Set of parameters defined below.
            F_A (float): Flux density of the primary Gaussian component in mJy.
            F_B (float): Flux density of the secondary Gaussian component in mJy.
            FWHM_A (float): FWHM size of the primary Gaussian component in mas.
            FWHM_B (float): FWHM size of the primary Gaussian component in mas.
            R_B (float): The radial distance of the secondary Gaussian
              component to the primary given in mas.
            angle (str): A string input determining where the secondary
              component will be placed w.r.t. the clean beam. Can be 'maj' or
              'min' to be placed along the clean beam major or minor axis,
              respectively.
            scan_length (float): Length of the observing scan in minutes.
            ra (float): Right ascension of the observation in hours.
            dec (float): Declination of the observation in degrees.
            n_ant (int): Number of antennas to be observing. If fewer than the
              amount of antennas in the array file are defined, antennas are
              removed at random from the array.
            gain_noise (str): A string determining the data quality to be
              desired. More gain noise is added for lower quality cases. Now
              accepts 'A', 'B', and 'C', where 'A' applies only thermal noise,
              and 'B' and 'C' progressively more station gain errors in both
              amplitude and phase. See below for more details.
        use_dataset (str): full path to uvfits file being used to generate simulated
          data based on the observational parameters of that dataset.
        
        # Returns:
            Creates five files: the output .uvfits file, a scratch _test.uvfits
            file, a .fits file with the Gaussian models, and two files
            containing the ground-truth Gaussian modelfit parameters (.mfit
            readable by difmap and a .mfitid file with the component names).
        
        # Additional information:
        Descriptions for noise modeling in simulated data in eht-imaging:
            add_th_noise (bool) [default True] – if True, baseline-dependent thermal noise is added
            ampcal (bool) [default True] – if False, time-dependent gaussian errors are added to station gains
            phasecal (bool) [default True] – if False, time-dependent station-based random phases are added
            stabilize_scan_phase (bool) [default False] – if True, random phase errors are constant over scans
            stabilize_scan_amp (bool) [default False] – if True, random amplitude errors are constant over scans
            gainp (float) [default 0.1] – the fractional std. dev. of the random error on the gains or a dict giving one std. dev. per site
            gain_offset (float) [default 0.1] – the base gain offset at all sites, or a dict giving one gain offset per site
            phase_std (float) [default -1] – std. dev. of LCP phase, or a dict giving one std. dev. per site a negative value samples from uniform
            sigmat (float) [default None] – temporal std for a Gaussian Process used to generate gains. If sigmat=None then an iid gain noise is applied.
            phasesigmat (float) [default None] – temporal std for a Gaussian Process used to generate phases. If phasesigmat=None then an iid gain noise is applied.
        
        Some informative datasets:
            'A' quality: 0902+210(X) - 94 mJy, DR 179, amp scatter 14.8 %, phase rms 10.2 deg, 10 antennas              , BW 16*32 MHz, RR, t_int 6 min   (3 scans)
            'B' quality: 1743+169(X) - 87 mJy, DR 39 , amp scatter 16.2 %, phase rms  9.8 deg,  9 antennas (no MK)      , BW  8*32 MHz, RR, t_int 1 min   (1 scan)
            'C' quality: 0824+399(X) - 71 mJy, DR 21 , amp scatter 31.3 %, phase rms 18.8 deg,  7 antennas (no FD,HN,MK), BW  8*32 MHz, RR, t_int 1.3 min (1 scan)
            'D' quality: 1440+232(C) - 53 mJy, DR 17 , amp scatter 21.7 %, phase rms 15.0 deg,  8 antennas (no FD,HN)   , BW  8*32 MHz, RR, t_int 1 min   (1 scan)
            'E' quality: 1452+540(C) - 60 mJy, DR 12 , amp scatter 39.2 %, phase rms 37.5 deg,  8 antennas (no HN,SC)   , BW  8*32 MHz, RR, t_int 1 min   (1 scan)
        '''
        
        try:
            (F_A, F_B, FWHM_A, FWHM_B, R_B, angle, scan_length, ra, dec, n_ant, gain_noise) = params
        except ValueError:
            (F_A, F_B, FWHM_A, FWHM_B, R_B, angle, gain_noise, use_dataset) = params
        
        # ### Construct filenames ###
        # file_suffix = f"_FA{round(F_A,1)}_FWHMA{round(FWHM_A,1)}_FB{round(F_B,1)}_FWHMB{round(FWHM_B,1)}_RB{R_B}_b{angle}_scan{scan_length}_ra{round(ra,1)}_dec{round(dec,1)}_Nant{n_ant}_gain{gain_noise}"
        # fits_filename = self.case_name + file_suffix + '.fits'
        # uvfits_filename = self.case_name + file_suffix + '.uvfits'
        # print(f'uvfits_filename: {uvfits_filename}')
        # uvfits_nonoise_filename = self.case_name + file_suffix + '_NO_NOISE.uvfits'
        # mod_filename = self.case_name + file_suffix + '.mfit'
        # log_filename = self.case_name + file_suffix + '_sim.log'
        
        if not os.path.exists('./' + self.output_dir):
            os.mkdir('./' + self.output_dir)
        
        old_sys = sys.stdout
        
        if use_dataset == '':
            print('\n')
            print('### Creating simulated data set with parameters: ###')
            print('F_A, F_B, FWHM_A, FWHM_B, R_B, angle, scan_length, ra, dec, n_ant, gain_noise')
            print(params)
            
            ### Construct filenames ###
            file_suffix = f"_FA{round(F_A,1)}_FWHMA{round(FWHM_A,1)}_FB{round(F_B,1)}_FWHMB{round(FWHM_B,1)}_RB{R_B}_b{angle}_scan{scan_length}_ra{round(ra,1)}_dec{round(dec,1)}_Nant{n_ant}_gain{gain_noise}"
            fits_filename = self.case_name + file_suffix + '.fits'
            uvfits_filename = self.case_name + file_suffix + '.uvfits'
            print(f'uvfits_filename: {uvfits_filename}')
            uvfits_nonoise_filename = self.case_name + file_suffix + '_NO_NOISE.uvfits'
            mod_filename = self.case_name + file_suffix + '.mfit'
            log_filename = self.case_name + file_suffix + '_sim.log'
            
            sys.stdout = open(self.output_dir + log_filename, 'w+')
            
            print('\n')
            print('# Load array #')
            ### Load array ###
            array = eh.array.load_txt(self.array_file)
            sites = [site for site in array.tkey.keys()]
            if n_ant < len(sites):
                np.random.seed(self.seed)
                # remove_sites = np.random.randint(0, len(sites), len(sites)-n_ant)
                remove_sites = np.random.choice(range(len(sites)), len(sites)-n_ant,
                                                replace=False)
                for i, rs in enumerate(remove_sites):
                    # print(sites[remove_sites[i]])
                    array = array.remove_site(sites[remove_sites[i]])
            print(f'Stations for observation: {array.tkey.keys()}')
            
            ### Add gain noise specifics ###
            if gain_noise == 'A':
                jones = False
                ampcal = True
                phasecal = True
                gainp = 0.1
                gain_offset = 0.1
                phase_std = -1
            elif gain_noise == 'B':
                jones = True
                ampcal = False
                phasecal = True
                gainp = 0.1
                gain_offset = 0.1
                phase_std = 10*np.pi/180
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 10*np.pi/180
                # phase_std = phase_std_dict
            elif gain_noise == 'C':
                jones = True
                ampcal = False
                phasecal = False
                gainp = 0.2
                gain_offset = 0.2
                phase_std = 20*np.pi/180
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 20*np.pi/180
                # phase_std = phase_std_dict
            elif gain_noise == 'D':
                jones = True
                ampcal = False
                phasecal = False
                gainp = 0.2
                gain_offset = 0.2
                phase_std = -1
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 20*np.pi/180
                # phase_std = phase_std_dict
            elif type(gain_noise) == float:
                jones = True
                ampcal = False
                phasecal = False
                gainp = gain_noise
                gain_offset = gain_noise
                phase_std = gain_noise*100*np.pi/180
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 20*np.pi/180
                # phase_std = phase_std_dict
            # print('Ampcal', ampcal)
            # print('Phasecal', phasecal)
            print('# Calculate optimal observation start time #')
            ### Choose appropriate observation start time when the source is close to
                # the elevation peak at Pie Town
            # lat_PT = (34+20/60+24/3600)    # coordinates of PT
            # lon_PT = (-108+10/60+15/3600)
            lat_PT = self.array_lats[8]
            lon_PT = self.array_lons[8]
            utcoffset_PT = -7    # PT DST, change to -6 for standard time
            
            LST_max = get_LST_max(obs_date_string=self.obs_date_string,
                                  centre_lat=lat_PT, centre_lon=lon_PT,
                                  ra=ra, dec=dec, array_lats=self.array_lats,
                                  array_lons=self.array_lons,
                                  elevmin=self.elevmin, elevmax=self.elevmax,
                                  scan_length=scan_length)
                                  # LST at PT at max. source elev. while no antenna
                                  # above max telescope elev.
            
            if self.tstart == None:
                tstart = (LST_max - utcoffset_PT - scan_length/60./2.)%24
            else:
                tstart = self.tstart
            h = int(tstart)
            m = int((tstart - h)*60)
            s = (tstart - h - m/60)*3600
            print(f'Observation start time: {h:02d}:{m:02d}:{s:05.2f} UT')
            tstop = tstart + scan_length/60.
            
            print('# Determine position angle of the clean beam #')
            ### Determine position angle of the clean beam for later use ###
            clean_beam = get_clean_beam(npix=128, fov=50, ra=ra, dec=dec,
                                        freq=self.freq, mjd=self.mjd, F_A=F_A,
                                        X_A=self.X_A, Y_A=self.Y_A, FWHM_A=FWHM_A,
                                        array=array, tint=self.tint,
                                        tadv=self.tadv, bw=self.bw,
                                        tstart=tstart, tstop=tstop,
                                        timetype=self.timetype,
                                        elevmin=self.elevmin, elevmax=self.elevmax,
                                        ampcal=ampcal, phasecal=phasecal,
                                        seed=self.seed, output_dir=self.output_dir,
                                        uvfits_filename=uvfits_filename)
            
            PA_B = clean_beam[2]    # clean beam position angle
            
            if angle == 'maj':
                X_B = -R_B*np.cos((90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+PA_B)*np.pi/180.)
            elif angle == 'min':
                X_B = -R_B*np.cos((90+90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+90+PA_B)*np.pi/180.)
            elif type(angle) == float or type(angle) == int:
                PA_B = angle
                X_B = -R_B*np.cos((90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+PA_B)*np.pi/180.)
            else:
                print('Warning: no valid position angle for component B provided, set to 0.')
                PA_B = 0
                X_B = -R_B*np.cos((90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+PA_B)*np.pi/180.)
            
            print('# Create ground truth image #')
            ### Create a new image ###
            img = eh.image.make_empty(self.npix, self.fov*1e3*eh.RADPERUAS,
                                      ra, dec, self.freq)
            img.source = self.source_name
            img.mjd = self.mjd
            
            ### Add primary component (A) to the image ###
            img = img.add_gauss(F_A*1e-3, [FWHM_A*1e3*eh.RADPERUAS,
                                           FWHM_A*1e3*eh.RADPERUAS,
                                           0,
                                           self.X_A*1e3*eh.RADPERUAS,
                                           self.Y_A*1e3*eh.RADPERUAS])
            
            ### Add secondary component (B) to the image ###
            if F_B > 1e-3:
                # add only if sufficiently bright (more than 1 microJansky)
                # needed because ehtim complains when flux is too low...
                # TO-DO: clean up this implementation
                img = img.add_gauss(F_B*1e-3, [FWHM_B*1e3*eh.RADPERUAS,
                                               FWHM_B*1e3*eh.RADPERUAS,
                                               0,
                                               X_B*1e3*eh.RADPERUAS,
                                               Y_B*1e3*eh.RADPERUAS])
            
            print('# Create simulated observation #')
            ### Simulate observation ###
            obs_sim = img.observe(
                array,
                tint=self.tint,
                tadv=self.tadv,
                bw=self.bw,
                tstart=tstart,
                tstop=tstop,
                mjd=self.mjd,
                timetype=self.timetype,
                elevmin=self.elevmin,
                elevmax=self.elevmax,
                add_th_noise=True,
                jones=jones,
                ampcal=ampcal,
                phasecal=phasecal,
                gainp=gainp,
                gain_offset=gain_offset,
                phase_std=phase_std,
                seed=self.seed,
                ttype=self.ttype
                )

            ### Simulate observations without noise ###
            # obs_sim_nonoise = img.observe_same_nonoise(obs_sim, ttype='direct')
            
            print('# Save results #')
            ### Save results ###
            img.save_fits(self.output_dir + fits_filename)
            obs_sim.save_uvfits(self.output_dir + uvfits_filename)
            # obs_sim_nonoise.save_uvfits(self.output_dir + uvfits_nonoise_filename)
            
            ### Save Difmap-friendly ground-truth model files ###        
            write_difmap_mod(F_A, F_B, self.X_A, self.Y_A, X_B, Y_B, FWHM_A,
                             FWHM_B, self.freq, self.output_dir + mod_filename)
            write_difmap_mod(F_A, F_B, self.X_A, self.Y_A, X_B, Y_B, FWHM_A,
                             FWHM_B, self.freq, self.output_dir + mod_filename[:-4]+'mfitid', names=True)
            
            ### Save .json file with all relevant parameters ###
            params_dict = {
                'Config': {
                    'Source name': self.source_name,
                    'Integration time [s]':self.tint,
                    'Observing date': self.obs_date_string,
                    'MJD': self.mjd,
                    'Observation start time ['+self.timetype+' hours]': round(tstart, 2),
                    'Observing frequency [Hz]': self.freq,
                    'Observing bandwidth [Hz]': self.bw,
                    'Minimum elevation [deg]': self.elevmin,
                    'Maximum elevation [deg]': self.elevmax,
                    'Ground truth image map size [px]': self.npix,
                    'Ground truth image fov [mas]': self.fov,
                    'Fourier transform type': self.ttype,
                    },
                'Simulation parameters': {
                    'File name': uvfits_filename,
                    'Component A total flux [mJy]': round(F_A, 3),
                    'Component B total flux [mJy]': round(F_B, 3),
                    'Component A size [mas]': round(FWHM_A, 3),
                    'Component B size [mas]': round(FWHM_B, 3),
                    'Component B distance [mas]': round(R_B, 1),
                    'Component B position angle [deg]': angle,
                    'Component B RA [mas]': round(X_B, 1),
                    'Component B Dec [mas]': round(Y_B, 1),
                    'Scan length [min]': round(scan_length, 1),
                    'Source right ascension [hours]': round(ra, 1),
                    'Source declination [deg]': round(dec, 1),
                    'Number of antennas': n_ant,
                    'Gain noise quality factor (A thermal noise, B 10% amp and 10 deg phase scatter, C 20% amp and 20 deg phase scatter': gain_noise,
                    }
                }
            
            with open(self.output_dir + self.case_name + file_suffix + '_params.json', 'w') as f:
                json.dump(params_dict, f, indent=4)
        
        else:
            print('\n')
            print('### Creating simulated data set with parameters: ###')
            print('F_A, F_B, FWHM_A, FWHM_B, R_B, angle, gain_noise, use_datafile')
            print(params)
            
            if '.uvfits' in use_dataset:
                use_dataset_print = use_dataset.replace('.uvfits', '')
            elif '.uvf' in use_dataset:
                use_dataset_print = use_dataset.replace('.uvf', '')
            
            pol = detect_pol(use_dataset)
            if pol == None:
                print('Input dataset has no detectable pol. Aborting.')
                return
            if pol == 'R' or pol == 'L':
                # print('Input dataset is single pol.')
                print('Input dataset is single pol. Aborting.')
                return
            
            ### Construct filenames ###
            if type(gain_noise) == str:
                gain_noise_print = gain_noise
            else:
                gain_noise_print = round(gain_noise,2)
            file_suffix = f"_FA{round(F_A,1)}_FWHMA{round(self.FWHM_A,1)}_FB{round(F_B,1)}_FWHMB{round(FWHM_B,1)}_RB{R_B}_b{angle}_file{use_dataset_print.split('/')[-1]}_gain{gain_noise_print}"
            fits_filename = self.case_name + file_suffix + '.fits'
            uvfits_filename = self.case_name + file_suffix + '.uvfits'
            print(f'uvfits_filename: {uvfits_filename}')
            uvfits_nonoise_filename = self.case_name + file_suffix + '_NO_NOISE.uvfits'
            mod_filename = self.case_name + file_suffix + '.mfit'
            log_filename = self.case_name + file_suffix + '_sim.log'
            
            sys.stdout = open(self.output_dir + log_filename, 'w+')
            
            if gain_noise == '' or gain_noise == 0:
                gain_noise = 'A'
            
            ### Add gain noise specifics ###
            if gain_noise == 'A':
                jones = False
                ampcal = True
                phasecal = True
                gainp = 0.1
                gain_offset = 0.1
                phase_std = -1
            elif gain_noise == 'B':
                jones = True
                ampcal = False
                phasecal = False
                gainp = 0.1
                gain_offset = 0.1
                phase_std = 10*np.pi/180
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 10*np.pi/180
                # phase_std = phase_std_dict
            elif gain_noise == 'C':
                jones = True
                ampcal = False
                phasecal = False
                gainp = 0.2
                gain_offset = 0.2
                phase_std = 20*np.pi/180
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 20*np.pi/180
                # phase_std = phase_std_dict
            elif gain_noise == 'D':
                jones = True
                ampcal = False
                phasecal = False
                gainp = 0.2
                gain_offset = 0.2
                phase_std = -1
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 20*np.pi/180
                # phase_std = phase_std_dict
            elif type(gain_noise) == float:
                jones = True
                ampcal = False
                phasecal = False
                gainp = gain_noise
                gain_offset = gain_noise
                phase_std = gain_noise*100*np.pi/180
                # phase_std_dict = {}
                # for key in array.tkey.keys():
                    # phase_std_dict[key] = 20*np.pi/180
                # phase_std = phase_std_dict
            
            print('# Determine position angle of the clean beam #')
            try:
                obs = eh.obsdata.load_uvfits(use_dataset)
            except Exception as e:
                print('Error: uvfits file could not be read into ehtim!')
                print('Error message: ', e)
                return
            
            self.ra = obs.ra
            self.dec = obs.dec
            
            # self.tint = obs.tint
            self.mjd = obs.mjd
            self.freq = obs.rf
            self.bw = obs.bw
            # self.elevmin = obs.elevmin
            # self.elevmax = obs.elevmax
            
            ### Determine position angle of the clean beam for later use ###
            fov = 50
            npix = 128
            clean_beam = obs.cleanbeam(fov=fov*eh.RADPERUAS*1000, npix=npix).fit_gauss(units='natural')
            
            clean_beam[2] = clean_beam[2] - 180    # consistent with angle defined positive North to East
            
            PA_B = clean_beam[2]    # clean beam position angle
            
            if angle == 'maj':
                X_B = -R_B*np.cos((90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+PA_B)*np.pi/180.)
            elif angle == 'min':
                X_B = -R_B*np.cos((90+90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+90+PA_B)*np.pi/180.)
            elif type(angle) == float or type(angle) == int:
                PA_B = angle
                X_B = -R_B*np.cos((90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+PA_B)*np.pi/180.)
            else:
                print('Warning: no valid position angle for component B provided, set to 0.')
                PA_B = 0
                X_B = -R_B*np.cos((90+PA_B)*np.pi/180.)
                Y_B = R_B*np.sin((90+PA_B)*np.pi/180.)
            
            print('# Create ground truth image #')
            ### Create a new image ###
            # img = eh.image.make_empty(self.npix, self.fov*1e3*eh.RADPERUAS,
                                      # self.ra, self.dec, self.freq)
            img = eh.image.make_square(obs, self.npix, self.fov*1e3*eh.RADPERUAS)
            self.source_name = img.source
            self.mjd = img.mjd
            
            ### Add primary component (A) to the image ###
            img = img.add_gauss(F_A*1e-3, [FWHM_A*1e3*eh.RADPERUAS,
                                           FWHM_A*1e3*eh.RADPERUAS,
                                           0,
                                           self.X_A*1e3*eh.RADPERUAS,
                                           self.Y_A*1e3*eh.RADPERUAS])
            
            ### Add secondary component (B) to the image ###
            if F_B > 1e-3:
                # add only if sufficiently bright (more than 1 microJansky)
                # needed because ehtim complains when flux is too low...
                # TO-DO: clean up this implementation
                img = img.add_gauss(F_B*1e-3, [FWHM_B*1e3*eh.RADPERUAS,
                                               FWHM_B*1e3*eh.RADPERUAS,
                                               0,
                                               X_B*1e3*eh.RADPERUAS,
                                               Y_B*1e3*eh.RADPERUAS])
            
            # Check if SEFDs are present in array information and add them if not #
            printed = False
            for i in range(len(obs.tarr)):
                if obs.tarr[i][4] == 0 and obs.tarr[i][5] == 0:
                    if printed == False:
                        print('No SEFDs given in array information, adding them.')
                        printed = True
                    try:
                        obs.tarr[i][4] = self.sefds[i][0]
                        obs.tarr[i][5] = self.sefds[i][1]
                    except:
                        obs.tarr[i][4] = self.sefds_default[0]
                        obs.tarr[i][5] = self.sefds_default[1]
            array_use = eh.array.Array(obs.tarr)
            
            # Determine relevant observation parameters from data #
            dat = obs.unpack(['time'], mode='all')
            tint_use = np.unique(dat)[1][0] - np.unique(dat)[0][0]    # in hours
            tint_use *= 3600    # in seconds
            tadv_use = tint_use
            bw_use = obs.bw
            tstart_use = obs.tstart
            tstop_use = obs.tstop
            mjd_use = obs.mjd
            elevmax_use = 90
            
            print('# Create simulated observation #')
            ### Simulate observation ###
            obs_sim = img.observe(
                array_use,
                tint=tint_use,
                tadv=tadv_use,
                bw=bw_use,
                tstart=tstart_use,
                tstop=tstop_use,
                mjd=mjd_use,
                timetype=self.timetype,
                elevmin=self.elevmin,
                elevmax=elevmax_use,
                add_th_noise=True,
                jones=jones,
                ampcal=ampcal,
                phasecal=phasecal,
                gainp=gainp,
                gain_offset=gain_offset,
                phase_std=phase_std,
                seed=self.seed,
                ttype=self.ttype
                )
            
            # eh.plotall_obs_compare([obs,obs_sim], 'uvdist', 'sigma', show=True)
            # input()
            
            obs_sim.save_uvfits(self.output_dir + uvfits_filename)
            
            print('# Save results #')
            ### Save results ###
            img.save_fits(self.output_dir + fits_filename)
            if pol == 'R' or pol == 'L':
                print(f'Single polarization ({pol}) detected, adjusting output file.')
                remove_pol(self.output_dir + uvfits_filename, pol=pol)
                
                # obs_sim.save_uvfits(self.output_dir + uvfits_filename, force_singlepol=pol)
            # obs_sim_nonoise.save_uvfits(self.output_dir + uvfits_nonoise_filename)
            
            ### Save Difmap-friendly ground-truth model files ###        
            write_difmap_mod(F_A, F_B, self.X_A, self.Y_A, X_B, Y_B, FWHM_A,
                             FWHM_B, self.freq, self.output_dir + mod_filename)
            write_difmap_mod(F_A, F_B, self.X_A, self.Y_A, X_B, Y_B, FWHM_A,
                             FWHM_B, self.freq, self.output_dir + mod_filename[:-4]+'mfitid', names=True)
            
            ### Save .json file with all relevant parameters ###
            params_dict = {
                'Config': {
                    'Source name': self.source_name,
                    'Integration time [s]': tint_use,
                    # 'Observing date': self.obs_date_string,
                    'MJD': self.mjd,
                    'Observing frequency [Hz]': self.freq,
                    'Observing bandwidth [Hz]': self.bw,
                    'Minimum elevation [deg]': self.elevmin,
                    'Maximum elevation [deg]': elevmax_use,
                    'Ground truth image map size [px]': self.npix,
                    'Ground truth image fov [mas]': self.fov,
                    'Fourier transform type': self.ttype,
                    },
                'Simulation parameters': {
                    'File name': uvfits_filename,
                    'Component A total flux [mJy]': round(F_A, 3),
                    'Component B total flux [mJy]': round(F_B, 3),
                    'Component A size [mas]': round(FWHM_A, 3),
                    'Component B size [mas]': round(FWHM_B, 3),
                    'Component B distance [mas]': round(R_B, 1),
                    'Component B position angle [deg]': angle,
                    'Component B RA [mas]': round(X_B, 1),
                    'Component B Dec [mas]': round(Y_B, 1),
                    'Basis uvfits file': use_dataset,
                    }
                }
            
            with open(self.output_dir + self.case_name + file_suffix + '_params.json', 'w') as f:
                json.dump(params_dict, f, indent=4)
        
        sys.stdout = old_sys
    
    
    
    def run_all(self,
                F_A_arr,
                flux_ratio_arr,
                R_B_arr,
                angles,
                scan_length_arr=[],
                ra_arr=[],
                dec_arr=[],
                n_ant_arr=[],
                gain_noise_arr=[],
                parallelize=True,
                overwrite=True,
                n_processes=2,
                map_processes='imap_unordered',
                files_exist=None,
                use_datasets='',
                override_confirm=False):
        '''
        # Purpose: Function to run a set of simulations using the simulate_case() function.
        Input arrays of parameters determine how many datasets are generated.
        Inputs are expected to be lists or arrays, but single values can also
        be given.
        
        # Args:
            F_A_arr (list or float): Flux densities of the primary Gaussian
              component to be tested, given in mJy.
            F_B_arr (list or float): Flux densities of the secondary Gaussian
              component to be tested, given in mJy.
            flux_ratio_arr (list or float): flux ratios of secondary to primary
              component to be tested.
            R_B_arr (list or float): The radial distances of the secondary
              Gaussian component to the primary to be tested, given in mas.
            angles (list or str): A string input determining where the secondary
              component will be placed w.r.t. the clean beam. Can be 'maj' or
              'min' to be placed along the clean beam major or minor axis,
              respectively.
            scan_length_arr (list or float): Lengths of the observing scan to
              be tested, given in minutes.
            ra_arr (list or float): Right ascensions of the observation to be
              tested, given in hours.
            dec_arr (list or float): Declinations of the observation to be
              tested, given in degrees.
            n_ant_arr (list or int): Number of antennas to be observing. If
              fewer than the amount of antennas in the array file are defined,
              antennas are removed at random from the array.
            gain_noise_arr (str): A (list of) strings determining the data quality
              to be desired. More gain noise is added for lower quality cases.
              Now accepts 'A', 'B', and 'C', where 'A' applies only thermal
              noise, and 'B' and 'C' progressively more station gain errors in
              both amplitude and phase. See simulate_case() for more details.
            parallelize (bool): determines if code will be parallelized
              (default is True).
            overwrite (bool): determines if existing datasets are to be
              overwritten or skipped (default is True).
            n_processes (int): determine how many processes are done in
              parallel given that parellelize=True. Default is 2.
            map_processes (str): parameter which determines how processes are
              mapped, i.e. how workers are assigned to available cores. Options:
              'map', 'imap', 'imap_unordered' or 'starmap'. Default is
              'imap_unordered'. See documentation of multiprocessing for more
              details.
            files_exist (str): if this is given, will expect and load a text
              file with filenames that already have been created and are to be
              skipped from the list of possible simulated datasets (default None).
        
        # Returns:
            Returns the same output as simulate_case() for as many datasets as
            desired.
        '''
        start_time = datetime.datetime.now()
        
        if use_datasets == '':
            
            params = {'F_A':F_A_arr, 'flux_ratio':flux_ratio_arr, 'R_B':R_B_arr,
                      'angle':angles, 'scan_length':scan_length_arr, 'ra':ra_arr,
                      'dec':dec_arr, 'n_ant':n_ant_arr, 'gain_noise':gain_noise_arr}
            for i, key in enumerate(params):
                if not isinstance(params[key], (list, np.ndarray)):
                    params[key] = [params[key]]
            
            fwhm_ratio_arr = np.sqrt(params['flux_ratio'])
            
            n_datasets = len(params['F_A'])*len(params['flux_ratio'])\
                         * len(params['R_B'])*len(params['scan_length'])\
                         * len(params['ra'])*len(params['dec'])*len(params['angle'])\
                         * len(params['n_ant'])*len(params['gain_noise'])
            if n_datasets == 0:
                print('! Warning: at least one parameter to be varied for the simulated'
                      ' data has not been set. Check input parameters:')
                print(params)
        else:
            if not type(use_datasets) == list and  not type(use_datasets) == np.ndarray:
                use_datasets = [use_datasets]
            print('Use provided dataset to calculate simulated visibilities')
            params = {'F_A':F_A_arr, 'flux_ratio':flux_ratio_arr, 'R_B':R_B_arr,
                      'angle':angles, 'gain_noise':gain_noise_arr,
                      'datasets':use_datasets}
            for i, key in enumerate(params):
                if not isinstance(params[key], (list, np.ndarray)):
                    params[key] = [params[key]]
            
            fwhm_ratio_arr = np.sqrt(params['flux_ratio'])
            
            n_datasets = len(params['F_A'])*len(params['flux_ratio'])\
                         * len(params['R_B'])*len(params['angle'])\
                         * len(params['gain_noise'])*len(params['datasets'])
        
        print('\n')
        print('### run_all script started... ###')
        print('# Possible data sets: {} cases #'.format(n_datasets))
        
        if files_exist != None:
            with open(files_exist, 'r') as f:
                files_existing = f.readlines()
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        if use_datasets == '':
            cases = []
            current_files_exist = []
            uvf_files_list = []
            for F_A in params['F_A']:
                for flux_ratio, fwhm_ratio in zip(params['flux_ratio'], fwhm_ratio_arr):
                    F_B = F_A * flux_ratio
                    FWHM_B = self.FWHM_A * fwhm_ratio
                    for R_B in params['R_B']:
                        for scan_length in params['scan_length']:
                            for ra in params['ra']:
                                for dec in params['dec']:
                                    for angle in params['angle']:
                                        for n_ant in params['n_ant']:
                                            for gain_noise in params['gain_noise']:
                                                if type(gain_noise) == str:
                                                    gain_noise_print = gain_noise
                                                else:
                                                    gain_noise_print = round(gain_noise,2)
                                                file_suffix = f"_FA{round(F_A,1)}_FWHMA{round(self.FWHM_A,1)}_FB{round(F_B,1)}_FWHMB{round(FWHM_B,1)}_RB{R_B}_b{angle}_scan{scan_length}_ra{round(ra,1)}_dec{round(dec,1)}_Nant{n_ant}_gain{gain_noise_print}"
                                                uvf_files_list.append(self.case_name + file_suffix)
                                                if os.path.exists(self.output_dir + self.case_name + file_suffix + '.uvfits'):
                                                    current_files_exist.append(self.output_dir + self.case_name + file_suffix + '.uvfits')
                                                    if overwrite == True:
                                                        cases.append((F_A, F_B, self.FWHM_A, FWHM_B, R_B, angle, scan_length, ra, dec, n_ant, gain_noise))
                                                    else:
                                                        continue
                                                else:
                                                    if files_exist != None:
                                                        if any(self.case_name + file_suffix + '.uvfits' in file_existing for file_existing in files_existing):
                                                            continue
                                                    cases.append((F_A, F_B, self.FWHM_A, FWHM_B, R_B, angle, scan_length, ra, dec, n_ant, gain_noise))
        else:
            cases = []
            current_files_exist = []
            uvf_files_list = []
            for F_A in params['F_A']:
                for flux_ratio, fwhm_ratio in zip(params['flux_ratio'], fwhm_ratio_arr):
                    F_B = F_A * flux_ratio
                    FWHM_B = self.FWHM_A * fwhm_ratio
                    for R_B in params['R_B']:
                        for angle in params['angle']:
                            for gain_noise in params['gain_noise']:
                                if type(gain_noise) == str:
                                    gain_noise_print = gain_noise
                                else:
                                    gain_noise_print = round(gain_noise,2)
                                for dataset in use_datasets:
                                    if '.uvfits' in dataset:
                                        dataset_print = dataset.replace('.uvfits', '')
                                    elif '.uvf' in dataset:
                                        dataset_print = use_dataset.replace('.uvf', '')
                                    file_suffix = f"_FA{round(F_A,1)}_FWHMA{round(self.FWHM_A,1)}_FB{round(F_B,1)}_FWHMB{round(FWHM_B,1)}_RB{R_B}_b{angle}_file{dataset_print.split('/')[-1]}_gain{gain_noise_print}"
                                    uvf_files_list.append(self.case_name + file_suffix)
                                    if os.path.exists(self.output_dir + self.case_name + file_suffix + '.uvfits'):
                                        current_files_exist.append(self.output_dir + self.case_name + file_suffix + '.uvfits')
                                        if overwrite == True:
                                            cases.append((F_A, F_B, self.FWHM_A, FWHM_B, R_B, angle, gain_noise, dataset))
                                        else:
                                            continue
                                    else:
                                        if files_exist != None:
                                            if any(self.case_name + file_suffix + '.uvfits' in file_existing for file_existing in files_existing):
                                                continue
                                        cases.append((F_A, F_B, self.FWHM_A, FWHM_B, R_B, angle, gain_noise, dataset))
        
        print('# Already existing simulated data sets: {} cases #'.format(n_datasets-len(cases)))
        print('# Creating simulated data sets: {} cases #'.format(len(cases)))
        if override_confirm == False:
            input('Confirm by pressing Enter \n')
        
        with open('current_files_exist.txt', 'w') as f:
            for filename in current_files_exist:
                f.write(filename + '\n')
        
        with open('uvf_files_list.txt', 'w') as f:
            for filename in uvf_files_list:
                f.write(filename + '\n')
        
        if parallelize == True:
            # Use multiprocessing to parallelize the simulations
            with Pool(processes=n_processes) as pool:
                if map_processes == 'map':
                    pool.map(self.simulate_case, cases)
                elif map_processes == 'imap':
                    for __ in pool.imap(self.simulate_case, cases, chunksize=10):
                        pass
                elif map_processes == 'imap_unordered':
                    for __ in pool.imap_unordered(self.simulate_case, cases):
                        pass
                elif map_processes == 'starmap':
                    for __ in pool.starmap(self.simulate_case, [(params,) for params in cases]):
                        pass
        else:
            # Without multiprocessing
            for case in cases:
                self.simulate_case(case)
        
        print("Finished!")
        end_time = datetime.datetime.now()
        print("Script run time:")
        print(end_time - start_time)



if __name__ == "__main__":
    
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
        # files_exist='files_exist.txt'    # can provide file here, uncomment
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


