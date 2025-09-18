"""
Ocean glider sensor correction and plotting routines
"""
import xarray as xr
import numpy as np
import pandas as pd
import seawater
import scipy.stats as stats
from scipy import signal
from scipy.optimize import curve_fit
from shapely.geometry import Polygon
from scipy import signal
import seawater as sw
import warnings
warnings.filterwarnings('ignore')

def get_timeseries(deploy_prefix, deploy_name, level='L0'):
    return xr.open_dataset(f'{deploy_prefix}/{deploy_name}_delayed.nc')
# def get_timeseries(filename, level='L0'):
    # return xr.open_dataset(f'{filename}')

def get_gridfile(deploy_prefix, deploy_name, level='L0'):
    return xr.open_dataset(f'{deploy_prefix}/{deploy_name}_grid_delayed.nc')
	
### IDENTIFYING UNPHYSICAL CONDUCTIVITY VALUES ### 

def get_conductivity_clean(ts0, dT, dz, flag_stdev, clean_stdev, accuracy):
    ts = ts0.copy(deep=True).load()
    ts = ts.where(np.isfinite(ts.conductivity), drop=False)
    ts = ts.assign(conductivityClean = np.nan*ts.conductivity)

    condBad = np.zeros(len(ts.conductivityClean.values),dtype=bool)

    # Set up bins and find clean conductivity values
    Tbins = np.arange(np.min(ts.profile_index),
    np.max(ts.profile_index)+dT,dT)
    zbins = np.arange(np.min(ts.depth),np.max(ts.depth)+dz,dz)
    for n, Tbin in enumerate(Tbins[:-1]):
        ind_Tbin = np.logical_and(ts.profile_index.values>=Tbin, 
        ts.profile_index.values<=Tbins[n+1])
        cond = ts.conductivity[ind_Tbin].values

        ind_bad_z = np.zeros(len(cond),dtype=bool)
        for m, zbin in enumerate(zbins[:-1]):
            ind_zbin = np.logical_and(ts.depth[ind_Tbin].values>=zbin,
            ts.depth[ind_Tbin].values<=zbins[m+1])
            cond_z = cond[ind_zbin]
            cond_mean = np.nanmean(cond_z)
            cond_std = np.nanstd(cond_z)
            ind_flag = np.logical_and(np.fabs(cond_z-cond_mean) > 
            (flag_stdev * cond_std), np.fabs(cond_z-cond_mean) > accuracy)
            ind_bad = np.logical_and(np.fabs(cond_z-np.nanmean(cond_z[~ind_flag])) > 
            (clean_stdev * np.nanstd(cond_z[~ind_flag])), 
            np.fabs(cond_z-np.nanmean(cond_z[~ind_flag])) > accuracy)
            ind_bad_z[ind_zbin] = ind_bad
        condBad[ind_Tbin] = ind_bad_z
        cond[ind_bad_z] = np.nan
        ts['conductivityClean'][ind_Tbin] = cond

		# Recalculate other fields
    ts['salinity'].values = seawater.eos80.salt(
        ts.conductivityClean / seawater.constants.c3515 * 10, 
        ts.temperature, 
        ts.pressure
    )
    ts['density'].values = seawater.eos80.dens(
        ts.salinity, 
        ts.temperature, 
        ts.pressure
    )

    return ts 
	
### SENSOR ALIGNMENT CORRECTION ### 
	

def advance_field(advance, interval, field):
#Advance the conductivity or temperature and calculate the corrected fields
    if advance == 0:
        x0 = field
    else:
        # make a time vector in seconds based upon the sampling interval
        time_vector = np.arange(interval, interval*len(field)+1, interval)

        # Align the data with linear interpolation
        x0 = np.interp(time_vector - advance, time_vector, field)
        
        return x0
	
	
def get_spectrogram(fname, n, density_cutoff, clean_profs_start,
					clean_profs_end, num_profs, fs, c_val, freq, bad_profiles):
	# Spectrogram with density cutoff for chosen subset of profiles

	# Load dataset and assign profile index coord
	with xr.open_dataset(fname).load() as ts0:
		ts_tmp = ts0.copy(deep=True).load()
		ts_tmp = ts_tmp.assign_coords(pind=ts_tmp.profile_index)
		# ts_tmp = ts_tmp.where(~ts_tmp.profile_index.isin(bad_profiles))
		# ts_tmp['temperature'] = ts_tmp.temperature # added!
		# ts_tmp['density'] = ts_tmp.density # added as well

	# Set up our constants
	if n == None:
		nft = 256
	else:
		nft = n

	print('Applying density cutoff')
	density_bool = ts_tmp.density >= density_cutoff
	ts_tmp = ts_tmp.where(density_bool, drop=True)
	ts_tmp = ts_tmp.where(np.isfinite(ts_tmp.temperature +
									  ts_tmp.conductivityClean), drop=True)
	ts_tmp = ts_tmp.where(~np.isnan(ts_tmp.temperature +
									ts_tmp.conductivityClean), drop=True)
	C_full = ts_tmp.conductivityClean.values
	time = ts_tmp.time
	T_full = ts_tmp.temperature.values

	tot_profs = int(np.max(ts_tmp.profile_index.values))
	print('Total number of profiles:', tot_profs)

	profile_one = ts_tmp.profile_index[0] + clean_profs_start
	profile_end = ts_tmp.profile_index.max() - clean_profs_end
	profile_list = np.round(np.linspace(int(profile_one), int(profile_end),
										num=num_profs))
	profile_bins = np.unique(profile_list)  # the actual profile numbers to be used
	profile_bins = profile_bins[np.isfinite(profile_bins)]

	print('Ready to loop over profiles')

	for nn, prof in enumerate(profile_bins):
		index = ts_tmp.profile_index.isin(prof)
		C = C_full[index]
		T = T_full[index]
		fldx = (T - np.nanmean(T)) / np.nanstd(T)
		fldy = (C - np.nanmean(C)) / np.nanstd(C)
		frqc, coher = signal.coherence(fldx, fldy, fs=fs, nperseg=n, nfft=None)
		frq, spec = signal.csd(fldx, fldy, fs=fs, nperseg=n, nfft=None,
							   scaling='spectrum')

		xphase = np.arctan2(-np.imag(spec), np.real(spec))  # *180/np.pi

		edof = (8/3) * (len(fldx) / (nft/2))
		gamma95 = 1. - (0.05)**(1. / (edof - 1.))
		conf95 = (coher > gamma95)

		xphase[(coher < c_val) | ~conf95] = np.nan
		coher[(coher < c_val) | ~conf95] = np.nan
		xmag = np.abs(spec)

		if nn == 0:
			mag_all = np.interp(freq, frq, xmag)
			phase_all = np.interp(freq, frq, xphase)
			coh_all = np.interp(freq, frqc, coher)
		else:
			mag_all = np.column_stack((mag_all,
									   np.interp(freq, frq, xmag)))
			phase_all = np.column_stack((phase_all,
										 np.interp(freq, frq, xphase)))
			coh_all = np.column_stack((coh_all,
									   np.interp(freq, frqc, coher)))

	return mag_all, phase_all, coh_all, profile_bins


def tau_func(freq,tau_C):
#Function to fit to phase
    y = 2*np.pi*freq*tau_C
    return y
	
def get_tau_C(freq, freq_cut, fs, phase_all):
	#Note that phase is in radians, while frequency is in Hz, converted to radians within tau_func
	fr_cut = (freq<freq_cut*fs)# & (freq>freq_cutmin*fs))
	fld = phase_all[fr_cut,:].flatten()
	fr_fld = np.repeat(freq[fr_cut],np.shape(phase_all)[1])
	fr_fld = fr_fld[~np.isnan(fld)]
	fld = fld[~np.isnan(fld)]
	parameters, covariance = curve_fit(tau_func, fr_fld, fld)
	tau_C = parameters[0]
	fit_y = tau_func(fr_fld, tau_C)
	
	return tau_C, covariance, fr_fld, fld, fit_y

def alignment_correction(fname, srate, advance_C, advance_T):
	ts = xr.open_dataset(fname).load()
	ts_tmp = ts.copy(deep=True)
	ts_tmp = ts_tmp.where(np.isfinite(ts_tmp.temperature + 
									  ts_tmp.conductivity), drop=False)

	interval = srate.astype(float) #Sampling interval in seconds

	advance = advance_C #How much to advance field in seconds
	field = ts_tmp.conductivityClean
	x0 = advance_field(advance, interval, field)
	ts_tmp['conductivityClean'].values =  x0

	advance = advance_T #How much to advance field in seconds
	field = ts_tmp.temperature
	x0 = advance_field(advance, interval, field)
	ts_tmp['temperature'].values =  x0

	# Recalculate other fields
	ts_tmp['salinity'].values = seawater.eos80.salt(
		ts_tmp.conductivityClean / seawater.constants.c3515 * 10, 
		ts_tmp.temperature, 
		ts_tmp.pressure
	)
	ts_tmp['density'].values = seawater.eos80.dens(
		ts_tmp.salinity, 
		ts_tmp.temperature, 
		ts_tmp.pressure
	)

	return ts_tmp

### IDENTIFYING UNPHYSICAL SALINITY PROFILES ### 
	
def get_salinity_grid(ts, Tmean, clean_profs, flag_stdev, clean_stdev, clean_cutoff, dtbin): 
# Function to bin the salinity into temperature space and identify bad profiles
    ts = ts.where(np.isfinite(ts.salinity), drop=True)
    ts_tmp = ts.assign_coords(pind=ts.profile_index)
    
    # Bin the data with temperature and profile index
    tbins = np.array(Tmean[::dtbin]) 
    
    profile_bins = np.unique(ts['profile_index']) 
    profile_bins = np.hstack([profile_bins, profile_bins[-1]+1])
    
    salin, xx, yy, binn = stats.binned_statistic_2d(
                        ts['temperature'].values,
                        ts['profile_index'].values,
                        values=ts['salinity'].values, statistic='mean',
                        bins=[tbins, profile_bins-0.5])

    # Create new xarray dataset with the binned salinity data
    profile_index = ts_tmp.profile_index.groupby('pind').median(dim='time')
    direction = ts_tmp.profile_direction.groupby('pind').median(dim='time')
    
    sal = xr.Dataset({
        'profiles'         : profile_bins[:-1],
        'temperature'      : tbins[:-1]+0.5*np.diff(tbins),
        'salinity'         : (('temperature', 'profiles'), salin),
        'profile_index'    : (('profiles'), profile_index.values),
        'profile_direction': (('profiles'), direction.values)})
    
    # Determine salinity values that are not temporarily flagged as bad
    sal['dS'] = np.fabs(sal.salinity - sal.salinity.mean(dim='profiles'))
    sal['salinityNoflag'] = sal.salinity.where((sal.dS < flag_stdev * sal.salinity.std(dim='profiles')) | 
                                               (sal.profile_index < clean_profs) | 
                                               (sal.profile_index > (np.max(sal.profile_index)-clean_profs)))

    # Determine salinity values that are 'clean' - not flagged as bad
    sal['dS'] = np.fabs(sal.salinity - sal.salinityNoflag.mean(dim='profiles'))
    sal['salinityClean'] = sal.salinity.where((sal.dS < clean_stdev * sal.salinityNoflag.std(dim='profiles')) | 
                                              (sal.profile_index < clean_profs) | 
                                              (sal.profile_index > (np.max(sal.profile_index)-clean_profs)))  
                                                                                                           
    
    # Find the sum of clean values in each profile relative to the total number of values
    sal['NSal'] = np.isfinite(sal.salinity).sum(dim='temperature')
    sal['NSalClean'] = np.isfinite(sal.salinityClean).sum(dim='temperature')

    # Calculate the fraction of values per profile that are bad
    sal['bad'] = (sal.NSal - sal.NSalClean) / sal.NSal
    
    # Save the profiles that are not flagged as bad in 'salinityGood'
    bad_profiles = sal.profiles.where(sal.bad >= clean_cutoff, drop=True)
    sal['salinityGood'] = sal.salinity.where(
	[index not in bad_profiles.values for index in sal.profile_index.values]) 
    
    return sal



### THERMAL LAG CORRECTION ALGORITHM ### 

def profile_area(ts_grp):
# Function to find the area between pairs of profiles using the shapely polygon routine

    #index of downcasts and upcasts    
    dn_index = ts_grp.profile_direction>0
    up_index = ts_grp.profile_direction<0
    
    #downcast T,S and upcast T,S
    dn_T = ts_grp.temperature[dn_index]
    dn_S = ts_grp.salinity[dn_index]
    up_T = ts_grp.temperature[up_index]
    up_S = ts_grp.salinity[up_index]
    
    #remove nans
    dn_not_nan = (~pd.isnull(dn_T)) & (~pd.isnull(dn_S)) # changed np.isnan to pd.isnull after error - Andrea July 9 2024
    up_not_nan = (~pd.isnull(up_T)) & (~pd.isnull(up_S))
    dn_T = dn_T[dn_not_nan]
    dn_S = dn_S[dn_not_nan]
    up_T = up_T[up_not_nan]
    up_S = up_S[up_not_nan]

    #create a polygon from the T,S downcast and upcast
    polygon_points_x = [] 
    polygon_points_y = []

    polygon_points_x += dn_S.values.tolist()
    polygon_points_y += dn_T.values.tolist() #append all xy points for curve 1

    polygon_points_x += up_S.values.tolist()
    polygon_points_y += up_T.values.tolist() #append all xy points for curve 2 in the reverse order (from last point to first point)
    
    polygon_points_x += dn_S[0:1].values.tolist()
    polygon_points_y += dn_T[0:1].values.tolist() #append the first point in curve 1 again, so it "closes" the polygon

    polygon = Polygon(zip(polygon_points_x, polygon_points_y))
    
    #determine the area of the polygon
    ts_grp['area'] = polygon.area
    
    #check and make sure we have both a downcast and an upcast (no empty profiles)
    if not (dn_T.values.tolist() and 
            dn_S.values.tolist() and 
            up_T.values.tolist() and 
            up_S.values.tolist()):
        ts_grp['area'] = np.nan   
    
    return ts_grp

def downcast_area(ts_sub, profile_bins_all, direction, dn_area):
# Function to find the area between subsequent downcasts using the shapely polygon routine

    #Loop through all the downcasts
    for n in range(len(direction)-1):
        if direction[n]>0: #downcasts
            dn_1 = (
                (ts_sub['profile_direction']>0) & 
                (ts_sub['profile_index']==profile_bins_all[n])
            )
            dn_2 = (
                (ts_sub['profile_direction']>0) & 
                (ts_sub['profile_index']==(profile_bins_all[n]+2))
            )

            #temperature and salinity for each downcast in the pair
            dn_1_T = ts_sub.where(dn_1, drop=True).temperature
            dn_1_S = ts_sub.where(dn_1, drop=True).salinity
            dn_2_T = ts_sub.where(dn_2, drop=True).temperature
            dn_2_S = ts_sub.where(dn_2, drop=True).salinity

            #remove nans
            dn_1_not_nan = (~np.isnan(dn_1_T)) & (~np.isnan(dn_1_S))
            dn_2_not_nan = (~np.isnan(dn_2_T)) & (~np.isnan(dn_2_S))
            dn_1_T = dn_1_T[dn_1_not_nan]
            dn_1_S = dn_1_S[dn_1_not_nan]
            dn_2_T = dn_2_T[dn_2_not_nan]
            dn_2_S = dn_2_S[dn_2_not_nan] 
           
            #check and make sure we have both a downcast and an upcast (no empty profiles)
            if not (dn_1_T.values.tolist() and 
                    dn_1_S.values.tolist() and 
                    dn_2_T.values.tolist() and 
                    dn_2_S.values.tolist()):
                dn_area[n] = np.nan
                dn_area[n+1] = np.nan

            else: 
                #create a polygon from the T,S downcast and upcast
                polygon_points_x = [] 
                polygon_points_y = []

                polygon_points_x += dn_1_S.values.tolist()
                polygon_points_y += dn_1_T.values.tolist() #append all xy points for curve 1

                polygon_points_x += np.flip(dn_2_S.values).tolist()
                polygon_points_y += np.flip(dn_2_T.values).tolist() #append all xy points for curve 2 in the reverse order (from last point to first point)
    
                polygon_points_x += dn_1_S[0:1].values.tolist()
                polygon_points_y += dn_1_T[0:1].values.tolist() #append the first point in curve 1 again, so it "closes" the polygon

                polygon = Polygon(zip(polygon_points_x, polygon_points_y))
    
                #determine the area of the polygon
                dn_area[n] = polygon.area 
                dn_area[n+1] = polygon.area 

    return dn_area
	
def profile_pairs(ts, clean_profs_start, clean_profs_end, num_profs, bad_profiles=[]):
# Function to separate out a subset of data and identify pairs of profiles
	profile_one = ts.profile_index.min()+clean_profs_start #chose the range of profiles to work with
	profile_end = ts.profile_index.max()-clean_profs_end
    
	profile_list = np.round(np.linspace( #n evenly spaced profiles 
	int(profile_one),int(profile_end),num=num_profs)) 
    
	profile_list_all = np.arange(profile_one,profile_end+1,1) #all profiles in the chosen range

	profile_bins = np.unique(profile_list) #the actual profile numbers to be used
	profile_bins = profile_bins[np.isfinite(profile_bins)]

	profile_bins_all = np.unique(profile_list_all) #all the profile numbers in the chosen range
	profile_bins_all = profile_bins_all[np.isfinite(profile_bins_all)]

	direction = np.empty_like(profile_bins_all) #find direction of each profile, for all profiles in range
	direction = (ts.profile_direction.where(ts.profile_index.isin(profile_bins_all),
											drop=True)).groupby('pind').median(dim='time') 

	#Create a new array to work with, and extract all profiles in the chosen range
	ts_sub = ts.copy(deep=True)
	ts_sub = ts_sub.where(np.isfinite(ts_sub.temperature + 
									  ts_sub.conductivity), drop=True)
	ts_sub = ts_sub.where(ts_sub.profile_index.isin(profile_bins_all), 
						  drop=True)

	#Create a new variable 'pair' that will be used to group pairs of profiles
	ts_sub['pair'] = np.nan*ts_sub.profile_index 
	for n in range(len(direction)-1):
		if direction[n]>0: #downcasts
			is_pair = (
				(
					(ts_sub['profile_direction']>0) & 
					(ts_sub['profile_index']==profile_bins_all[n])##profile index = pair idx identified 
				) |
				(
					(ts_sub['profile_direction']<0) & 
					(ts_sub['profile_index']==(profile_bins_all[n]+1))##set all values in prof. as pair idx 
				)
			)
			pair_counter = profile_bins_all[n]        
			ts_sub['pair'][is_pair] = pair_counter

		elif direction[n]<0: #upcasts
			is_pair = (
				(
					(ts_sub['profile_direction']>0) & 
					(ts_sub['profile_index']==(profile_bins_all[n]-1))
				) |
				(
					(ts_sub['profile_direction']<0) & 
					(ts_sub['profile_index']==(profile_bins_all[n]))
				)
			)
			pair_counter = profile_bins_all[n]-1        
			ts_sub['pair'][is_pair] = pair_counter
			
	#Remove the salinity profiles identified as bad
	if np.any(bad_profiles):
		for bad in bad_profiles.values:
			ts_sub = ts_sub.where(~(ts_sub.profile_index==bad), drop=True)
			direction = direction[~(profile_bins_all==bad)]
			profile_bins = profile_bins[~(profile_bins==bad)]
			profile_bins_all = profile_bins_all[~(profile_bins_all==bad)]
	
	#Remove any pairs that are nan
	nan_pair_values = ts_sub.profile_index[~np.isfinite(ts_sub.pair)]
	for nan_pair in np.unique(nan_pair_values):
		ts_sub = ts_sub.where(~(ts_sub.profile_index==nan_pair), drop=True)
		direction = direction[~(profile_bins_all==nan_pair)]
		profile_bins = profile_bins[~(profile_bins==nan_pair)]
		profile_bins_all = profile_bins_all[~(profile_bins_all==nan_pair)]

	return ts_sub, profile_bins, profile_bins_all, direction



def TS_diff(alphatau, fn, density_bool, area_bad, profile_bins, profile_bins_all, ts_sub, ret_err=True):
# Main function that tests all the values of alpha and tau and finds the minimum area

    #set area for all profiles to nan
    area = np.nan*np.zeros([len(np.round(0.5*profile_bins,0))+1])
    
    #unpack alpha and tau
    alpha, tau = alphatau
    alpha = alpha / 1e3
    
    #calculate the filter coefficient
    coefa = 4 * fn * alpha * tau / (1 + 4 * fn * tau)
    if coefa == 0:
        coefb = 0
    else:
        coefb = 1 - 2 * coefa / alpha
    b = np.array([1, -1]) * coefa
    a = np.array([1, coefb], dtype=object) # added dtype=object, andrea jul 9 2024

    #new array to apply the filter to
    ts_tmp = ts_sub.copy(deep=True)

    #apply filter to temperature
    x0 = ts_tmp.temperature.values
    x0 = signal.lfilter(b, a, ts_tmp.temperature.values)

    #recalculate temperature and salinity
    ts_tmp['temperature'] =  ts_tmp.temperature - x0
    ts_tmp = ts_tmp.where(density_bool, drop=True)
    ts_tmp = ts_tmp.where(ts_tmp.conductivity>=0, drop=True)
    ts_tmp['salinity'].values = seawater.eos80.salt(
        ts_tmp.conductivity / seawater.constants.c3515 * 10, 
        ts_tmp.temperature, 
        ts_tmp.pressure
    )   
    
    #remove densities lighter than density_cutoff above, then recalculate density
    ts_tmp['density'].values = seawater.eos80.dens(
        ts_tmp.salinity, 
        ts_tmp.temperature, 
        ts_tmp.pressure
    )
    
    #exclude profiles with large areas between subsequent downcasts
    ts_tmp = ts_tmp.where(ts_tmp.profile_index.isin(
                 profile_bins_all[(~area_bad)]), 
                 drop=True)
				 
    #extract the subset of pairs of profiles we want to work with
    ts_tmp = ts_tmp.where(ts_tmp.profile_index.isin(profile_bins) | 
                  ts_tmp.profile_index.isin(profile_bins+1) | 
                  ts_tmp.profile_index.isin(profile_bins-1), 
                  drop=True)

    #Apply the function profile_area to each pair of profiles
    ts_tmp = ts_tmp.groupby('pair').apply(profile_area, shortcut=True)
    area = (ts_tmp.area).groupby('pind').median(dim='time') 
    p_ind = np.unique(ts_tmp.pind)
    
    #calculate root-mean squared difference for all profile pairs 
    #with area within 2 standard deviation of the median
    area_lim = area<(np.nanmedian(area)+2*np.nanstd(area))
    err = np.sqrt(np.nansum(area[area_lim]**2) / (len(np.unique(area[area_lim].pind))-1)) 

    ts_tmp.close()
    
    if ret_err:
        return err
    else:
        return area, p_ind

def TS_preprocess(density_bool, dn_stdev, profile_bins, profile_bins_all, direction, ts_sub):
# Function that tests the area between subsequent downcasts to exclude data across fronts etc.

    #new array to work with
    ts_tmp = ts_sub.copy(deep=True)

    #These lines could be a problem if a full profile is cut
    ts_tmp = ts_tmp.where(density_bool, drop=True)
    ts_tmp = ts_tmp.where(ts_tmp.conductivity>=0, drop=True)

    #Apply the function profile_area to each pair of profiles
    dn_area = np.nan*np.zeros(len(profile_bins_all))
    dn_area = downcast_area(ts_tmp.load(), profile_bins_all, direction, dn_area)

    dn_mean = np.nanmean(dn_area)
    dn_std = np.nanstd(dn_area)
    dn_cutoff = dn_mean + dn_stdev*dn_std

    area_bad = np.abs(dn_area)>dn_cutoff

    ts_tmp.close()
    
    return dn_area, area_bad
	
def TS_apply(alphatau, fn, bad_profiles, ts):
# Function to apply the thermal lag correction to the original dataset

    alpha, tau = alphatau
    alpha = alpha / 1e3

    coefa = 4 * fn * alpha * tau / (1 + 4 * fn * tau)
    if coefa == 0:
        coefb = 0
    else:
        coefb = 1 - 2 * coefa / alpha
    b = np.array([1, -1]) * coefa
    a = np.array([1, coefb])

    ts_tmp = ts.copy(deep=True)
    
    #Remove the salinity profiles identified as bad
    for bad in bad_profiles:
        ts_tmp = ts_tmp.where(~(ts_tmp.profile_index==bad), drop=False)
    
    ts_tmp = ts_tmp.where(np.isfinite(ts_tmp.temperature + 
                                      ts_tmp.conductivityClean), drop=False)

    #Apply the filter and calculate the corrected fields
    ind = ~np.isnan(ts_tmp.temperature)
    x0 = ts_tmp.temperature.where(ind, drop=True)
    x0 = signal.lfilter(b, a, x0)

    ts_tmp['temperature_adjusted'] =  ts_tmp.temperature.copy()
    ts_tmp['temperature_adjusted'][ind] =  ts_tmp.temperature_adjusted[ind] - x0
    ts_tmp = ts_tmp.where(ts_tmp.conductivityClean>=0, drop=False)
    ts_tmp['salinity_adjusted'] = ts_tmp.salinity.copy()
    ts_tmp['salinity_adjusted'].values = seawater.eos80.salt(
        ts_tmp.conductivityClean / seawater.constants.c3515 * 10, 
        ts_tmp.temperature_adjusted, 
        ts_tmp.pressure
    )
    ts_tmp['density_adjusted'] = ts_tmp.density.copy()
    ts_tmp['density_adjusted'].values = seawater.eos80.dens(
        ts_tmp.salinity_adjusted, 
        ts_tmp.temperature_adjusted, 
        ts_tmp.pressure
    )
    
    ts_final = ts_tmp
    ts_tmp.close()
    print(f'alpha = {alpha}, tau = {tau}')
    
    return ts_final, x0

############################
# Functions from Jody for thermal lag correction:
def get_error(ds, sal, tbins, indbins):
    """
    get_error(ds, sal, tbins, indbins)
    Get the up/down changes in salinity, binned by profile.
    Parameters
    ----------
    ds : xr.DataSet
        Must have *temperature* and *profile_index* variables.  This is the timeseries data set we are 
        correcting
    sal : array
        Array of the salinity we are trying to correct.  This is passed in separately so that a 
        corrected salinity can be computed before putting into ds.  *sal* must have the same
        number of elements as *ds.temperature*
    tbins: array
        Array of temperatures to bin the salinities on.  Usually from a mean temperature depth profile
        of the data being corrected.  Note this is best spaced in depth space rather than 
        temperature space so that regions of high temperature gradients are not over represented in the error
    indbins: array
        unique profile indices to bin on.  
    Returns
    -------
    ss : array
        [tbins, indbins] sized array of the binned salinity
    err : array
        [tbins, indbins-1] sized array of the salinity difference between bins, normalized
        by the standard deviation at each depth, and the sign changed so that the up profiles 
        and down profiles have the same positive err sign on average.  _Most_ of these 
        should be greater than zero
    totalerr : array
        [tbins] length array with the mean of err across all profiles.  
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ss = np.histogram2d(ds.temperature, ds.profile_index, [tbins, indbins], weights=sal)[0] / np.histogram2d(ds.temperature, ds.profile_index, [tbins, indbins])[0]
    err = np.diff(ss, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        err = err / np.nanstd(ss, axis=1)[:, np.newaxis]
    err[:, 1::2] = -err[: , 1::2]
    if np.nanmean(np.nanmean(err,axis=0)) < 0:
        err = -err
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        totalerr = np.nanmean(err, axis=1).sum()
    
    return ss, err, totalerr

def correct_sal(ds, fn, alpha, tau):
    a = 4 * fn * alpha * tau / (1 + 4*fn*tau)
    b = 1 - 2 * a / alpha
    aa = [1, b]
    bb = [a, -a]
    tempcorr = ds.temperature.values.copy()
    tempcell = ds.temperature.values.copy()
    good = ~np.isnan(tempcell)
    tempcorr[good] = signal.lfilter(bb, aa, ds.temperature.values[good])
    tempcell = tempcell - tempcorr
    # tempcell = ds.temperature
    sal = sw.salt(10 * ds.conductivity / sw.constants.c3515, tempcell, ds.pressure)
    return sal


def correct_sal_temp_dens(ds, fn, alpha, tau):
    a = 4 * fn * alpha * tau / (1 + 4*fn*tau)
    b = 1 - 2 * a / alpha
    aa = [1, b]
    bb = [a, -a]
    tempcorr = ds.temperature.values.copy()
    tempcell = ds.temperature.values.copy()
    good = ~np.isnan(tempcell)
    tempcorr[good] = signal.lfilter(bb, aa, ds.temperature.values[good])
    tempcell = tempcell - tempcorr
    # tempcell = ds.temperature
    sal = sw.salt(10 * ds.conductivity / sw.constants.c3515, tempcell, ds.pressure)
    dens = sw.dens(sal, tempcell, ds.pressure)
    return sal, tempcell, dens

# A copy of the make_gridfiles function from pyglider. Testing to see if it fixes filtering issues by loading instead of opening.

import logging
_log = logging.getLogger(__name__)
import os
import pyglider.utils as utils

def make_gridfiles_load(
    inname, outdir, deploymentyaml, *, fnamesuffix='', dz=1, starttime='1970-01-01'
):
    """
    Turn a timeseries netCDF file into a vertically gridded netCDF.

    Parameters
    ----------
    inname : str or Path
        netcdf file to break into profiles

    outdir : str or Path
        directory to place profiles

    deploymentyaml : str or Path
        location of deployment yaml file for the netCDF file.  This should
        be the same yaml file that was used to make the timeseries file.

    dz : float, default = 1
        Vertical grid spacing in meters.

    Returns
    -------
    outname : str
        Name of gridded netCDF file. The gridded netCDF file has coordinates of
        'depth' and 'profile', so each variable is gridded in depth bins and by
        profile number.  Each profile has a time, latitude, and longitude.
    """
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    deployment = utils._get_deployment(deploymentyaml)

    profile_meta = deployment['profile_variables']

    ds = xr.load_dataset(inname, decode_times=True)
    ds = ds.where(ds.time > np.datetime64(starttime), drop=True)
    _log.info(f'Working on: {inname}')
    _log.debug(str(ds))
    _log.debug(str(ds.time[0]))
    _log.debug(str(ds.time[-1]))

    profiles = np.unique(ds.profile_index)
    profiles = [p for p in profiles if (~np.isnan(p) and not (p % 1) and (p > 0))]
    profile_bins = np.hstack((np.array(profiles) - 0.5, [profiles[-1] + 0.5]))
    _log.debug(profile_bins)
    Nprofiles = len(profiles)
    _log.info(f'Nprofiles {Nprofiles}')
    depth_bins = np.arange(0, 1100.1, dz)
    depths = depth_bins[:-1] + 0.5
    xdimname = 'time'
    dsout = xr.Dataset(
        coords={'depth': ('depth', depths), 'profile': (xdimname, profiles)}
    )
    dsout['depth'].attrs = {
        'units': 'm',
        'long_name': 'Depth',
        'standard_name': 'depth',
        'positive': 'down',
        'coverage_content_type': 'coordinate',
        'comment': 'center of depth bins',
    }

    ds['time_1970'] = ds.temperature.copy()
    ds['time_1970'].values = ds.time.values.astype(np.float64)
    for td in ('time_1970', 'longitude', 'latitude'):
        good = np.where(~np.isnan(ds[td]) & (ds['profile_index'] % 1 == 0))[0]
        dat, xedges, binnumber = stats.binned_statistic(
            ds['profile_index'].values[good],
            ds[td].values[good],
            statistic='mean',
            bins=[profile_bins],
        )
        if td == 'time_1970':
            td = 'time'
            dat = dat.astype('timedelta64[ns]') + np.datetime64('1970-01-01T00:00:00')
        _log.info(f'{td} {len(dat)}')
        dsout[td] = (('time'), dat, ds[td].attrs)
    ds.drop('time_1970')
    good = np.where(~np.isnan(ds['time']) & (ds['profile_index'] % 1 == 0))[0]
    _log.info(f'Done times! {len(dat)}')
    dsout['profile_time_start'] = ((xdimname), dat, profile_meta['profile_time_start'])
    dsout['profile_time_end'] = ((xdimname), dat, profile_meta['profile_time_end'])

    for k in ds.keys():
        if k in ['time', 'profile', 'longitude', 'latitude', 'depth'] or 'time' in k:
            continue
        _log.info('Gridding %s', k)
        good = np.where(~np.isnan(ds[k]) & (ds['profile_index'] % 1 == 0))[0]
        if len(good) <= 0:
            continue
        if 'average_method' in ds[k].attrs:
            average_method = ds[k].attrs['average_method']
            ds[k].attrs['processing'] = (
                f'Using average method {average_method} for '
                f'variable {k} following deployment yaml.'
            )
            if average_method == 'geometric mean':
                average_method = stats.gmean
                ds[k].attrs['processing'] += (
                    ' Using geometric mean implementation ' 'scipy.stats.gmean'
                )
        else:
            average_method = 'mean'
        dat, xedges, yedges, binnumber = stats.binned_statistic_2d(
            ds['profile_index'].values[good],
            ds['depth'].values[good],
            values=ds[k].values[good],
            statistic=average_method,
            bins=[profile_bins, depth_bins],
        )

        _log.debug(f'dat{np.shape(dat)}')
        dsout[k] = (('depth', xdimname), dat.T, ds[k].attrs)

        # fill gaps in data:
        dsout[k].values = utils.gappy_fill_vertical(dsout[k].values)

    # fix u and v, because they should really not be gridded...
    if ('water_velocity_eastward' in dsout.keys()) and ('u' in profile_meta.keys()):
        _log.debug(str(ds.water_velocity_eastward))
        dsout['u'] = dsout.water_velocity_eastward.mean(axis=0)
        dsout['u'].attrs = profile_meta['u']
        dsout['v'] = dsout.water_velocity_northward.mean(axis=0)
        dsout['v'].attrs = profile_meta['v']
        dsout = dsout.drop(['water_velocity_eastward', 'water_velocity_northward'])
    dsout.attrs = ds.attrs
    dsout.attrs.pop('cdm_data_type')
    # fix to be ISO parsable:
    if len(dsout.attrs['deployment_start']) > 18:
        dsout.attrs['deployment_start'] = dsout.attrs['deployment_start'][:19]
        dsout.attrs['deployment_end'] = dsout.attrs['deployment_end'][:19]
        dsout.attrs['time_coverage_start'] = dsout.attrs['time_coverage_start'][:19]
        dsout.attrs['time_coverage_end'] = dsout.attrs['time_coverage_end'][:19]
    # fix standard_name so they don't overlap!
    try:
        dsout['waypoint_latitude'].attrs.pop('standard_name')
        dsout['waypoint_longitude'].attrs.pop('standard_name')
        dsout['profile_time_start'].attrs.pop('standard_name')
        dsout['profile_time_end'].attrs.pop('standard_name')
    except:
        pass
    # set some attributes for cf guidance
    # see H.6.2. Profiles along a single trajectory
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/aphs06.html
    dsout.attrs['featureType'] = 'trajectoryProfile'
    dsout['profile'].attrs['cf_role'] = 'profile_id'
    dsout['mission_number'] = np.int32(1)
    dsout['mission_number'].attrs['cf_role'] = 'trajectory_id'
    dsout = dsout.set_coords(['latitude', 'longitude', 'time'])
    for k in dsout:
        if k in ['profile', 'depth', 'latitude', 'longitude', 'time', 'mission_number']:
            dsout[k].attrs['coverage_content_type'] = 'coordinate'
        else:
            dsout[k].attrs['coverage_content_type'] = 'physicalMeasurement'

    outname = outdir + '/' + ds.attrs['deployment_name'] + '_grid' + fnamesuffix + '.nc'
    _log.info('Writing %s', outname)
    # timeunits = 'nanoseconds since 1970-01-01T00:00:00Z'
    dsout.to_netcdf(
        outname,
        encoding={
            'time': {
                'units': 'seconds since 1970-01-01T00:00:00Z',
                '_FillValue': np.nan,
                'calendar': 'gregorian',
                'dtype': 'float64',
            }
        },
    )
    _log.info('Done gridding')

    return outname