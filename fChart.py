import pandas as pd
import numpy as np
import os as os
import scipy.optimize as optimize

def readSolarData(filename):
	"""
	Import location solar and surface data
	Insolation = month-average daily horizontal insolation (kWh/m2/day)
	Albedo = surface albedo
	"""
	return pd.read_csv(filename)

def addStandardColumns(df, solar_cons = 1367):
	"""
	This function calculates a number of parameters that are based only on the day of the year (not location/collector information)
	units of solar_cons = W/m2
	The typical day is a representative average day in each month in terms of solar insolation.  The values below have been recommended by source 4.
	The declination is the angle made by the line joining the centers of the sun and the earth with the projection of this line on the equatorial plane.  
	It can vary from 23.45° on June 21 to -23.45° on December 21.  It is 0° on March 21 and September 22, the two equinoxes. 
	These values were obtained using Equation 7.6 from source 2. 
	ISC' accounts for the variation of distance between the sun and the earth throughout the year.  Therefore, the solar constant is assumed to be 1367 W/mS but this value is not constant year round.  
	These values were obtained using Equation 3.1.1 from source 1. 
	"""
	df['n'] = [17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]
	df['declination'] = 23.45 * np.sin(np.deg2rad(360 / 365 * (284 + df['n'])))
	df['Isc_prime'] = solar_cons * (1.00011 + \
									(0.034221 * np.cos(np.deg2rad(1 * ((df['n'] - 1) * 360 / 365)))) + \
									(0.001280 * np.sin(np.deg2rad(1 * ((df['n'] - 1) * 360 / 365)))) + \
									(0.000719 * np.cos(np.deg2rad(2 * ((df['n'] - 1) * 360 / 365)))) + \
									(0.000077 * np.sin(np.deg2rad(2 * ((df['n'] - 1) * 360 / 365)))))
	return df

def addCalcSolarVars(df, latitude):
	"""
	This function calculates the sunset hour angle, extraterrestrial insolation on a horizontal surface, clearness index, and diffuse fraction, given the location latitude.
	The sunset hour angle for the month is the number of degrees that the earth must rotate before the sun will be directly over the line of longitude from sunset.  
	The E.T. horizontal insolation is the amount of radiation that would hit a horizontal collector on the earth's surface if there was no atmosphere.  In other words, this is the amount of radiation that would hit a horizontal collector in space.   
	These values were obtained using Equation 7.43 from source 2. 
	The clearness index is the ratio of the average horizontal insolation to the extraterrestrial insolation on a horizontal surface.  
	It can vary from 0 to 1.  In essence, this index accounts for various weather patterns concerning solar radiation collection (i.e. clouds).
	These values were obtained using Equation 7.42 from source 2. 
	"""
	df['sunset_hour_angle'] = np.rad2deg(np.arccos(-np.tan(np.deg2rad(latitude)) * \
												   np.tan(np.deg2rad(df['declination']))))
	df['ET_insol'] = (24 / np.pi) * \
					 (df['Isc_prime'] / 1000) * \
					 ((np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(df['declination'])) * np.sin(np.deg2rad(df['sunset_hour_angle']))) + \
	 				 (np.deg2rad(df['sunset_hour_angle']) * np.sin(np.deg2rad(latitude)) * np.sin(np.deg2rad(df['declination']))))
	df['clearness'] = df['insolation_horizontal'] / df['ET_insol']
	# Calculate diffuse fraction
	df['diffuse_fraction'] = (df['insolation_horizontal'] * (1.39 - (4.027 * df['clearness']) + (5.531 * (df['clearness'] ** 2)) - (3.108 * (df['clearness'] ** 3)))) / df['insolation_horizontal']
	return df

def addCalcMethodVars(df, latitude, azimuth, slope):
	"""
	This function takes an intermediate dataframe as well as location latitude and collector slope and azimuth angles to calculate a number of parameters required for
	estimating the collected solar energy on the tilted collector.
	"""
	df['a'] = 0.409 + (0.5016 * np.sin(np.deg2rad(df['sunset_hour_angle'] - 60)))
	df['a_prime'] = df['a'] - df['diffuse_fraction']
	df['b'] = 0.6609 - (0.4767 * np.sin(np.deg2rad(df['sunset_hour_angle'] - 60)))
	df['d'] = np.sin(np.deg2rad(df['sunset_hour_angle'])) - np.deg2rad(df['sunset_hour_angle'] * np.cos(np.deg2rad(df['sunset_hour_angle'])))
	df['A'] = np.cos(np.deg2rad(slope)) + (np.tan(np.deg2rad(latitude)) * np.cos(np.deg2rad(azimuth)) * np.sin(np.deg2rad(slope)))
	df['B'] = (np.cos(np.deg2rad(df['sunset_hour_angle'])) * np.cos(np.deg2rad(slope))) + (np.tan(np.deg2rad(df['declination'])) * np.sin(np.deg2rad(slope)) * np.cos(np.deg2rad(azimuth)))
	df['C'] = np.sin(np.deg2rad(slope)) * np.sin(np.deg2rad(azimuth)) / np.cos(np.deg2rad(latitude))
	df['omega_sr_abs'] = np.absolute(
		np.minimum(
			df['sunset_hour_angle'], 
			np.rad2deg(np.arccos(((df['A'] * df['B']) + (df['C'] * np.sqrt((df['A'] ** 2) - (df['B'] ** 2) + (df['C'] ** 2)))) / ((df['A'] ** 2) + (df['C'] ** 2))))
			)
		)
	df['omega_sr'] = np.where(
		((df['A'] > 0.0) & (df['B'] > 0)) | (df['A'] >= df['B']), 
		-df['omega_sr_abs'], 
		df['omega_sr_abs']
		)
	df['omega_ss_abs'] = np.absolute(
		np.minimum(
			df['sunset_hour_angle'], 
			np.rad2deg(np.arccos(((df['A'] * df['B']) - (df['C'] * np.sqrt((df['A'] ** 2) - (df['B'] ** 2) + (df['C'] ** 2)))) / ((df['A'] ** 2) + (df['C'] ** 2))))
			)
		)
	df['omega_ss'] = np.where(
		((df['A'] > 0.0) & (df['B'] > 0)) | (df['A'] >= df['B']), 
		df['omega_ss_abs'], 
		-df['omega_ss_abs']
		)
	df['D'] = np.where(
		df['omega_ss'] >= df['omega_sr'],
		np.maximum(0.0,
			((1 / (2 * df['d'])) * \
		     (np.deg2rad(((df['b'] * df['A'] / 2) - (df['a_prime'] * df['B'])) * (df['omega_ss'] - df['omega_sr'])) + \
		     			 (((df['a_prime'] * df['A']) - (df['b'] * df['B'])) * (np.sin(np.deg2rad(df['omega_ss'])) - np.sin(np.deg2rad(df['omega_sr'])))) - \
						 (df['a_prime'] * df['C'] * (np.cos(np.deg2rad(df['omega_ss'])) - np.cos(np.deg2rad(df['omega_sr'])))) + \
						 ((df['b'] * df['A'] / 2) * ((np.sin(np.deg2rad(df['omega_ss'])) * np.cos(np.deg2rad(df['omega_ss']))) - \
						  		   					 (np.sin(np.deg2rad(df['omega_sr'])) * np.cos(np.deg2rad(df['omega_sr']))))) + \
						 ((df['b'] * df['C'] / 2) * (((np.sin(np.deg2rad(df['omega_ss']))) ** 2) - ((np.sin(np.deg2rad(df['omega_sr']))) ** 2)))
						 )
		     )
			),
		np.maximum(0.0,
			((1 / (2 * df['d'])) * \
			 (np.deg2rad(((df['b'] * df['A'] / 2) - (df['a_prime'] * df['B'])) * \
						 (df['omega_ss'] - (-df['sunset_hour_angle']))) + \
			  (((df['a_prime'] * df['A']) - (df['b'] * df['B'])) * \
			   (np.sin(np.deg2rad(df['omega_ss'])) - np.sin(np.deg2rad(-df['sunset_hour_angle'])))) - \
			  (df['a_prime'] * df['C'] * (np.cos(np.deg2rad(df['omega_ss'])) - np.cos(np.deg2rad(-df['sunset_hour_angle'])))) + \
			  ((df['b'] * df['A'] / 2) * ((np.sin(np.deg2rad(df['omega_ss'])) * \
						  		   		   np.cos(np.deg2rad(df['omega_ss']))) - \
						  		   		  (np.sin(np.deg2rad(-df['sunset_hour_angle'])) * np.cos(np.deg2rad(-df['sunset_hour_angle']))))) + \
			  ((df['b'] * df['C'] / 2) * (((np.sin(np.deg2rad(df['omega_ss']))) ** 2) - ((np.sin(np.deg2rad(-df['sunset_hour_angle']))) ** 2)))
			  )
			 ) + \
			((1 / (2 * df['d'])) * (np.deg2rad(((df['b'] * df['A'] / 2) - (df['a_prime'] * df['B'])) * (df['sunset_hour_angle'] - df['omega_sr'])) + \
						  		 	(((df['a_prime'] * df['A']) - (df['b'] * df['B'])) * (np.sin(np.deg2rad(df['sunset_hour_angle'])) - np.sin(np.deg2rad(df['omega_sr'])))) - \
						  		 	(df['a_prime'] * df['C'] * (np.cos(np.deg2rad(df['sunset_hour_angle'])) - np.cos(np.deg2rad(df['omega_sr'])))) + \
						  		 	((df['b'] * df['A'] / 2) * ((np.sin(np.deg2rad(df['sunset_hour_angle'])) * np.cos(np.deg2rad(df['sunset_hour_angle']))) - \
						  		 	                            (np.sin(np.deg2rad(df['omega_sr'])) * np.cos(np.deg2rad(df['omega_sr']))))) + \
						  		 	((df['b'] * df['C'] / 2) * (((np.sin(np.deg2rad(df['sunset_hour_angle']))) ** 2) - ((np.sin(np.deg2rad(df['omega_sr']))) ** 2)))
						  		 	)
			)
			)
		)
	df['r_bar'] = df['D'] + \
				  (df['diffuse_fraction'] * \
				   (1 + np.cos(np.deg2rad(slope))) / 2) + \
				  (df['albedo'] * \
				   (1 - np.cos(np.deg2rad(slope))) / 2)
	return df

def calcTotalInsolation(latitude, slope, azimuth):
	"""
	This function returns a large dataframe containing the starting data, all of the intermediate calculation columns, as well
	as the final column ('insolation_tilted'), which is the month-average daily solar energy collected (in kWh/m2/day).
	"""
	data = readSolarData('Inputs/LocationSolarData.csv')
	df = addStandardColumns(data)
	df = addCalcSolarVars(df, latitude)
	df = addCalcMethodVars(df, latitude, azimuth, slope)
	df['insolation_tilted'] = df['r_bar'] * df['insolation_horizontal']
	return df

def calcAnnualWeightedAveInsolation(latitude, slope, azimuth):
	"""
	This function takes the location latitude and collector slope and azimuth angles to calculate the annual weighted average
	collected solar insolation (in kWh/m2/day).
	"""
	df = calcTotalInsolation(latitude, slope, azimuth)
	return np.dot(
		np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
		df['insolation_tilted']
		) / 365.0

def calcAWAIforOptim(slopeaz, latitude):
	"""
	This function does the exact same calculation as calcAnnualWeightedAveInsolation, but it returns a negative result for use 
	in the optimization part of getOptimSlopeAz.
	"""
	df = calcTotalInsolation(latitude, slopeaz[0], slopeaz[1])
	return np.dot(
		np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
		df['insolation_tilted']
		) / -365.0

def getOptimSlopeAz(latitude):
	"""
	This function takes a location latitude and maximizes the weighted average solar insolation by optimizing the collector slope and azimuth angle.
	The returned result is a list containing the optimized slope [0] and azimuth [1] angles, respectively.
	"""
	result = optimize.minimize(calcAWAIforOptim,
							   [30.0, 5.0],
							   (latitude),
							   method='nelder-mead',
                               tol=1e-6,
                               options={'disp': False})
	return result.x














