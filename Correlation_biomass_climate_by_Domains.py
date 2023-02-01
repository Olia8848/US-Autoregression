# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:03:04 2019

@author: olga
"""

import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import random
from numpy import random
from numpy import mean
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'

# in the data below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.dropna()

df = df.drop('Unnamed: 0', 1)
# df.columns

data0 = df.values  
NOBSERV0 = numpy.size(data0, 0) 


ecoreg = numpy.array([str(data0[k,26]) for k in range(NOBSERV0)])  # Eco region
Ecoregions  = numpy.unique(ecoreg)

###########################################################################
#########    ######################################
###########################################################################

BioVarsAllNames = ['Annual Mean Temperature (BIO1)',
'Mean Diurnal Range (BIO2)',  
'Isothermality (BIO3)',
'Temperature Seasonality (BIO4)',
'Max Temperature of Warmest Month (BIO5)',     
'Min Temperature of Coldest Month (BIO6)',   
'Temperature Annual Range (BIO7)',
'Mean Temperature of Wettest Quarter (BIO8)',
'Mean Temperature of Driest Quarter (BIO9)',   
'Mean Temperature of Warmest Quarter (BIO10)', 
'Mean Temperature of Coldest Quarter (BIO11)', 
'Annual Precipitation (BIO12)', 
'Precipitation of Wettest Month (BIO13)',
'Precipitation of Driest Month (BIO14)',   
'Precipitation Seasonality (BIO15)',
'Precipitation of Wettest Quarter (BIO16)', 
'Precipitation of Driest Quarter (BIO17)',   
'Precipitation of Warmest Quarter (BIO18)',  
'Precipitation of Coldest Quarter (BIO19)']

#for i in range(19):
#    print(BioVarsAllNames[i])

###########################################################################
#########   corr. for each domain: ######################################
###########################################################################

Humid_200 = ['211', '212', '221', '222', '223', '231', '232', '234', '242',
       '251', '255', '261', '262', '263',  'M211', 'M221', 'M223', 'M231',
       'M242', 'M261', 'M262']

Dry_300 = ['313', '315', '321', '322',
       '331', '332', '341', '342', 'M313', 'M331', 'M332', 'M333', 'M334',
       'M341']

Humid_Tropical_400 = ['411']

corrs = pandas.DataFrame()

for i in range(36):
    Domain = Ecoregions[i]
    
    df2 = df[numpy.isin(ecoreg, Domain)]
    data = df2.values
    NOBSERV = numpy.size(data, 0)
    
     # inside the domain:   
    bio1 = data[:,2]  # AnnualMeanTemperature
    bio2 = data[:,3]  # MeanDiurnalRange
    bio3 = data[:,4]  # Isothermality
    bio4 = data[:,5]  # TemperatureSeasonality
    bio5 = data[:,6]  # MaxTemperatureofWarmestMonth
    bio6 = data[:,7]  # MinTemperatureofColdestMonth
    bio7 = data[:,8]  # TemperatureAnnualRange
    bio8 = data[:,9]  # MeanTemperatureofWettestQuarter
    bio9 = data[:,10]  # MeanTemperatureofDriestQuarter
    bio10 = data[:,11]  # MeanTemperatureofWarmestQuarter
    bio11 = data[:,12]  # MeanTemperatureofColdestQuarter
    bio12 = data[:,13]  # AnnualPrecipitation
    bio13 = data[:,14]  # PrecipitationofWettestMonth
    bio14 = data[:,15]  # PrecipitationofDriestMonth
    bio15 = data[:,16]  # PrecipitationSeasonality
    bio16 = data[:,17]  # PrecipitationofWettestQuarter
    bio17 = data[:,18]  # PrecipitationofDriestQuarter
    bio18 = data[:,19]  # PrecipitationofWarmestQuarte
    bio19 = data[:,20]  # PrecipitationofColdestQuarter
    
    
    biomass = data[:,23] # Biomass
    barea = data[:,24]   # Basal Area
    
    value = barea # this is what we consider now: biomass 
    
    
    c1 = numpy.corrcoef(value.astype(float), bio1.astype(float), rowvar=False)[0][1]
    c2 = numpy.corrcoef(value.astype(float), bio2.astype(float), rowvar=False)[0][1]
    c3 = numpy.corrcoef(value.astype(float), bio3.astype(float), rowvar=False)[0][1]
    c4 = numpy.corrcoef(value.astype(float), bio4.astype(float), rowvar=False)[0][1]
    c5 = numpy.corrcoef(value.astype(float), bio5.astype(float), rowvar=False)[0][1]
    c6 = numpy.corrcoef(value.astype(float), bio6.astype(float), rowvar=False)[0][1]
    c7 = numpy.corrcoef(value.astype(float), bio7.astype(float), rowvar=False)[0][1]
    c8 = numpy.corrcoef(value.astype(float), bio8.astype(float), rowvar=False)[0][1]
    c9 = numpy.corrcoef(value.astype(float), bio9.astype(float), rowvar=False)[0][1]
    c10 = numpy.corrcoef(value.astype(float), bio10.astype(float), rowvar=False)[0][1]
    c11 = numpy.corrcoef(value.astype(float), bio11.astype(float), rowvar=False)[0][1]
    c12 = numpy.corrcoef(value.astype(float), bio12.astype(float), rowvar=False)[0][1]
    c13 = numpy.corrcoef(value.astype(float), bio13.astype(float), rowvar=False)[0][1]
    c14 = numpy.corrcoef(value.astype(float), bio14.astype(float), rowvar=False)[0][1]
    c15 = numpy.corrcoef(value.astype(float), bio15.astype(float), rowvar=False)[0][1]
    c16 = numpy.corrcoef(value.astype(float), bio16.astype(float), rowvar=False)[0][1]
    c17 = numpy.corrcoef(value.astype(float), bio17.astype(float), rowvar=False)[0][1]
    c18 = numpy.corrcoef(value.astype(float), bio18.astype(float), rowvar=False)[0][1]
    c19 = numpy.corrcoef(value.astype(float), bio19.astype(float), rowvar=False)[0][1]
    
    climcorrBA = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19] 
    # for i in range(19):
    #    print(round(climcorrBA[i],2))    
    
    corrs[Ecoregions[i,]] = climcorrBA


corrs.to_csv(path +'hhhhhhhhhh.csv', index=False)