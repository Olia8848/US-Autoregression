# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:32:34 2019

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
from sklearn.decomposition import PCA

path = 'C:/Users/olga/Desktop/US_forest/'
path1 = 'C:/Users/olga/Desktop/US Clim Extract/for_PCAs_BA_Clim_vars/'
    
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.dropna()
df = df.drop('Unnamed: 0', 1)
data0 = df.values  
NOBSERV0 = numpy.size(data0, 0) 

ecoreg = numpy.array([str(data0[k,26]) for k in range(NOBSERV0)])  # Eco region
Ecoregions  = numpy.unique(ecoreg)

Humid_200 = ['211', '212', '221', '222', '223', '231', '232', '234', '242',
       '251', '255', '261', '262', '263',  'M211', 'M221', 'M223', 'M231',
       'M242', 'M261', 'M262']
Dry_300 = ['313', '315', '321', '322',
       '331', '332', '341', '342', 'M313', 'M331', 'M332', 'M333', 'M334',
       'M341']
Humid_Tropical_400 = ['411']


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
    
    value = barea 
    
    X = pandas.DataFrame({'Basal Area':value,
                            'BIO1':bio1,
                            'BIO2':bio2,  
                            'BIO3':bio3,
                            'BIO4':bio4,
                            'BIO5':bio5,     
                            'BIO6':bio6,   
                            'BIO7':bio7,
                            'BIO8':bio8,
                            'BIO9':bio9,   
                            'BIO10':bio10, 
                            'BIO11':bio11, 
                            'BIO12':bio12, 
                            'BIO13':bio13,
                            'BIO14':bio14,   
                            'BIO15':bio15,
                            'BIO16':bio16, 
                            'BIO17':bio17,   
                            'BIO18':bio18,  
                            'BIO19':bio19})
    
    X.to_csv(path1 + 'BA_clim_vars_for_PCA_eco'+ str(i) +'.csv')



for i in range(36):
    print(i)



# 'Basal Area':value,
#X = pandas.DataFrame({'AnnualMeanTemperature': bio1,
#                            'MeanDiurnalRange': bio2,
#                           'Isothermality': bio3,
#                           'TemperatureSeasonality': bio4,
#                            'MaxTemperatureofWarmestMonth': bio5,
#                            'MinTemperatureofColdestMonth': bio6,
#                            'TemperatureAnnualRange': bio7,
#                            'MeanTemperatureofWettestQuarter': bio8,
#                            'MeanTemperatureofDriestQuarter': bio9,
#                            'MeanTemperatureofWarmestQuarter': bio10,
#                            'MeanTemperatureofColdestQuarter': bio11,
#                            'AnnualPrecipitation': bio12,
#                            'PrecipitationofWettestMonth': bio13,
#                            'PrecipitationofDriestMonth': bio14,
#                            'PrecipitationSeasonality': bio15,
#                            'PrecipitationofWettestQuarter': bio16,
#                            'PrecipitationofDriestQuarter': bio17,
#                            'PrecipitationofWarmestQuarte': bio18,
#                            'PrecipitationofColdestQuarter': bio19})
    
    X = pandas.DataFrame({'Basal Area':value,
                            'Annual Mean Temperature (BIO1)':bio1,
                            'Mean Diurnal Range (BIO2)':bio2,  
                            'Isothermality (BIO3)':bio3,
                            'Temperature Seasonality (BIO4)':bio4,
                            'Max Temperature of Warmest Month (BIO5)':bio5,     
                            'Min Temperature of Coldest Month (BIO6)':bio6,   
                            'Temperature Annual Range (BIO7)':bio7,
                            'Mean Temperature of Wettest Quarter (BIO8)':bio8,
                            'Mean Temperature of Driest Quarter (BIO9)':bio9,   
                            'Mean Temperature of Warmest Quarter (BIO10)':bio10, 
                            'Mean Temperature of Coldest Quarter (BIO11)':bio11, 
                            'Annual Precipitation (BIO12)':bio12, 
                            'Precipitation of Wettest Month (BIO13)':bio13,
                            'Precipitation of Driest Month (BIO14)':bio14,   
                            'Precipitation Seasonality (BIO15)':bio15,
                            'Precipitation of Wettest Quarter (BIO16)':bio16, 
                            'Precipitation of Driest Quarter (BIO17)':bio17,   
                            'Precipitation of Warmest Quarter (BIO18)':bio18,  
                            'Precipitation of Coldest Quarter (BIO19)':bio19})

