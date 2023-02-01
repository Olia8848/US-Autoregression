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

Domain = Ecoregions

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



X = pandas.DataFrame({'bio1': bio1,
                            'bio2': bio2,
                            'bio3': bio3,
                            'bio4': bio4,
                            'bio5': bio5,
                            'bio6': bio6,
                            'bio7': bio7,
                            'bio8': bio8,
                            'bio9': bio9,
                            'bio10': bio10,
                            'bio11': bio11,
                            'bio12': bio12,
                            'bio13': bio13,
                            'bio14': bio14,
                            'bio15': bio15,
                            'bio16': bio16,
                            'bio17': bio17,
                            'bio18': bio18,
                            'bio19': bio19})

X.to_csv(path + 'BA_clim_vars_for_PCA_USA.csv')


pca = PCA(n_components=3)
pca.fit(X)

