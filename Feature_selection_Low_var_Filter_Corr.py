# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:38:08 2020

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
from sklearn.ensemble import RandomForestRegressor

path = 'C:/Users/olga/Desktop/US_forest/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df.columns

df = df.drop('Unnamed: 0', 1)
df.columns



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

imps = numpy.array([])
impchars = numpy.array([])

for i in range(36):

    Domain = Ecoregions[i]
    df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
    df.columns
    df = df.drop('Unnamed: 0', 1)
    df.columns
    data0 = df.values  
    NOBSERV0 = numpy.size(data0, 0) 
    df2 = df[numpy.isin(ecoreg, Domain)]
    data = df2.values
    
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
    
    
    dataset = pandas.DataFrame({'Basal Area':barea.astype(float),
                                'Annual Mean Temperature (BIO1)': bio1.astype(float),
                                'Mean Diurnal Range (BIO2)': bio2.astype(float),
                                'Isothermality (BIO3)': bio3.astype(float),
                                'Temperature Seasonality (BIO4)': bio4.astype(float),
                                'Max Temperature of Warmest Month (BIO5)': bio5.astype(float),
                                'Min Temperature of Coldest Month (BIO6)': bio6.astype(float),
                                'Temperature Annual Range (BIO7)': bio7.astype(float),
                                'Mean Temperature of Wettest Quarter (BIO8)': bio8.astype(float),
                                'Mean Temperature of Driest Quarter (BIO9)': bio9.astype(float),
                                'Mean Temperature of Warmest Quarter (BIO10)': bio10.astype(float),
                                'Mean Temperature of Coldest Quarter (BIO11)': bio11.astype(float),
                                'Annual Precipitation (BIO12)': bio12.astype(float),
                                'Precipitation of Wettest Month (BIO13)': bio13.astype(float),
                                'Precipitation of Driest Month (BIO14)': bio14.astype(float),
                                'Precipitation Seasonality (BIO15)': bio15.astype(float),
                                'Precipitation of Wettest Quarter (BIO16)': bio16.astype(float),
                                'Precipitation of Driest Quarter (BIO17)': bio17.astype(float),
                                'Precipitation of Warmest Quarte (BIO18)': bio18.astype(float),
                                'Precipitation of Coldest Quarter (BIO19)': bio19.astype(float)})
    
    #dataset.isnull().sum()/len(dataset)*100
    #dataset.var()
    
    numeric = dataset[['Annual Mean Temperature (BIO1)', 'Mean Diurnal Range (BIO2)',
                       'Isothermality (BIO3)', 'Temperature Seasonality (BIO4)',
                       'Max Temperature of Warmest Month (BIO5)', 'Min Temperature of Coldest Month (BIO6)',
                       'Temperature Annual Range (BIO7)', 'Mean Temperature of Wettest Quarter (BIO8)',
                       'Mean Temperature of Driest Quarter (BIO9)', 'Mean Temperature of Warmest Quarter (BIO10)',
                       'Mean Temperature of Coldest Quarter (BIO11)', 'Annual Precipitation (BIO12)',
                       'Precipitation of Wettest Month (BIO13)', 'Precipitation of Driest Month (BIO14)',
                       'Precipitation Seasonality (BIO15)', 'Precipitation of Wettest Quarter (BIO16)',
                       'Precipitation of Driest Quarter (BIO17)', 'Precipitation of Warmest Quarte (BIO18)',
                       'Precipitation of Coldest Quarter (BIO19)']]
    
    var = numeric.var() #average of the squared deviations from the mean
    
    numeric = numeric.columns
    
    variable = numpy.array([])
    # Low variance filter:
    for i in range(0, len(var)):
          s = var[i]
          if s.astype(float)<0.1:   #setting the threshold as 10%
             variable = numpy.append(variable , numeric[i])
           
    arr = numpy.array([])
    arr = numpy.append(arr, 'Basal Area')
    arr = numpy.append(arr, variable)
    
    df = dataset.drop(arr, 1)
    
    
    correlation = df.corr()
    high_corr = numpy.array([]) # only highly correlated vars
    col = df.columns
    
    for c1 in col:
      for c2 in col:
          if c1 != c2 and c2 not in high_corr and correlation[c1][c2] > 0.7:
              print(c1, c2, correlation[c1][c2])
              high_corr = numpy.append(high_corr, (c1))
          
    high_corr = numpy.unique(high_corr)   
    
    df = df.drop(high_corr, 1)
    
    df.columns
    model = RandomForestRegressor(random_state=1, max_depth=10)
    df=pandas.get_dummies(df)
    labels = numpy.array(dataset['Basal Area'])
    features  = df
    feature_list = list(features.columns)
    model.fit(features, labels)
    
    features = df.columns
    importances = model.feature_importances_
    indices = numpy.argsort(importances)[-5:]  # top 10 features
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    ind = numpy.argmax(importances)
    imps = numpy.append(imps, round(importances[ind], 2))
    impchars = numpy.append(impchars, features[ind])
    

for j in range(36):  
     print(imps[j])