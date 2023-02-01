# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:01 2020

@author: olga
"""


import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import random
# import docx

from numpy import random
from numpy import mean
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


 

###########################################################################
#########   Load data: ######################################
###########################################################################

NSIMS = 1000

Humid_200 = ['211', '212', '221', '222', '223', '231', '232', '234', '242',
       '251', '255', '261', '262', '263',  'M211', 'M221', 'M223', 'M231',
       'M242', 'M261', 'M262']

Dry_300 = ['313', '315', '321', '322',
       '331', '332', '341', '342', 'M313', 'M331', 'M332', 'M333', 'M334',
       'M341']

Humid_Tropical_400 = ['411']

AllUSA = ['211', '212', '221', '222', '223', '231', '232', '234', '242',
       '251', '255', '261', '262', '263',  'M211', 'M221', 'M223', 'M231',
       'M242', 'M261', 'M262', '313', '315', '321', '322',
       '331', '332', '341', '342', 'M313', 'M331', 'M332', 'M333', 'M334',
       'M341', '411']

Domain = Humid_Tropical_400
    
# path = 'C:/Users/Olga Rumyantseva/Desktop/US_forest/' # home comp.
path = 'C:/Users/olga/Desktop/US_forest/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.drop('Unnamed: 0', 1)
# df.columns
df = df.dropna()

data0 = df.values
NOBSERV0 = numpy.size(data0, 0)
ecoreg = numpy.array([str(data0[k,26]) for k in range(NOBSERV0)])  # Eco region

df2 = df[numpy.isin(ecoreg, Domain)]
data = df2.values
NOBSERV = numpy.size(data, 0)

patch = data[:,21]   # Plot ID 
year = numpy.array([int(data[k,22]) for k in range(NOBSERV)]) # Year 
biomass = data[:,23] # Biomass
barea = data[:,24]   # Basal Area
shTol = data[:,25]   # Basal Area 

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
    

    
   
###########################################################################
#########   Regression BA vs 1 clim var : ######################################
###########################################################################


dataset = pandas.DataFrame({'Basal Area': barea,
                            'AnnualMeanTemperature': bio1,
                            'MeanDiurnalRange': bio2,
                            'Isothermality': bio3,
                            'TemperatureSeasonality': bio4,
                            'MaxTemperatureofWarmestMonth': bio5,
                            'MinTemperatureofColdestMonth': bio6,
                            'TemperatureAnnualRange': bio7,
                            'MeanTemperatureofWettestQuarter': bio8,
                            'MeanTemperatureofDriestQuarter': bio9,
                            'MeanTemperatureofWarmestQuarter': bio10,
                            'MeanTemperatureofColdestQuarter': bio11,
                            'AnnualPrecipitation': bio12,
                            'PrecipitationofWettestMonth': bio13,
                            'PrecipitationofDriestMonth': bio14,
                            'PrecipitationSeasonality': bio15,
                            'PrecipitationofWettestQuarter': bio16,
                            'PrecipitationofDriestQuarter': bio17,
                            'PrecipitationofWarmestQuarte': bio18,
                            'PrecipitationofColdestQuarter': bio19})
    
climvars = ['Annual Mean Temperature (BIO1)',
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

climvarscolors = ['temperature', #1
                  'variability',   #2
                  'variability',   #3
                  'variability',   #4
                  'temperature', #5     
                  'temperature', #6   
                  'variability',   #7
                  'temperature',  #8
                  'temperature',  #9  
                  'temperature',  #10 
                  'temperature',  #11
                  'precipitation',    #12
                  'precipitation',    #13
                  'precipitation',    #14  
                  'precipitation',    #15
                  'precipitation',    #16
                  'precipitation',    #17  
                  'precipitation',    #18 
                  'precipitation']    #19



dataset.shape

y = dataset[dataset.columns[0]].values  # dataset['Basal Area']
R2results_for_clim_vars = numpy.array([])

for j in range(19):
    X = dataset[dataset.columns[j+1]].values
    # split 80% of the data to training set while 
    # 20% of the data to test set:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train= X_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result = r2_score(y_test, y_pred)
    print(round(R2result*100, 2))
#    print(j+1, round(R2result*100, 2))
    R2results_for_clim_vars = numpy.append(R2results_for_clim_vars, R2result)


clvar1 = numpy.argmax(R2results_for_clim_vars)
climvars[clvar1]

###########################################################################
#########   Regression BA vs 2 clim vars : ######################################
###########################################################################

y = dataset[dataset.columns[0]].values  # dataset['Basal Area']

R2results_for_clim_vars2 = numpy.array([])

for j in range(19):
    if j == clvar1:
        R2results_for_clim_vars2 = numpy.append(R2results_for_clim_vars2, -100)
        continue
    X = dataset.iloc[:, [clvar1+1,j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result2 = r2_score(y_test, y_pred)
    
    print(round(R2result2*100, 3))
#    print(j+1, round(R2result2*100, 2))
    R2results_for_clim_vars2 = numpy.append(R2results_for_clim_vars2, R2result2)

clvar2 = numpy.argmax(R2results_for_clim_vars2)
climvars[clvar2]

###########################################################################
#########   Regression BA vs 3 clim vars : ######################################
###########################################################################

R2results_for_clim_vars3 = numpy.array([])


for j in range(19):
    if (j == clvar1 or j == clvar2):
        R2results_for_clim_vars3 = numpy.append(R2results_for_clim_vars3, -100)
        continue
    X = dataset.iloc[:, [clvar1+1, clvar2+1, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result3 = r2_score(y_test, y_pred)
    
    print(round(R2result3*100, 3))
#    print(j+1, round(R2result3*100, 2))
    R2results_for_clim_vars3 = numpy.append(R2results_for_clim_vars3, R2result3)




clvar3 = numpy.argmax(R2results_for_clim_vars3)


###########################################################################
#########   Regression BA vs 4 clim vars : ######################################
###########################################################################

R2results_for_clim_vars4 = numpy.array([])


for j in range(19):
    if (j == clvar1 or j == clvar2 or j == clvar3):
        R2results_for_clim_vars4 = numpy.append(R2results_for_clim_vars4, -100)
        continue
    X = dataset.iloc[:, [clvar1+1, clvar2+1, clvar3+1, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result4 = r2_score(y_test, y_pred)
    
    print(round(R2result4*100, 2))
#    print(j+1, round(R2result4*100, 2))
    R2results_for_clim_vars4 = numpy.append(R2results_for_clim_vars4, R2result4)



clvar4 = numpy.argmax(R2results_for_clim_vars4)

###########################################################################
#########   Regression BA vs 5 clim vars : ######################################
###########################################################################

R2results_for_clim_vars5 = numpy.array([])


for j in range(19):
    if (j == clvar1 or j == clvar2 or j == clvar3 or j == clvar4):
        R2results_for_clim_vars5 = numpy.append(R2results_for_clim_vars5, -100)
        continue
    X = dataset.iloc[:, [clvar1+1, clvar2+1, clvar3+1, clvar4+1, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result5 = r2_score(y_test, y_pred)
    
    print(round(R2result5*100, 2))
#    print(j+1, round(R2result4*100, 2))
    R2results_for_clim_vars5 = numpy.append(R2results_for_clim_vars5, R2result5)



clvar5 = numpy.argmax(R2results_for_clim_vars5)


for j in range(19):
    print(numpy.round(R2results_for_clim_vars[j]*100,2))
    
for j in range(19):
    print(numpy.round(R2results_for_clim_vars2[j]*100,2))
   
for j in range(19):
    print(numpy.round(R2results_for_clim_vars3[j]*100,2))
   
for j in range(19):
    print(numpy.round(R2results_for_clim_vars4[j]*100,2))
   
for j in range(19):
    print(numpy.round(R2results_for_clim_vars5[j]*100,2))
       

print(climvars[numpy.argmax(R2results_for_clim_vars)])
print(climvars[numpy.argmax(R2results_for_clim_vars2)])
print(climvars[numpy.argmax(R2results_for_clim_vars3)])
print(climvars[numpy.argmax(R2results_for_clim_vars4)])
print(climvars[numpy.argmax(R2results_for_clim_vars5)])

print(climvarscolors[numpy.argmax(R2results_for_clim_vars)])
print(climvarscolors[numpy.argmax(R2results_for_clim_vars2)])
print(climvarscolors[numpy.argmax(R2results_for_clim_vars3)])
print(climvarscolors[numpy.argmax(R2results_for_clim_vars4)])
print(climvarscolors[numpy.argmax(R2results_for_clim_vars5)])


print(round(numpy.max(R2results_for_clim_vars)*100, 1))
print(round(numpy.max(R2results_for_clim_vars2)*100, 1))
print(round(numpy.max(R2results_for_clim_vars3)*100, 1))
print(round(numpy.max(R2results_for_clim_vars4)*100, 1))
print(round(numpy.max(R2results_for_clim_vars5)*100, 1))
