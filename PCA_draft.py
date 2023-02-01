# -*- coding: utf-8 -*-


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
from sklearn import preprocessing

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

a1s = numpy.array([])
cl1s = numpy.array([])

a2s = numpy.array([])
cl2s = numpy.array([])

a3s = numpy.array([])
cl3s = numpy.array([])

for eco in range(36):

    Domain = Ecoregions[eco]

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
    
    X = pandas.DataFrame({'BA':value,
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
    
    X = X.drop(X.columns[[0]], axis=1)
    X = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=3)
    pca.fit(X)
    # print(pca.components_)
    
    
    PC1 = pca.components_[0]
    PC2 = pca.components_[1]
    PC3 = pca.components_[2]
    
    
    S = numpy.dot(pca.components_, X.T)
    
    numpy.shape(S)
    
    
    dataset = pandas.DataFrame({'Basal Area':value,
                                'PC1': S[0],
                                'PC2': S[1],
                                'PC3': S[2]})
        
    
    dataset.shape
    
    dataset = dataset.fillna(method='ffill')
    dataset[dataset == numpy.inf] = numpy.nan
    dataset.fillna(dataset.mean(), inplace=True)
    
    
    y = dataset[dataset.columns[0]].values  # dataset['Basal Area']
    
    
    
    R2results_for_clim_vars = numpy.array([])
    
    for j in range(3):
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
    
    
    print(numpy.argmax(R2results_for_clim_vars), round(numpy.max(R2results_for_clim_vars)*100, 1), '%')
    
    
    cl1 = numpy.argmax(R2results_for_clim_vars)
    
    a1 = round(numpy.max(R2results_for_clim_vars)*100, 1)
    ###########################################################################
    #########   Regression BA vs 2 clim vars : ######################################
    ###########################################################################
    
    y = dataset[dataset.columns[0]].values  # dataset['Basal Area']
    
    R2results_for_clim_vars2 = numpy.array([])
    
    
    for j in range(3):
        if j == cl1:
            R2results_for_clim_vars2 = numpy.append(R2results_for_clim_vars2, -1)
            continue
        X = dataset.iloc[:, [cl1+1,j+1]].values
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
       
        regressor = LinearRegression()  
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)
        
        R2result2 = r2_score(y_test, y_pred)
        
        print(round(R2result2*100, 3))
    #    print(j+1, round(R2result2*100, 2))
        R2results_for_clim_vars2 = numpy.append(R2results_for_clim_vars2, R2result2)
    
    
    print(round(numpy.max(R2results_for_clim_vars2)*100, 2), '%')
    
    
    cl2 = numpy.argmax(R2results_for_clim_vars2)
    
    a2 = round(numpy.max(R2results_for_clim_vars2)*100, 2)
    
    ###########################################################################
    #########   Regression BA vs 3 clim vars : ######################################
    ###########################################################################
    
    R2results_for_clim_vars3 = numpy.array([])
    
    
    for j in range(3):
        if (j == cl1 or j == cl2):
            R2results_for_clim_vars3 = numpy.append(R2results_for_clim_vars3, -1)
            continue
        X = dataset.iloc[:, [cl1+1, cl2+1, j+1]].values
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
       
        regressor = LinearRegression()  
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)
        
        R2result3 = r2_score(y_test, y_pred)
        
        print(round(R2result3*100, 3))
    #    print(j+1, round(R2result3*100, 2))
        R2results_for_clim_vars3 = numpy.append(R2results_for_clim_vars3, R2result3)
    
    
    print(round(numpy.max(R2results_for_clim_vars3)*100, 2), '%')
    
    cl3 = numpy.argmax(R2results_for_clim_vars3)
    
    a3 = round(numpy.max(R2results_for_clim_vars3)*100, 2)
    

    a1s = numpy.append(a1s, a1)
    cl1s = numpy.append(cl1s, cl1)
    
    a2s = numpy.append(a2s, a2)
    cl2s = numpy.append(cl2s, cl2)

    a3s = numpy.append(a3s, a3)
    cl3s = numpy.append(cl3s, a3)
    
    
    
for k in range(36):
    print(Ecoregions[k], a1s[k], '%', a2s[k], '%', a3s[k], '%' )
