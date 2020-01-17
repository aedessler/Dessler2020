# coding: utf-8

# # calculate feedbacks in MPI ensemble 

# In[1]:


import numpy as np
import os,ols,glob,cftime

from scipy import stats
import xarray as xr
import pandas as pd
from data import check
from xradd import *

print('running!')

# ## set up

# In[2]:

### the following code defines different functions depending on the type of model run that is being analyzed
histx=False # set true if analyzing model historical run of 1850-2005
pictrlx=False # set true if analyzing control run
a4xCO2=True # set true if analyzing 4xCO2 run

def procModel(dat1):
    """make some slight changes to the data structure of the model"""

    datax=fixdims(dat1)
    if np.max(datax.level) > 2000: datax['level'] /= 1e2 # pressure in hPa
    # rename for consistency with CMIP5
    datax=datax.rename({'q':'hus','t':'ta','temp2':'tas','srad0':'rsut','trad0':'rlut','sraf0':'rsutcs',
                        'traf0':'rlutcs'})
    datax['lat']=np.round(datax.lat,2)
    
    return datax

def readModelPI():
    """read piControl run"""
    
    # read climatology
    # cdo ymonmean pictrl.nc piclim.nc
    datax=xr.open_dataset('/sn1/dessler/MPI-feedback/piclim.nc')
    datax['time']=cftime.num2date(np.arange(len(datax.time))*30,'days since 1850-01-30',calendar='360_day')
    print('** reading control run **')
        
    return procModel(datax)

def readModel4xCO2(period1):
    """read 4xCO2 run"""
    
    # read climatology
    datax=xr.open_dataset('/sn1/dessler/MPI-feedback/abrupt4xCO2-clim-{}.nc'.format(period1))
    datax['time']=cftime.num2date(np.arange(len(datax.time))*30,'days since 1850-01-30',calendar='360_day')
    print('** reading 4xCO2 run **')
        
    return procModel(datax)

# In[7]:

## read forcing
## from 1% run
forcing=xr.open_dataset('/home/dessler/energyBalance/1percent/global/forcing_1pct_90x45.nc')
forcing['lat']=np.round(forcing.lat,2)
forcing=forcing.isel(time=slice(130,150)).mean(dim='time') # average years 130-150
forcing=forcing.rename({'srad0':'rsut','trad0':'rlut','sraf0':'rsutcs','traf0':'rlutcs'})

# ## start calculation 

## read models: PI control as the base period
clim=readModelPI()
clim=clim.assign(t500=clim.ta.sel(level=500))
clim=clim.drop('srads,sradsu,hus,ta'.split(','))


# In[8]:
for period in 'early,mid,late'.split(','):

    print('creating input file: {}'.format(period))
    
    steps={'early':'300/419','mid':'1800/2399','late':'25369/31368'}
    
    # this code calls cdo to create seasonal cycle averages over early, mid, or late periods
#     os.system('cdo seltimestep,{} abrupt4xCO2.nc abrupt4xCO2-{}.nc'.format(steps[period],period))
#     os.system('cdo ymonmean abrupt4xCO2-{0}.nc abrupt4xCO2-clim-{0}.nc'.format(period))

    print('running ...')

    ## read 4xCO2 model:
    modout=readModel4xCO2(period)

    ## drop fields you don't need
    modout=modout.assign(t500=modout.ta.sel(level=500))
    modout=modout.drop('srads,sradsu,hus,ta'.split(','))

    # subtract off piCtrl run
    modout -= clim

    # subtract off forcing
    for ii in 'rlut,rsut,rlutcs,rsutcs'.split(','):
        modout[ii] -= forcing[ii]

    modout.to_netcdf('/sn1/dessler/MPI-feedback/abrupt4xCO2-clim-{}-crf.nc'.format(period))
    
#     os.system('rm -f abrupt4xCO2-{0}.nc abrupt4xCO2-clim-{0}.nc'.format(period))

