
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
period='late'
print('analyzing {} period'.format(period))

# ## set up

# In[2]:


## read in surface pressure
era=xr.open_dataset('/sn1/dessler/era/era_ceres_200003-201812_90x45.nc')
era=fixdims(era)
ps=era['sp'].mean(dim='time')/100. # average surface pressure
ps['lat']=np.round(ps.lat,2)


# In[3]:


def avg(ivar):
    """avg(ivar): returns the average annual cycle of the xarray ivar"""
    ovar=ivar.groupby('time.month').mean(dim='time')
    ovar=ovar.rename({'month':'time'})
    return ovar


# In[4]:


def wvsat(t):
	"""wvsat(t): return water vapor saturation mixing ratio (g/kg)
	over liquid for t > 273 K and over ice for t < 273 K
    t is an xarray DataArray, which includes pressures on coordinate 'level'"""

# 	ind1=np.ma.where(t >= 273.15)
# 	ind2=np.ma.where(t < 273.15)

    # air > freezing
    # Source Hyland, R. W. and A. Wexler, Formulations for the Thermodynamic Properties of the saturated 
    # Phases of H2O from 173.15K to 473.15K, ASHRAE Trans, 89(2A), 500-519, 1983.
	p1 = np.exp(  -0.58002206e4 / t + 0.13914993e1 - 0.48640239e-1 * t 
	+ 0.41764768e-4 * t**2. - 0.14452093e-7 * t**3. 
	+ 0.65459673e1 * np.log(t) ) / 100.

# Source : Goff-Gratch, Smithsonian Meteorological Tables, 5th edition, p. 350, 1984
	ei0	   = 6.1071		  # mbar
	T0	   = 273.16		  # freezing point in K
	p2 = 10.**(-9.09718 * (T0 / t - 1.) - 3.56654 * np.log10(T0 / t) + 0.876793 * (1. - t / T0) + np.log10(ei0))
    
	Psat=t*0
	Psat=Psat.where(t > 273,p2)
	Psat=Psat.where(t < 273,p1)

	return Psat/t.level*18/29.  # return to kg/kg


# In[5]:


def getKernel(kernel):
    """read kernels"""

    k1=xr.open_dataarray('/home/dessler/kernels/'+kernel+'_gfdl_std_90x45.nc',decode_times=False)
    k1=fixdims(k1)
    k1['lat']=np.round(k1.lat,2)
    k1['time']=np.arange(12)+1
    
    k1.name=kernel
    
    k1=k1.fillna(0) # set kernel to zero for missing points

    # if albedo kernel, we are done
    if kernel[-1] == 'a': return k1
    
    # for T kernel, extract surface T kernel 
    if kernel[-1] == 't':
        k1=k1.sortby('level',ascending=False)
        ## extract surface kernel from 1050-hPa pressure level
        ts=k1.sel(level=1050)
        ts=ts.drop('level')
    
    # get rid of bottom layer and layers above 100 hPa 
    k1=k1.sel(level=slice(1025,100))
    
    # set kernel to zero above tropopause
    for pp in k1.lat:
        if np.abs(pp) < 30: 
            cutoff = 99
        else:
            cutoff=np.interp(np.abs(float(pp)),[30.,90.],[125.,275.])

        x1=k1.sel(lat=pp).where(cutoff < k1.level,0) # set pressures below cutoff to zero
        k1.loc[{'lat' : pp}]=x1
        
    ## units of kernel are per 100 hPa, so we have to adjust this
    ## to account for actual thickness of layer 
    lev1=np.array(k1.level)
    thickness=(lev1[:-2]-lev1[2:])/2
    thickness=np.append((lev1[0]-lev1[1])/2,thickness)
    thickness=np.append(thickness,(lev1[-2]-lev1[-1])/2)
    thickness = np.abs(thickness/100.)
    thickness=xr.DataArray(thickness,dims='level',coords={'level':k1.level})
    k1 *= thickness
    
    ## zero the kernels for regions below the surface
    ## uses average surface pressure from reanalysis
    for pp in k1.level:
        x1=k1.sel(level=pp).where(ps > pp,0)
        k1.loc[{'level' : pp}]=x1
        
    if kernel[-1] == 'q': 
        return -k1
    else:
        return -k1,-ts


# In[6]:

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

# In[8]:

print('running ...')

## read models: PI control as the base period, and 4xCO2
clim=readModelPI()
modout=readModel4xCO2(period)

# index for kernels
kernelInd=np.array(modout.time.dt.month-1)

# In[9]:

## calculate averages required for the water vapor feedback
## this includes average rh and dhus/dt
avgt=avg(modout['ta']);avgq=avg(modout['hus']) # avg. annual cycle
husSat=wvsat(avgt)
rh=avgq/husSat

# limit RH to reasonable values 1-100%
rh=rh.where(rh > 0.01,0.01)
rh=rh.where(rh < 1,1.0)

eta=np.log(wvsat(avgt+1)*rh)-np.log(husSat*rh)


#### reference temperatures
## surface temperature

tempref2d=modout.tas-clim.tas
tempref=gavg(tempref2d)

## 500-hPa temperature
t500=modout.ta.sel(level=500,drop=True)-clim.ta.sel(level=500,drop=True)
sum=0

# In[10]:


#### read water vapor and temperature kernels
# all-sky
kt,kts=getKernel('lw_t')

# clear-sky
kclrt,kclrts=getKernel('lwclr_t')

#### planck feedback
# uniform warming of the surface and atmosphere
x1=kts.isel(time=kernelInd)
x1['time']=modout.time
x1=x1*tempref2d

x2=kt.isel(time=kernelInd)
x2['time']=modout.time
x2=x2*tempref2d
x2=x2.sum(dim='level')

feedback=np.average(gavg(x1+x2))/np.average(tempref)
print('planck = {:.2f} '.format(feedback))
sum += feedback

# In[11]:


fluxes=(x1+x2).to_dataset(name='planck')


# In[12]:


### lapse-rate feedback

# lapse-rate feedback
# differential warming of the surface and atmosphere
x1=kts.isel(time=kernelInd)
x1['time']=modout.time

dt=modout.tas-clim.tas-tempref2d
x1=x1*dt

x2=kt.isel(time=kernelInd)
x2['time']=modout.time
dt=modout.ta-clim.ta-tempref2d
x2=x2*dt
x2=x2.sum(dim='level')

feedback=np.average(gavg(x1+x2))/np.average(tempref)
print('lapse rate = {:.2f} '.format(feedback))
sum += feedback

fluxes=xr.merge([fluxes,(x1+x2).to_dataset(name='lapserate')])

## calculate temperature masks
x1=(kclrt-kt).isel(time=kernelInd) # atmospheric component
x1['time']=modout.time
maskT=((modout.ta-clim.ta)*x1).sum(dim='level')

x1=(kclrts-kts).isel(time=kernelInd) # surface component
x1['time']=modout.time
maskT += (modout.tas-clim.tas)*x1


# In[14]:


### Planck-RH feedback
kqlw=getKernel('lw_q');kqsw=-getKernel('sw_q') # sum of LW and SW kernels
kclrqlw=getKernel('lwclr_q');kclrqsw=-getKernel('swclr_q') # sum of LW and SW kernels
kq=kqlw+kqsw;kclrq=kclrqlw+kclrqsw

# planck feedback, constant RH
# uniform warming of the surface and atmosphere
x1=kts.isel(time=kernelInd)
x1['time']=modout.time
dt=tempref2d
x1=x1*dt

x2=(kt+kq).isel(time=kernelInd)
x2['time']=modout.time
x2=x2*dt
x2=x2.sum(dim='level')

feedback=np.average(gavg(x1+x2))/np.average(tempref)
print('planck RH = {:.2f} '.format(feedback))

# In[15]:


fluxes=xr.merge([fluxes,(x1+x2).to_dataset(name='planckRH')])


# In[16]:


### lapse-rate-RH feedback
# lapse-rate feedback, constant RH
# differential warming of the surface and atmosphere
x1=kts.isel(time=kernelInd)
x1['time']=modout.time
dt=modout['tas']-clim.tas-tempref2d
x1=x1*dt

x2=(kt+kq).isel(time=kernelInd)
x2['time']=modout.time
dt=modout.ta-clim.ta-tempref2d
x2=x2*dt
x2=x2.sum(dim='level')

feedback=np.average(gavg(x1+x2))/np.average(tempref)
print('lapse rate RH = {:.2f} '.format(feedback))



# In[17]:


fluxes=xr.merge([fluxes,(x1+x2).to_dataset(name='lapserateRH')])


# In[18]:


## water vapor feedback
x1=kq.isel(time=kernelInd)
x1['time']=modout.time
xeta=eta.isel(time=kernelInd)
xeta['time']=modout.time

dq=(np.log(modout.hus)-np.log(clim.hus))/xeta 

x1=x1*dq
x1=x1.sum(dim='level')

feedback=np.average(gavg(x1))/np.average(tempref)
print('water vapor = {:.2f} '.format(feedback))
sum += feedback

fluxes=xr.merge([fluxes,(x1).to_dataset(name='watervapor')])

## calculate water vapor mask
## dq = scaled water vapor (from above)
x1=(kclrqlw-kqlw).isel(time=kernelInd)
x1['time']=modout.time
maskqlw=dq*x1
maskqlw=maskqlw.sum(dim='level')

x1=(kclrqsw-kqsw).isel(time=kernelInd)
x1['time']=modout.time
maskqsw=dq*x1
maskqsw=maskqsw.sum(dim='level')


# In[19]:


## water vapor feedback, constant RH
x1=kq.isel(time=kernelInd)
x1['time']=modout.time
xeta=eta.isel(time=kernelInd)
xeta['time']=modout.time

dq=(np.log(modout.hus)-np.log(clim.hus))/xeta-(modout.ta-clim.ta)

x1=x1*dq
x1=x1.sum(dim='level')

feedback=np.average(gavg(x1))/np.average(tempref)
print('water vapor RH = {:.2f} '.format(feedback))

# In[20]:


fluxes=xr.merge([fluxes,(x1).to_dataset(name='watervaporRH')])


# In[21]:

## albedo feedback

ka=getKernel('sw_a').squeeze()
kclra=getKernel('swclr_a').squeeze()

# albedo feedback
x1=ka.isel(time=kernelInd)
x1['time']=modout.time

modout=modout.assign(down=modout.srads-modout.sradsu)
modout['sradsu']=modout.sradsu.where(modout.sradsu < -1,0)
modout['down']=modout.down.where(modout.down > 0,1)
clim=clim.assign(down=clim.srads-clim.sradsu)
clim['sradsu']=clim.sradsu.where(clim.sradsu < -1,0)
clim['down']=clim.down.where(clim.down > 0,1)

## calculate albedo
albedo=-modout.sradsu/modout.down
albedo=albedo.where(albedo > 0,0)

calbedo=-clim.sradsu/clim.down
calbedo=calbedo.where(calbedo > 0,0)

da=(albedo-calbedo).squeeze()*1e2 # convert from fraction to percent

x1=x1*da

feedback=np.average(gavg(x1))/np.average(tempref)
print('albedo = {:.2f} '.format(feedback))
sum += feedback

# In[22]:

fluxes=xr.merge([fluxes,(x1).to_dataset(name='albedo')])

## calculate albedo mask
## da = albedo change (from above)
x1=(kclra-ka).isel(time=kernelInd)
x1['time']=modout.time
maska=da*x1


# In[23]:

###### cloud feedback components
### short wave
# calculate clear-sky flux
toa_net_clr = modout.rsutcs-clim.rsutcs-(forcing.rsutcs)

# calculate all-sky flux
toa_net = modout.rsut-clim.rsut-forcing.rsut

# calculate CRF
crf=toa_net - toa_net_clr

# calculate cloud feedback
# I'm using here the Soden et al. equation (25), which defines CRF as clear minus all-sky
# mask is also clear minus all, and it is added to the CRF
mask= maskqsw + maska 
x1=crf + mask

feedback=np.average(gavg(x1))/np.average(tempref)
print('cloud SW = {:.2f} '.format(feedback))
sum += feedback

fluxes=xr.merge([fluxes,(x1).to_dataset(name='cloudsw')])

### long wave
# calculate clear-sky flux
toa_net_clr = modout.rlutcs-clim.rlutcs-(forcing.rlutcs)

# calculate all-sky flux
toa_net = modout.rlut-clim.rlut-forcing.rlut

# calculate CRF
crf=toa_net - toa_net_clr

# calculate cloud feedback
mask=maskT + maskqlw
x1=crf + mask

feedback=np.average(gavg(x1))/np.average(tempref)
print('cloud LW = {:.2f} '.format(feedback))
sum += feedback

fluxes=xr.merge([fluxes,(x1).to_dataset(name='cloudlw')])

# ## cloud feedback 

# # calculate clear-sky flux
# toa_net_clr = (modout.rsutcs+modout.rlutcs-(clim.rsutcs+clim.rlutcs))-(forcing.rsutcs+forcing.rlutcs)

# # calculate all-sky flux
# toa_net = (modout.rsut+modout.rlut-(clim.rsut+clim.rlut))-forcing.f

# # calculate CRF
# crf=toa_net - toa_net_clr

# # calculate cloud feedback
# # I'm using here the Soden et al. equation (25), which defines CRF as clear minus all-sky
# # mask is also clear minus all, and it is added to the CRF
# mask=maskT + maskqlw + maskqsw + maska #+ (forcing.rlutcs+forcing.rsutcs-forcing.f) forcing already added
# x1=crf + mask

# feedback=np.average(gavg(x1))/np.average(tempref)
# print('cloud = {:.2f} '.format(feedback))
# sum += feedback

# In[24]:

## total feedback 

# calculate all-sky flux
toa_net = (modout.rsut+modout.rlut-(clim.rsut+clim.rlut))-forcing.f

# calculate total sensitivity as dR/dT
feedback=np.average(gavg(toa_net))/np.average(tempref)
print('total = {:.2f} '.format(feedback))

fluxes=xr.merge([fluxes,toa_net.to_dataset(name='total')])

# plot(tempref,flux,'o')
# plot(np.linspace(-0.4,0.5),np.linspace(-0.4,0.5)*ecs.b[1])

#### compare these results to previous calculation

print('sum = {:.2f} W/m^2/K'.format(sum))
print('actual lambda = {:.2f} W/m^2/K'.format(feedback))

# plot(tempref,x1,'o')
# plot(np.linspace(-0.4,0.5),np.linspace(-0.4,0.5)*cloud.b[1])


# In[28]:

fluxes=xr.merge([fluxes,tempref2d.to_dataset(name='temperature'),t500.to_dataset(name='t500')])
fluxes=fluxes.drop(['level'])


# In[31]:
fluxes.to_netcdf('/sn1/dessler/MPI-feedback/abrupt4xCO2fluxes-clim-{}.nc'.format(period))
print('done!')