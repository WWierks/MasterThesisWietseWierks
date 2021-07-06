# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:49:34 2021

@author: Wietse Wierks
"""
#%% Importing packages
# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime
import pathlib

import sys
before = {str(m) for m in sys.modules}
# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas as pd


# coordinate systems
import pyproj 

# statistics
import statsmodels.api as sm
import statsmodels.multivariate.pca
import statsmodels.tsa.seasonal


# plotting
import bokeh.io
import bokeh.plotting
import bokeh.tile_providers
import bokeh.palettes

# this package gives a lot of warnings about incompatibility with 3.1 
# replace 
import windrose
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
matplotlib.projections.register_projection(windrose.WindroseAxes)
import cmocean.cm

# displaying things
from ipywidgets import Image
import IPython.display

# add the top level package
sys.path.append('..')
import lib.models
import lib.psmsl_hydro 

from functools import reduce
from scipy.misc import derivative

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from trendfit.models import LinearBrokenTrendFourier
from trendfit.bootstrap import block_ar_wild

#%% Set path and coordinate system
# Some coordinate systems
WEBMERCATOR = pyproj.Proj('epsg:3857')
WGS84 = pyproj.Proj('epsg:4326')


bokeh.io.output_notebook()

# we're using matplotlib for polar plots (non-interactive)
#%matplotlib inline


# use local data, make sure you updated the datasets with the makefiles
local = True

# directory to the sealevel 
src_dir = pathlib.Path(r'C:\Hydro\Masterthesis\Python\Spyder').expanduser()
psmsl_data_dir = src_dir

#%% define data location (local)

psmsl_urls_local = {
    'met_monthly': psmsl_data_dir / 'met_monthly.zip',
    'rlr_monthly': psmsl_data_dir / 'rlr_monthly.zip',
    'rlr_annual': psmsl_data_dir / 'rlr_annual.zip'
}

psmsl_urls = psmsl_urls_local
default_dataset_name = 'rlr_annual'

#%% Select stations (insert loop here!!)

main_stations_path = src_dir / 'stations_list_loop.txt'
main_stations = pd.read_csv(main_stations_path, delimiter=';')
main_stations = main_stations.set_index('id')
print(main_stations)


main_stations_idx = list(main_stations.index) #list main station(s) id
main_stations_idx

zipfiles = {}


for dataset_name in psmsl_urls:
    zf = zipfile.ZipFile(psmsl_urls_local[dataset_name])
    zipfiles[dataset_name] = zf

# this list contains a table of 
# station ID, latitude, longitude, station name, coastline code, station code, and quality flag
csvtext = zipfiles[dataset_name].read('{}/filelist.txt'.format(dataset_name))

stations = pd.read_csv(
    io.BytesIO(csvtext), 
    sep=';',
    names=('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')
stations

selected_stations = pd.merge(main_stations, stations, left_index=True, right_index=True)
# set the main stations
selected_stations['name'] = selected_stations['name_x']
selected_stations

url_names = {
    'datum': '{dataset_name}/RLR_info/{id}.txt',
    'diagram': '{dataset_name}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'rlr_monthly': '{dataset_name}/data/{id}.rlrdata',
    'rlr_annual': '{dataset_name}/data/{id}.rlrdata',
    'met_monthly': '{dataset_name}/data/{id}.metdata',
    'doc': '{dataset_name}/docu/{id}.txt',
    'contact': '{dataset_name}/docu/{id}_auth.txt'
}

def get_url(station, dataset_name):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset_name=dataset_name,
        id=station.name
    )
    url = url_names['url'].format(**info)
    return url

for dataset_name in psmsl_urls:
    # fill in the dataset parameter using the global dataset_name
    f = functools.partial(get_url, dataset_name=dataset_name)
    # compute the url for each station
    selected_stations[dataset_name] = selected_stations.apply(f, axis=1)
selected_stations


print(selected_stations)

u_file = pathlib.Path(r'C:\Hydro\Masterthesis\Python\wind\uwnd.10m.mon.mean.nc').expanduser()
v_file = pathlib.Path(r'C:\Hydro\Masterthesis\Python\wind\vwnd.10m.mon.mean.nc').expanduser()

df_with_wind = pd.DataFrame(np.arange(1890,2021,1), columns = ["Year"])
df_without_wind = pd.DataFrame(np.arange(1890,2021,1), columns = ["Year"])

#%% Loop
#def Loop_Stations(selected_stations):
    
for index, row in selected_stations.iterrows():  # Loop over rows in stations list

   def make_wind_df(lat_i, lon_i):
       """create a dataset for wind, for 1 latitude/longitude"""
       
       u_file = pathlib.Path(r'C:\Hydro\Masterthesis\Python\wind\uwnd.10m.mon.mean.nc').expanduser()
       v_file = pathlib.Path(r'C:\Hydro\Masterthesis\Python\wind\vwnd.10m.mon.mean.nc').expanduser()
   
       # open the 2 files
       ds_u = netCDF4.Dataset(u_file)
       ds_v = netCDF4.Dataset(v_file)
       
       if lon_i < 0:
           lon_i = 360 + lon_i
       
       # read lat,lon, time from 1 dataset
       lat, lon, time = ds_u.variables['lat'][:], ds_u.variables['lon'][:], ds_u.variables['time'][:]
       
       # check with the others
       lat_v, lon_v, time_v = ds_v.variables['lat'][:], ds_v.variables['lon'][:], ds_v.variables['time'][:]
       assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
       
       # convert to datetime
       # Now defaults to return cftime dates https://github.com/Unidata/cftime/issues/136
       # cftime dates are not recognized by pandas
       # in cftime < 1.2.1 there is a bug that this flag doesn't not function properly
       t = netCDF4.num2date(time, ds_u.variables['time'].units, only_use_cftime_datetimes=False)
       
       def find_closest(lat, lon, lat_i=lat_i, lon_i=lon_i):
           """lookup the index of the closest lat/lon"""
           Lon, Lat = np.meshgrid(lon, lat)
           idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
           Lat.ravel()[idx], Lon.ravel()[idx]
           [i, j] = np.unravel_index(idx, Lat.shape)
           return i, j
       
       # this is the index where we want our data
       i, j = find_closest(lat, lon)
       
       # get the u, v variables
       print('found point', lat[i], lon[j])
       u = ds_u.variables['uwnd'][:, i, j]
       v = ds_v.variables['vwnd'][:, i, j]
   
       # compute derived quantities
       speed = np.sqrt(u**2 + v**2)
       
       # compute direction in 0-2pi domain
       direction = np.mod(np.angle(u + v * 1j), 2*np.pi)
       
       # put everything in a dataframe
       wind_df = pd.DataFrame(data=dict(u=u, v=v, t=t, speed=speed, direction=direction))
       wind_df = wind_df.set_index('t')
     # return it
       return wind_df
   
   
   wind_df = make_wind_df(row["lat"],row["lon"]) #!!!!! Define lon lat !!
   
   # label set to xxxx-01-01 of the current year
   annual_wind_df = wind_df.resample('A', label='left', loffset=datetime.timedelta(days=1)).mean()
   annual_wind_df['speed'] = np.sqrt(annual_wind_df['u']**2 + annual_wind_df['v']**2)
   annual_wind_df['direction'] = np.mod(np.angle(annual_wind_df['u'] + annual_wind_df['v'] * 1j), 2*np.pi)
   
   # create a wide figure, showing 2 wind roses with some extra info
   fig = plt.figure(figsize=(13, 6))
   # we're creating 2 windroses, one boxplot
   ax = fig.add_subplot(1, 2, 1, projection='windrose')
   ax = windrose.WindroseAxes.from_ax(ax=ax)
   # from radians 0 east, ccw to 0 north cw, use meteo convention of "wind from" (270 - math degrees)
   # see for example: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv.html
   wind_direction_meteo = np.mod(270 - (360.0 * wind_df.direction / (2*np.pi)), 360)
   # create a box plot
   ax.box(wind_direction_meteo, wind_df.speed, bins=np.arange(0, 8, 1), cmap=cmocean.cm.speed)
   ax.legend(loc='best')
   
   # and a scatter showing the seasonal pattern (colored by month)
   ax = fig.add_subplot(1, 2, 2, 
       projection='polar',
       theta_direction=-1,
       theta_offset=np.pi/2.0
   )
   N = matplotlib.colors.Normalize(1, 12)
   months = [x.month for x in wind_df.index]
   sc = ax.scatter(
       # here we need radians, but again use math -> meteo conversion
       (np.pi + np.pi/2)-wind_df.direction, 
       wind_df.speed, 
       c=months, 
       cmap=cmocean.cm.phase, 
       vmin=0, 
       vmax=12,
       alpha=0.5,
       s=10,
       edgecolor='none'
   )
   _ = plt.colorbar(sc, ax=ax)
   _ = fig.suptitle('wind from '+str(row["name"])+', average/month\nspeed [m/s] and direction [deg]')
   
   
   plt.savefig("Windrose_"+str(row["name"])+".png") 
   
   fig, ax = plt.subplots(figsize=(13, 8))
   wind_df['speed'].plot(alpha=0.2, ax=ax)
   annual_wind_df['speed'].plot(alpha=0.5, ax=ax)
   ax.set_xlabel('year')
   ax.set_ylabel('average wind speed [m/s]')
   ax.set_title('Average wind speed for '+str(row["name"])+ ' station')
   ax.grid()
   
   plt.savefig("Average_speed_"+str(row["name"])+".png"); 
   
   # get data for all stations
   for dataset_name in psmsl_urls:
       f = functools.partial(
           lib.psmsl_hydro.get_data_with_wind, 
           dataset_name=dataset_name, 
           wind_df=wind_df, 
           annual_wind_df=annual_wind_df,
           zipfiles=zipfiles,
           url_names=url_names
       )
       # look up the data for each station
      
       row[dataset_name] = f(row)
       
       
       
 
   #     selected_stations[dataset_name] = [f(station) for _, station in selected_stations.iterrows()]
    
   #      # compute the mean
        
   # grouped = pd.concat(selected_stations[dataset_name].tolist())[['year', 'height', 'u2', 'v2']].groupby(['year'])
   # mean_df = grouped.mean().reset_index()
   # # filter out non-trusted part (before NAP)
   # mean_df = mean_df[mean_df['year'] >= 1890].copy()
  
   grouped = row[dataset_name][['year', 'height', 'u2', 'v2']]
   
   
   
   
   # Drop all data before 1890 due to incorrect standards, if dataset does not predate 1890 select first date
   def year_selection(df,drop_year): 
       if df.year.min() <= drop_year:
           df = df[df['year'] >= drop_year].copy()
           df.dropna(inplace=True)
           
       elif df.year.min() >= drop_year:
           df = df[df['year'] >= df.year.min()].copy()
           df.dropna(inplace=True)
       return df
   
   mean_df = year_selection(grouped,1890)
    
   def timeseries_plot(dataset_name=default_dataset_name):
       # show all the stations, including the mean
       title = 'Sea-surface height for European tide gauges [{year_min} - {year_max}]'.format(
           year_min=mean_df.year.min(),
           year_max=mean_df.year.max() 
       )
       fig = bokeh.plotting.figure(title=title, x_range=(1860, 2020), plot_width=900, plot_height=400)
      
       data = row[dataset_name]
       fig.circle(data.year, data.height, color='red', legend_label=row['name'], alpha=0.5, line_width=1)
       fig.legend.location = "bottom_right"
       fig.yaxis.axis_label = 'waterlevel [mm] above MSL'
       fig.xaxis.axis_label = 'year'
       fig.legend.click_policy = "hide"
       bokeh.io.export_png(fig,filename=("Data_plot_"+str(row["name"])+"_"+str(dataset_name)+".png"))
       return fig
   
   bokeh.io.show(timeseries_plot(default_dataset_name))
  
   bokeh.io.show(timeseries_plot('rlr_monthly'))        
   
   # first the model without wind and AR
   linear_fit, linear_names = lib.models.linear_model(mean_df, with_wind=False, with_ar=False)
   table = linear_fit.summary(
       yname='Sea-surface height', 
       xname=linear_names, 
       title='Linear model without (1890-current)'
   )
   IPython.display.display(table)
   
   # and then the model with wind
   linear_with_wind_fit, linear_with_wind_names = lib.models.linear_model(mean_df, with_wind=True, with_ar=True)
   table = linear_with_wind_fit.summary(
       yname='Sea-surface height', 
       xname=linear_with_wind_names,
       title='Linear model with wind (1948-current)'
   )
   IPython.display.display(table)
   
   
   
   #!!!!!
   DATA = linear_with_wind_fit.predict()
   YEAR = linear_with_wind_fit.model.exog[:, 1] + 1970
   linear_with_wind_fit_appended = pd.DataFrame([YEAR,DATA],index = None)
   linear_with_wind_fit_appended = linear_with_wind_fit_appended.swapaxes(0,1)
   linear_with_wind_fit_appended.rename(columns={0:'Year',1:'SL'}, inplace=True)
   
   DATA = linear_with_wind_fit.predict()
   YEAR = linear_with_wind_fit.model.exog[:, 1] + 1970
   linear_without_wind_fit_appended = pd.DataFrame([YEAR,DATA],index = None)
   linear_without_wind_fit_appended = linear_without_wind_fit_appended.swapaxes(0,1)
   linear_without_wind_fit_appended.rename(columns={0:'Year',1:'SL'}, inplace=True)
   
   
   def store_data_with_wind(test,df):
       df_with_wind[test] = np.nan
       first_year = df["Year"][0]
       lenght = len(df["SL"])
       index_nmbr = df_with_wind["Year"].index[df_with_wind["Year"] == first_year].tolist()
       df_with_wind.loc[index_nmbr[0]:index_nmbr[0]+lenght-1,str(row['name'])] = df["SL"].tolist()
       print(df_with_wind)
       return df_with_wind
   
   def store_data_without_wind(test,df):
       df_without_wind[test] = np.nan
       first_year = df["Year"][0]
       lenght = len(df["SL"])
       index_nmbr = df_without_wind["Year"].index[df_without_wind["Year"] == first_year].tolist()
       df_without_wind.loc[index_nmbr[0]:index_nmbr[0]+lenght-1,str(row['name'])] = df["SL"].tolist()
       print(df_without_wind)
       return df_without_wind
   
    
   store_data_with_wind(str(row['name']),linear_with_wind_fit_appended)
   store_data_without_wind(str(row['name']),linear_without_wind_fit_appended)
   
   
   # Wikipedia, based on Akaike(1974): Given a set of candidate models 
   # for the data, the preferred model is the one with the minimum AIC value. 
   if (linear_fit.aic < linear_with_wind_fit.aic):
       print('The linear model without wind is a higher quality model (smaller AIC) than the linear model with wind.')
   else:
       print('The linear model with wind is a higher quality model (smaller AIC) than the linear model without wind.')
   
   # plot the model with wind. 
   fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400, title = str(row["name"]+ " Station"))
   fig.circle(mean_df.year, mean_df.height, line_width=1, legend_label='Monthly mean sea level', color='black', alpha=0.5)
   fig.line(
       linear_with_wind_fit.model.exog[:, 1] + 1970, 
       linear_with_wind_fit.predict(), 
       line_width=3, 
       alpha=0.5,
       legend_label='Current sea level, corrected for wind influence'
   )
   fig.line(
       linear_fit.model.exog[:, 1] + 1970, 
       linear_fit.predict(), 
       line_width=3, 
       legend_label='Current sea level', 
       color='green',
       alpha=0.5
   )
   
   fig.legend.location = "top_left"
   fig.yaxis.axis_label = 'waterlevel [mm] MSL (1971-2006)'
   fig.xaxis.axis_label = 'year'
   fig.legend.click_policy = "hide"
   
   bokeh.io.show(fig)
   bokeh.io.export_png(fig,filename=("Model_plot_with_wind_"+str(row["name"])+".png"))
   
   
   p = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
   
   # Plot linear model with wind acceleration (deriviative)
   def deriv_num(data):
      data_deriv = []
      for i in range(len(data)-1):
          a = float(data[i+1]-data[i])
          data_deriv.append(a)
      data_deriv.append(0)
      data_deriv = np.array(data_deriv)
      
      return data_deriv
        
   fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400, title = str(row["name"]+ " Station"))
   fig.line(
       linear_with_wind_fit.model.exog[:, 1] + 1970, 
       deriv_num(linear_with_wind_fit.predict()), 
       line_width=3, 
       alpha=0.5,
       legend_label='Current sea level, corrected for wind influence'
   )
   
   fig.legend.location = "top_left"
   fig.yaxis.axis_label = 'waterlevel [mm] MSL (1971-2006)'
   fig.xaxis.axis_label = 'year'
   fig.legend.click_policy = "hide"
   
   bokeh.io.show(fig)
   bokeh.io.export_png(fig,filename=("Model_plot_with_wind_acceleration_"+str(row["name"])+".png"))
   
   
   p = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)   
   
   df = row[default_dataset_name]
   
   df = year_selection(df,1890)

   
   
   fit, linear_with_wind_names = lib.models.linear_model(df, with_wind=True)
   smry = fit.summary(xname=linear_with_wind_names, title=row['name'])
   
   
   IPython.display.display(smry.tables[1])
   p.circle(row[default_dataset_name].year, row[default_dataset_name].height, alpha=0.1, color="black")
   
   def Trend_Break(data,t,dataset_name_):
      
      t = t.values.flatten()
      data = data.values.flatten()
      fig, ax = plt.subplots(figsize=(10, 6))

      plt.scatter(t, data, alpha=0.3);
      
      if len(t) > 200:
          model = LinearBrokenTrendFourier(f_order=3)
      elif len(t) < 200:
          model = LinearBrokenTrendFourier(f_order=0)
          
      res = model.fit(t, data)
      res
      model.parameters
      fig, ax = plt.subplots(figsize=(10, 6))
      
      ax.scatter(t, data, alpha=0.3)
      ax.plot(t, model.predict(t), linewidth=2, color='b')
      ax.axvline(model.parameters['t_break'], color='k')
      
      boot_res = block_ar_wild(model, n_samples=10, use_cache=True)
      cf_intervals = boot_res.get_ci_bounds()
  
      fig, ax = plt.subplots(figsize=(10, 6))

      ax.scatter(t, data, alpha=0.3)
      ax.plot(t, model.predict(t), linewidth=2, color='b')
      ##moker
      ax.axvline(model.parameters['t_break'], color='k')
      ax.axvspan(*cf_intervals['t_break'], color='grey', alpha=0.2)
      ax.set_title('Trend break test for ' + str(row["name"]) + ' station',loc = 'left',size=20)
      ax.set_ylabel("Waterlevel [mm] relative to MSL (1971-2006)", size = 15)
      ax.set_xlabel("Year", size = 15)
      ax.grid(alpha=0.4)
      ax.tick_params(axis='x', labelsize=15)
      ax.tick_params(axis='y', labelsize=15)
      plt.savefig("Trend_Break_Test_3_"+str(row["name"])+"_"+str(dataset_name_)+".png");

   data_year = year_selection(row[dataset_name][['year', 'height']],1930)
   Trend_Break(data_year.height, data_year.year,'Yearly')
   
   data_month = year_selection(row['rlr_monthly'][['year', 'height']],1930)
   Trend_Break(data_month.height, data_month.year,'Monthly')
   
   
   # ignore wind in the plots 
   fit, linear_names = lib.models.linear_model(df, with_wind=True)
   p.line(
       fit.model.exog[:, 1] + 1970, 
       fit.predict(), 
       line_width=3, 
       alpha=0.8,
       legend_label=row['name'],
       color='red'
       )
   p.legend.click_policy = "hide"
   
  
   bokeh.io.show(p)
   
   #mean_df = mean_df.dropna(inplace=True)
   broken_linear_fit, broken_names = lib.models.broken_linear_model(mean_df)
   quadratic_fit, quadratic_names = lib.models.quadratic_model(mean_df)
   
   
   # summary of the broken linear model
   #print('rho=%s' % broken_linear_fit.history['rho'][-1])
   #IPython.display.display(broken_linear_fit.summary(yname='Sea-surface height', xname=broken_names))
   
   # summary of the quadratic model
   #print('rho=%s' % quadratic_fit.history['rho'][-1])
   
   #quadratic_fit.summary(yname='Sea-surface height', xname=quadratic_names)
   
    
   #mean_wind = mean_df.set_index('year').loc[1890][['u2', 'v2']]
   const = linear_with_wind_fit.params['const'] 
   trend = linear_with_wind_fit.params['x1'] 
   u2 = linear_with_wind_fit.params['x4'] 
   v2 = linear_with_wind_fit.params['x5'] 
   
   # linear_with_wind_fit.predict(mean_df)
   linear_with_wind_fit.model.exog.shape, linear_with_wind_fit.model.exog_names
   exog_df = pd.DataFrame(
       linear_with_wind_fit.model.exog, 
       columns=linear_with_wind_fit.model.exog_names
   ).copy()
   msg = 'Check variable numbers in code below, they changed'
   assert linear_with_wind_names == ['Constant', 'Trend', 'Nodal U', 'Nodal V', 'Wind $u^2$', 'Wind $v^2$'], msg
   # take 0 nodal tide
   exog_df['x2'] = 0 # 
   exog_df['x3'] = 0 # 
   # take the average wind
   exog_df['x4'] = exog_df['x4'].iloc[0]
   exog_df['x5'] = exog_df['x5'].iloc[0]
   linear_with_mean_wind = linear_with_wind_fit.predict(exog_df)
   linear_with_mean_wind_prediction = linear_with_wind_fit.get_prediction(exog=exog_df)
   linear_with_mean_wind_confidence_interval = linear_with_mean_wind_prediction.conf_int(obs=False)
   linear_with_mean_wind_prediction_interval = linear_with_mean_wind_prediction.conf_int(obs=True)
   linear_with_wind_confidence_interval = linear_with_wind_fit.get_prediction().conf_int(obs=False)
   linear_with_wind_prediction_interval = linear_with_wind_fit.get_prediction().conf_int(obs=True)
   
   def model_compare_plot():
       colors = bokeh.palettes.Category10[10]
       fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400,title = str(row["name"]+ " Station"))
       fig.circle(mean_df.year, mean_df.height, line_width=3, legend_label='Observed', color='black', alpha=0.5)
       fig.line(mean_df.year, linear_with_wind_fit.predict(), line_width=3, legend_label='Linear', color=colors[0])
       fig.patch(
           np.r_[mean_df.year[::-1], mean_df.year],
           np.r_[linear_with_wind_confidence_interval[::-1, 0], linear_with_wind_confidence_interval[:, 1]],
           color=colors[0],
           alpha=0.3,
           legend_label='Linear'
       )
       fig.patch(
           np.r_[mean_df.year[::-1], mean_df.year],
           np.r_[linear_with_wind_prediction_interval[::-1, 0], linear_with_wind_prediction_interval[:, 1]],
           color=colors[0],
           alpha=0.1,
           legend_label='Linear'
       )
       
       fig.line(mean_df.year, broken_linear_fit.predict(), line_width=3, color=colors[3], legend_label='Broken')
       fig.line(mean_df.year, quadratic_fit.predict(), line_width=3, color=colors[4], legend_label='Quadratic')
       #fig.line(mean_df.year, linear_with_mean_wind, line_width=3, color='#bb33bb', legend_label='Linear (mean wind)')
       #fig.patch(
       #     np.r_[mean_df.year[::-1], mean_df.year],
       #     np.r_[linear_with_mean_wind_confidence_interval[::-1, 0], linear_with_mean_wind_confidence_interval[:, 1]],
       #     color='#bb33bb',
       #     alpha=0.3, 
       #     legend_label='Linear (mean wind)'
       # )
       # fig.patch(
       #     np.r_[mean_df.year[::-1], mean_df.year],
       #     np.r_[linear_with_mean_wind_prediction_interval[::-1, 0], linear_with_mean_wind_prediction_interval[:, 1]],
       #     color='#bb33bb',
       #     alpha=0.1, 
       #     legend_label='Linear (mean wind)'
       # )
       #fig.line(mean_df.year, mean_df.height.rolling(18, center=True).mean(), line_width=3, color='#33bb33', legend_label='Rolling 18 year mean (centered)')
   
       fig.legend.location = "top_left"
       fig.yaxis.axis_label = 'waterlevel [mm] MSL (1971-2006)'
       fig.xaxis.axis_label = 'year'
       fig.legend.click_policy = "hide"
       bokeh.io.export_png(fig,filename=("Model_comparrison_"+str(row["name"])+".png"))
       return fig
   bokeh.io.show(model_compare_plot())
   
   
   export_df = mean_df.copy().reset_index()
   export_df['predicted_linear_with_wind'] = linear_with_wind_fit.predict()
   export_df['predicted_linear'] = linear_fit.predict()
   export_df['predicted_linear_mean_wind'] = linear_with_mean_wind
   export_df['predicted_linear_mean_wind_ci_025'] = linear_with_mean_wind_confidence_interval[:, 0]
   export_df['predicted_linear_mean_wind_ci_975'] = linear_with_mean_wind_confidence_interval[:, 1]
   export_df['predicted_linear_mean_wind_pi_025'] = linear_with_mean_wind_prediction_interval[:, 0]
   export_df['predicted_linear_mean_wind_pi_975'] = linear_with_mean_wind_prediction_interval[:, 1]
   export_df['date'] = export_df['year'].apply(lambda year: datetime.date(year=year, month=1, day=1))
   header = """\
   # Sea surface height for the Netherlands based on the 6 main tide gauges. 
   # Cite https://doi.org/10.2112/JCOASTRES-D-11-00169.1 (method) https://doi.org/10.5281/zenodo.1065964 (data)
   # Lineage: https://doi.org/10.5281/zenodo.1065964 (notebooks/dutch-sea-level-monitor.ipynb)
   # year -> year over which sea-level is averaged
   # date -> first day of the year over which the sea level is averaged
   # height -> mean sea surface height over main tide gauges, based on PSMSL RLR [mm relative to NAP(2005)]
   # u2/v2 -> signed squared mean wind velocity offshore [m2/s2]
   # predicted_linear_with_wind -> fitted sea surface height based on linear model with wind and tide [mm relative to NAP(2005)]
   # predicted_linear -> fitted sea surface height based on linear model with tide [mm relative to NAP(2005)]
   # predicted_linear_mean_wind  -> fitted sea surface height based on linear model with wind and 0 nodal tide   [mm relative to NAP(2005)]
   # predicted_linear_mean_wind_ci_025  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 2.5% confidence interval [mm relative to NAP(2005)]
   # predicted_linear_mean_wind_ci_975  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 97.5% confidence interval [mm relative to NAP(2005)]
   # predicted_linear_mean_wind_pi_025  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 2.5% prediction interval [mm relative to NAP(2005)]
   # predicted_linear_mean_wind_pi_975  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 97.5% prediction interval [mm relative to NAP(2005)]
   """
   stream = io.StringIO()
   stream.write(header)
   export_df.to_csv(stream, escapechar='#', index=False)
   date = datetime.datetime.now().date()
   path = pathlib.Path('dutch-sea-level-monitor-export-{}.csv'.format(date.isoformat()))
   path.write_text(stream.getvalue())
    
   export_df = mean_df.copy().reset_index()
   export_df['predicted_linear_with_wind'] = linear_with_wind_fit.predict()
   export_df['predicted_linear'] = linear_fit.predict()
   export_df['predicted_linear_mean_wind'] = linear_with_mean_wind
   export_df['predicted_linear_mean_wind_ci_025'] = linear_with_mean_wind_confidence_interval[:, 0]
   export_df['predicted_linear_mean_wind_ci_975'] = linear_with_mean_wind_confidence_interval[:, 1]
   export_df['predicted_linear_mean_wind_pi_025'] = linear_with_mean_wind_prediction_interval[:, 0]
   export_df['predicted_linear_mean_wind_pi_975'] = linear_with_mean_wind_prediction_interval[:, 1]
   export_df['date'] = export_df['year'].apply(lambda year: datetime.date(year=year, month=1, day=1))
   header = """\
   # Sea surface height for the Netherlands based on the 6 main tide gauges. 
   # Cite https://doi.org/10.2112/JCOASTRES-D-11-00169.1 (method) https://doi.org/10.5281/zenodo.1065964 (data)
   # Lineage: https://doi.org/10.5281/zenodo.1065964 (notebooks/dutch-sea-level-monitor.ipynb)
    # year -> year over which sea-level is averaged
    # date -> first day of the year over which the sea level is averaged
    # height -> mean sea surface height over main tide gauges, based on PSMSL RLR [mm relative to NAP(2005)]
    # u2/v2 -> signed squared mean wind velocity offshore [m2/s2]
    # predicted_linear_with_wind -> fitted sea surface height based on linear model with wind and tide [mm relative to NAP(2005)]
    # predicted_linear -> fitted sea surface height based on linear model with tide [mm relative to NAP(2005)]
    # predicted_linear_mean_wind  -> fitted sea surface height based on linear model with wind and 0 nodal tide   [mm relative to NAP(2005)]
    # predicted_linear_mean_wind_ci_025  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 2.5% confidence interval [mm relative to NAP(2005)]
    # predicted_linear_mean_wind_ci_975  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 97.5% confidence interval [mm relative to NAP(2005)]
    # predicted_linear_mean_wind_pi_025  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 2.5% prediction interval [mm relative to NAP(2005)]
    # predicted_linear_mean_wind_pi_975  -> fitted sea surface height based on linear model with wind and 0 nodal tide, 97.5% prediction interval [mm relative to NAP(2005)]
    """
   stream = io.StringIO()
   stream.write(header)
   export_df.to_csv(stream, escapechar='#', index=False)
   date = datetime.datetime.now().date()
   path = pathlib.Path('dutch-sea-level-monitor-export-{}.csv'.format(date.isoformat()))
   path.write_text(stream.getvalue())
    
    
    
   msg = '''The current average waterlevel above NAP (in mm), 
   based on the 6 main tide gauges for the year {year} is {height:.1f} cm.
   The current sea-level rise is {rate:.0f} cm/century'''
   print(msg.format(year=mean_df['year'].iloc[-1], height=linear_fit.predict()[-1]/10.0, rate=linear_fit.params.x1*100.0/10))
    
   if (linear_fit.aic < quadratic_fit.aic):
       print('The linear model is a higher quality model (smaller AIC) than the quadratic model.')
   else:
       print('The quadratic model is a higher quality model (smaller AIC) than the linear model.')
   if (quadratic_fit.pvalues['x2'] < 0.05):
      print('The quadratic term is bigger than we would have expected under the assumption that there was no quadraticness.')
   else:
       print('Under the assumption that there is no quadraticness, we would have expected a quadratic term as big as we have seen.')
        
   if (linear_fit.aic < broken_linear_fit.aic):
       print('The linear model is a higher quality model (smaller AIC) than the broken linear model.')
   else:
       print('The broken linear model is a higher quality model (smaller AIC) than the linear model.')
   if (broken_linear_fit.pvalues['x2'] < 0.05):
       print('The trend break is bigger than we would have expected under the assumption that there was no trend break.')
   else:
       print('Under the assumption that there is no trend break, we would have expected a trend break as big as we have seen.')
    
    # Show trends before and after 1993 for comparison with altimetry
   broken_params = dict(zip(broken_names, broken_linear_fit.params.values))
   trend_since_1993 = broken_params['Trend'] + broken_params['+trend (1993)']
   print(' ')
   print('For station ' + str(row['name']))
   print('The trend since 1993 is {:.2f} mm/year'.format(trend_since_1993))
   print('The trend before 1993 is {:.2f} mm/year'.format(broken_params['Trend']))
