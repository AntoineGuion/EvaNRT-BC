#!/usr/bin/env python3

# **EvaNRT-BC : Data preparation**
## This notebook aims to prepare the data loaded from "Import_obs_ae33.py" and "Import_MOD_cams.py" for evaluation. Since the online import of observations and model simulations depends on each user's access settings, an example is provided for February 10, 2024 in the “tmp20240210” folder.
### Import libraries
import os
import numpy as np
import pandas as pd
import netCDF4
import json
from datetime import datetime
import warnings 
warnings.filterwarnings("ignore")
### Import parameters
#Use this code block if parameters are declared in Main_workflow.sh (and comment the one below).
#date_imp = os.environ['DATE']
#dir_imp = os.environ['DIR']
#station_list_imp = json.loads(os.environ['STATION_LIST'])
#model_list_imp = json.loads(os.environ['MODEL_LIST'])
#date_sta = datetime.strptime(date_imp, '%Y%m%d').strftime('%Y_%m_%d')
#Use this code block if you run the example from the "tmp20240210" folder (and comment the one above).
date_imp = "20240210"
dir_imp = "../"
station_list_imp = ["Airparif_Chatelet","APCG_Athens-Noa","APCG_Athens-Demokritos","CNR-ISAC_Bologna","CNR-ISAC_Milano","FMI_Helsinky","SIRTA_Palaiseau"]
model_list_imp = ["chimere","dehm","emep","euradim","gemaq","lotos","match","minni","mocage","monarch","silam","ensemble"]
date_sta = "2024_02_10"
#Two filters for data quality control. Outliers define the maximum value below which they are taken into account. The flag depends on the data quality protocol designated by the measurement processing system.
#Change as you want.
outlier = 50 
flag_ = 0
### Workflow
#eBC concentration observations taken from AE33 aethalometers are compared with EC concentrations from CTM simulations by applying a harmonization factor (H) equal to 1.76 (Yus-Diez et al., 2021).
#Source apportionment of solid and liquid fuel is based on and the source-specific Absorption Angstrom Exponents (AAE) values from Zotter et al. (2017).
#More information is available in Deliverable D19 (D3.4) “High resolution mapping over European urban areas” (https://riurbans.eu/wp-content/uploads/2024/04/RI-URBANS_D19_D3_4.pdf).
#make loop on each model
for mod_ls in model_list_imp: 
    print("- Preparing data for model : "+str(mod_ls)+"")
    if mod_ls == 'ensemble':
        mod_ls_MAJ = 'ENS'  
    else:
        mod_ls_MAJ = mod_ls.upper()
    
    #load model simulations (netcdf file)
    filepath_m = ""+str(dir_imp)+"/tmp"+str(date_imp)+"/"+str(mod_ls_MAJ)+"_ANALYSIS.nc"
    nc_m = netCDF4.Dataset(filepath_m)
    sf_m = nc_m.variables["ecres_conc"][:,0,:,:] #time,v_level,lat,lon
    tot_m = nc_m.variables["ectot_conc"][:,0,:,:] #time,v_level,lat,lon
    lf_m = tot_m - sf_m #solid fuels are allocated to the residential sector. Liquid fuels (for the transport sector) are calculated by complementarity of the total
    lon_m = nc_m.variables["longitude"][:]
    lat_m = nc_m.variables["latitude"][:]

    #read lon lat
    df_coord=pd.read_csv(""+str(dir_imp)+"/materials/loc_lon_lat_sites.txt",delimiter=r'\s+')

    #make loop on each station
    for site in station_list_imp:
        #load observations (pkl file organized in columns "time","tot","lf","sf","flag")
        file_dir = ""+str(dir_imp)+"/tmp"+str(date_imp)+"/pd_df_AE33_"+str(date_sta)+"_"+str(site)+".pkl" #
        if os.path.exists(file_dir):
            print("")
            print("    ... and for site : "+str(site)+"")
            print("")
            df=pd.read_pickle(file_dir)
            df2=df[0].str.split(' +', expand=True)       
            df3=df2.drop(df2.columns[0],axis=1)
            df3.columns =["time","tot","lf","sf","flag"]
            df3['time'] = pd.to_datetime(df3['time'])           
            df4 =df3['time'].dt.round('1min') #for time values that are decimal
            df3['time']=df4
            df3["tot"] = df3["tot"].astype(float) #for averaging after using pd.Grouper
            df3["lf"] = df3["lf"].astype(float)
            df3["sf"] = df3["sf"].astype(float)
            df3["flag"] = df3["flag"].astype(float)
            print("         Number of measurements for the considered day : "+str(len(df3))+"")

            #exclude if > outlier
            df4=df3[df3.tot < outlier]
            print("          ... after removing outlier > "+str(outlier)+" : "+str(len(df4))+"")

            #exclude if different of flag 
            df5=df4[df4.flag == flag_]
            print("             ... and after keeping flag "+str(flag_)+" only : "+str(len(df5))+"")
            print("")

            #average by hour and clean
            df6= df5.groupby(pd.Grouper(freq='H',key='time')).mean()
            df6.index = df6.index.strftime('%H')
            df7=df6.drop(df6.columns[3],axis=1)        
        
            #full 24 vector if missing obs
            vec_24=pd.date_range(start='1/1/2000', end='1/2/2000', freq='H')
            vec_24_=vec_24[0:24].strftime('%H')
            df8=df7.reindex(vec_24_,fill_value='NaN').reset_index()
            df9=df8.drop(df8.columns[0],axis=1).astype(float)

            #save 
            df9.to_pickle(""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Obs_AE33_eBC_"+str(date_sta)+"_"+str(site)+".pkl")
        
            #interpolation on the model grid 
            check_coord=df_coord.loc[df_coord['name'].str.contains(""+str(site)+"")]
            lon_o=np.array(check_coord["lon"]) 
            lat_o=np.array(check_coord["lat"])
               
            diff_lo = np.absolute(lon_m-lon_o)
            index_lo = diff_lo.argmin()
            diff_la = np.absolute(lat_m-lat_o)
            index_la = diff_la.argmin()

            diff_la_min1=lat_m[index_la-1]-lat_o
            diff_la_plus1=lat_m[index_la+1]-lat_o
            diff_la = lat_m[index_la]-lat_o
            diff_lo_min1=lon_m[index_lo-1]-lon_o
            diff_lo_plus1=lon_m[index_lo+1]-lon_o
            diff_lo = lon_m[index_lo]-lon_o
        
            w = abs(diff_la) + abs(diff_lo)
            w_a = abs(diff_la_min1) + abs(diff_lo)
            w_b = abs(diff_la_plus1) + abs(diff_lo)
            w_c = abs(diff_la) + abs(diff_lo_min1)
            w_d = abs(diff_la) + abs(diff_lo_plus1)

            #total EC (weighted average distance to neighboring grid cells, same for traffic and residential)
            prod_a_1tot_m = tot_m[:,index_la-1,index_lo]
            prod_a_2tot_m = tot_m[:,index_la+1,index_lo]
            prod_a_3tot_m = tot_m[:,index_la,index_lo-1]
            prod_a_4tot_m = tot_m[:,index_la,index_lo+1]
            prod_b_1tot_m = tot_m[:,index_la-1,index_lo]
            prod_b_2tot_m = tot_m[:,index_la+1,index_lo]
            prod_b_3tot_m = tot_m[:,index_la,index_lo-1]
            prod_b_4tot_m = tot_m[:,index_la,index_lo+1]
            prod_atot_m = tot_m[:,index_la,index_lo] 
            prod_btot_m = tot_m[:,index_la,index_lo] 
            prodtot_m = prod_atot_m + prod_btot_m
            prod_1tot_m = prod_a_1tot_m + prod_b_1tot_m
            prod_2tot_m = prod_a_2tot_m + prod_b_2tot_m
            prod_3tot_m = prod_a_3tot_m + prod_b_3tot_m
            prod_4tot_m = prod_a_4tot_m + prod_b_4tot_m
            prodtot_m = ((prodtot_m*(1/w)) + (prod_1tot_m*(1/w_a)) + (prod_2tot_m*(1/w_b)) + (prod_3tot_m*(1/w_c)) + (prod_4tot_m*(1/w_d)))/((1/w)+(1/w_a)+(1/w_b)+(1/w_c)+(1/w_d))
        
            #traffic EC
            prod_a_1lf_m = lf_m[:,index_la-1,index_lo]
            prod_a_2lf_m = lf_m[:,index_la+1,index_lo]
            prod_a_3lf_m = lf_m[:,index_la,index_lo-1]
            prod_a_4lf_m = lf_m[:,index_la,index_lo+1]
            prod_b_1lf_m = lf_m[:,index_la-1,index_lo]
            prod_b_2lf_m = lf_m[:,index_la+1,index_lo]
            prod_b_3lf_m = lf_m[:,index_la,index_lo-1]
            prod_b_4lf_m = lf_m[:,index_la,index_lo+1]
            prod_alf_m = lf_m[:,index_la,index_lo] 
            prod_blf_m = lf_m[:,index_la,index_lo] 
            prodlf_m = prod_alf_m + prod_blf_m
            prod_1lf_m = prod_a_1lf_m + prod_b_1lf_m
            prod_2lf_m = prod_a_2lf_m + prod_b_2lf_m
            prod_3lf_m = prod_a_3lf_m + prod_b_3lf_m
            prod_4lf_m = prod_a_4lf_m + prod_b_4lf_m
            prodlf_m = ((prodlf_m*(1/w)) + (prod_1lf_m*(1/w_a)) + (prod_2lf_m*(1/w_b)) + (prod_3lf_m*(1/w_c)) + (prod_4lf_m*(1/w_d)))/((1/w)+(1/w_a)+(1/w_b)+(1/w_c)+(1/w_d))

            #residential EC
            prod_a_1sf_m = sf_m[:,index_la-1,index_lo]
            prod_a_2sf_m = sf_m[:,index_la+1,index_lo]
            prod_a_3sf_m = sf_m[:,index_la,index_lo-1]
            prod_a_4sf_m = sf_m[:,index_la,index_lo+1]
            prod_b_1sf_m = sf_m[:,index_la-1,index_lo]
            prod_b_2sf_m = sf_m[:,index_la+1,index_lo]
            prod_b_3sf_m = sf_m[:,index_la,index_lo-1]
            prod_b_4sf_m = sf_m[:,index_la,index_lo+1]
            prod_asf_m = sf_m[:,index_la,index_lo]
            prod_bsf_m = sf_m[:,index_la,index_lo]
            prodsf_m = prod_asf_m + prod_bsf_m
            prod_1sf_m = prod_a_1sf_m + prod_b_1sf_m
            prod_2sf_m = prod_a_2sf_m + prod_b_2sf_m
            prod_3sf_m = prod_a_3sf_m + prod_b_3sf_m
            prod_4sf_m = prod_a_4sf_m + prod_b_4sf_m
            prodsf_m = ((prodsf_m*(1/w)) + (prod_1sf_m*(1/w_a)) + (prod_2sf_m*(1/w_b)) + (prod_3sf_m*(1/w_c)) + (prod_4sf_m*(1/w_d)))/((1/w)+(1/w_a)+(1/w_b)+(1/w_c)+(1/w_d))
               
            #save    
            array_m=np.array([prodtot_m,prodlf_m,prodsf_m])
            df_obs_m = pd.DataFrame(data=np.transpose(array_m),index=df9.index,columns=['tot', 'lf', 'sf'])
            df_obs_m.to_pickle(""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Model_"+str(mod_ls)+"_eBC_"+str(date_sta)+"_"+str(site)+".pkl")  
        else:
            continue