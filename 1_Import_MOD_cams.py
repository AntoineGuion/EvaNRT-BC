#!/usr/bin/env python3

### Libraries ###
import os
import json
import numpy as np
from datetime import datetime
import cdsapi

### Parameters ###
date_imp = os.environ['DATE'] #are imported from the declaration in Main_workflow.sh
model_list_imp=json.loads(os.environ['MODEL_LIST'])
date_sta=datetime.strptime(date_imp, '%Y%m%d').strftime('%Y-%m-%d')
date_end=datetime.strptime(date_imp, '%Y%m%d').strftime('%Y-%m-%d')
c = cdsapi.Client() #refer to https://ads.atmosphere.copernicus.eu/how-to-api for data access with CDS API client

### Workflow ###
print("- check date:")
print(date_sta)
print("")
print("- check model:")
for i in np.arange(0,np.size(model_list_imp)):
    print(model_list_imp[i])
    if model_list_imp[i] == 'ensemble':
        i_MAJ = 'ENS'
    else:
        i_MAJ = model_list_imp[i].upper()
    c.retrieve(
        'cams-europe-air-quality-forecasts',
        {
            'variable': [
                'residential_elementary_carbon', 'total_elementary_carbon', #necessary variables
            ],
            'model':[''+str(model_list_imp[i])+'',
            ],
            'level': '0',
            'date': ''+str(date_sta)+'/'+str(date_end)+'',
            'type': 'analysis',
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'leadtime_hour': '0',
            'format': 'netcdf',
        },
        ''+str(i_MAJ)+'_ANALYSIS.nc')
    print("")

#####################
######## END ########