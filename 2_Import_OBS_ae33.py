#!/usr/bin/env python3

### Libraries ###
from tqdm import tqdm
import paramiko
import stat
from io import BytesIO
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

### Parameters ###
server =  'xxx' #this code is adapted for a sftp server ('sftp.icare.univ-lille.fr')
username = 'xxx'
password = 'xxx'

key='SA' #search this word in the files
date_imp = os.environ['DATE'] #are imported from the declaration in Main_workflow.sh
dir_imp = os.environ['DIR']
station_list_imp = json.loads(os.environ['STATION_LIST'])

### Functions ###
def search_files_ftp(server, username, password, folder, keyword): #aims to search files in the sftp folders
    file_paths = []
    def search_files(sftp, path):
        files = sftp.listdir_attr(path)
        for file in files:
            full_path = path + '/' + file.filename
            if keyword in file.filename and file.filename.endswith('.txt'):
                file_paths.append(full_path)
            if stat.S_ISDIR(file.st_mode):
                search_files(sftp, full_path)
    with paramiko.Transport((server, 22)) as transport:
        transport.connect(username=username, password=password)
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            search_files(sftp, folder)
    return file_paths

def open_files(server, username, password, files2): #aims to open files and read data
    dfs_receptor_data_raw = []
    with tqdm(total=len(files2), desc='Downloading and parsing files') as pbar:
        with paramiko.Transport((server, 22)) as transport:
            transport.connect(username=username, password=password)
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                for file in files2:
                    with sftp.open(file) as f:
                        buffer = BytesIO(f.read())
                        buffer.seek(0)
                        dfs_receptor_data_raw.append(df)
                        pbar.update(1)
    return dfs_receptor_data_raw

### Workflow ###
date_sta=datetime.strptime(date_imp, '%Y%m%d').strftime('%Y_%m_%d')
yr=datetime.strptime(date_imp, '%Y%m%d').strftime('%Y')
for site in station_list_imp:
    print("")
    print("- Screening data for site : "+str(site)+"")
    folder = '/GROUND-BASED/ACMCC/'+str(site)+'/AETHA-AE33/AETHA-AE33_SA-RAW-NRT/'+str(yr)+'/'+str(date_sta)+'/'
    try :
        files= search_files_ftp(server, username, password, folder, key)
        if np.size(files) > 0:
            files2=sorted(files)
            dfs_receptor_data_raw = open_files(server,username, password,files2)
            result = pd.concat(dfs_receptor_data_raw,ignore_index=True)
            result.to_pickle("pd_df_AE33_"+str(date_sta)+"_"+str(site)+".pkl")
    except FileNotFoundError :
        print('    --> NO DATA (for this site and date) !')
    else :
        print('    --> There are data !')

#####################
######## END ########