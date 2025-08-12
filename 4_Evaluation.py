#!/usr/bin/env python3
# **EvaNRT-BC : Evaluation**
## This notebook aims to evaluate hourly EC concentrations (total, residential and traffic contribution) from CAMS analyses using AE33 observation data in near real time for a chosen day. In addition, the daily evaluation is transposed into the Model Quality Indicator framework proposed in the FAIRMODE Guidance Document on Modelling Quality Objectives and Benchmarking
### Import libraries
import os
import numpy as np
import pandas as pd
import scipy.stats as scp
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 22, 'font.family': 'serif'})
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
### Import parameters

#Use this code block if parameters are declared in Main_workflow.sh (and comment the one below).
#date_imp = os.environ['DATE']
#dir_imp = os.environ['DIR']
#station_list_imp = json.loads(os.environ['STATION_LIST'])
#model_list_imp=json.loads(os.environ['MODEL_LIST'])
#date_sta=datetime.strptime(date_imp, '%Y%m%d').strftime('%Y_%m_%d')
#yr=datetime.strptime(date_imp, '%Y%m%d').strftime('%Y')
#Use this code block if you run the example from the "tmp20240210" folder (and comment the one above).
date_imp = "20240210"
dir_imp = "../"
station_list_imp = ["Airparif_Chatelet","APCG_Athens-Noa","APCG_Athens-Demokritos","CNR-ISAC_Bologna","CNR-ISAC_Milano","FMI_Helsinky","SIRTA_Palaiseau"]
model_list_imp=['chimere', 'dehm', 'emep', 'euradim', 'gemaq', 'lotos', 'match', 'minni', 'mocage', 'monarch', 'silam', 'ensemble']
date_sta="2024_02_10"
yr="2024"
#Uncertainty parameters used to calculate the Model Quality Indicator. The values are proposed here based on a review of the literature about EC. More information is available in Deliverable D19 (D3.4) “High resolution mapping over European urban areas” (https://riurbans.eu/wp-content/uploads/2024/04/RI-URBANS_D19_D3_4.pdf). 
#Feel free to try other values.
U = 0.5
alpha = 0.5
RV = 1.6
Np = 20
Nnp = 1.5
perc = 0.95
beta = 2
threshold = 3
### Example of data format needed to perform the evaluation :
#Same format for both model and observations: .pkl file, 24 time steps in raws, 1 column for total EC (tot), 1 for liquid fuel fraction (traffic) and 1 for solid fuel fraction (residential).
file_dir_mod_ex = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Model_chimere_eBC_"+str(date_sta)+"_Airparif_Chatelet.pkl"
df_out_mod_ex = pd.read_pickle(file_dir_mod_ex)
print(df_out_mod_ex)

file_dir_obs_ex = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Obs_AE33_eBC_"+str(date_sta)+"_Airparif_Chatelet.pkl"
df_out_obs_ex = pd.read_pickle(file_dir_obs_ex)
print(df_out_obs_ex)
### Define function(s)
#used later in the 3rd diagnostic (FAIRMODE summary diagram)
def common_params(ax, xmin, xmax, points, mqi=None, sym=True): #aims to draw common features to all subplots (function from Evatool).

    #plot points
    ax.scatter(points, np.ones(len(points)), zorder=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    #if points live out of limits, plot a point in the dashed area
    if (points > xmax).any():
        ax.scatter([xmax+(xmax-xmin)*0.04], [1], clip_on=False,
                c=dot_col)
    if (points < xmin).any():
        ax.scatter([xmin-(xmax-xmin)*0.04], [1], clip_on=False,
                c=dot_col)

    #remove y ticks
    plt.tick_params(axis='y', which='both', left=False)
    ax.set_yticklabels([""])

    #draw a dashed rectangle at the end of the domain
    if sym is True:
        ax.spines['left'].set_visible(False)
        pol = plt.Polygon(
            xy=[[xmin, 2],
                [xmin-(xmax-xmin)*0.08, 2],
                [xmin-(xmax-xmin)*0.08, 0],
                [xmin, 0]],
            closed=False, ls='--',
            clip_on=False,
            fc='none',
            edgecolor='k')
        ax.add_patch(pol)
    ax.spines['right'].set_visible(False)
    pol = plt.Polygon(
        xy=[[xmax, 2],
            [xmax+(xmax-xmin)*0.08, 2],
            [xmax+(xmax-xmin)*0.08, 0],
            [xmax, 0]],
        closed=False,
        ls='--',
        clip_on=False,
        fc='none',
        edgecolor='k')
    ax.add_patch(pol)

    #reduce tick size
    ax.tick_params(axis='x', labelsize='small')

    #give more space to y label
    box = ax.get_position()
    ax.set_position([box.x0+box.width*0.15, box.y0, box.width*0.8,
                    box.height*0.4])

    #MQI fulfillment (plot a green or red dot)
    if mqi is not None:
        mqo = np.sum(mqi < 1)/float(len(mqi)) >= 0.9
        col = '#4CFF00'*int(mqo) + 'r'*int(~mqo)
        ax.scatter([xmax+(xmax-xmin)*0.15], [1], clip_on=False,
                c=col, s=100)    
### Evaluation divided into 3 diagnostics
#1 - Time series and scores

print("- 1st diagnostic (time series and scores):")
for site in station_list_imp:
    file_dir_obs = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Obs_AE33_eBC_"+str(date_sta)+"_"+str(site)+".pkl"    
    if os.path.isfile(file_dir_obs):
        print("     for site: "+str(site)+"")
        print("")
        df_out_obs=pd.read_pickle(file_dir_obs)

        ind_m=0
        for mod_ls in model_list_imp:
            file_dir_mod = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Model_"+str(mod_ls)+"_eBC_"+str(date_sta)+"_"+str(site)+".pkl"
            df_out_mod = pd.read_pickle(file_dir_mod)

            #calculate scores for EC tot
            df_out_obs_ = df_out_obs['tot']
            df_out_mod_ = df_out_mod['tot']       
            df_out_mod_b = np.array(df_out_mod_)
            df_out_obs_b = np.array(df_out_obs_)
            mask_b = ~np.isnan(df_out_obs_b)
            df_out_mod_b = df_out_mod_b[mask_b]
            df_out_obs_b = df_out_obs_b[mask_b]
           
            bias = np.round(np.nanmean(df_out_mod_b-df_out_obs_b),1)
            rmse = np.round(np.sqrt(np.nanmean((df_out_mod_b-df_out_obs_b)**2)),1)
            correl = np.round(np.corrcoef(df_out_mod_b,df_out_obs_b)[0,1],1)

            #plot results for EC tot
            if mod_ls == 'ensemble':
                mod_ls_MAJ = 'ENS'
            else:
                mod_ls_MAJ = mod_ls.upper()
            ind_m_list = np.arange(0,np.size(model_list_imp))
            color_list = ['blue','green','cyan','olive','magenta','gold','teal','orchid','brown','deepskyblue','darkgrey','red']
            if ind_m==np.min(ind_m_list):
                fig = plt.figure(figsize=(12,6))
                plt.title("Site: "+str(site)+" | Date: "+str(date_imp)+"",fontsize=15)
                plt.plot(df_out_obs_,'o-',color='black',linewidth=2,label = "Obs. AE33")
            if mod_ls == 'ensemble':
                plt.plot(df_out_mod_,'o--',linewidth=2,color=color_list[ind_m],label ="Mod. "+str(mod_ls_MAJ)+" (Bias: "+str(bias)+" | RMSE: "+str(rmse)+" | R: "+str(correl)+")")
            else:    
                plt.plot(df_out_mod_,'o--',linewidth=0.8,color=color_list[ind_m],label ="Mod. "+str(mod_ls_MAJ)+" (Bias: "+str(bias)+" | RMSE: "+str(rmse)+" | R: "+str(correl)+")")
            if ind_m==np.max(ind_m_list):
                plt.ylabel("EC concentration [$µg/m^{3}$]",fontsize=12)
                plt.xlabel("Hourly average",fontsize=12)
                plt.xticks(np.arange(0,23,4),fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=10,bbox_to_anchor=(1,1))
                plt.grid()
                plt.savefig(""+str(dir_imp)+"/outputs/"+str(date_imp)+"/comp_timeseries_"+str(date_sta)+"_"+str(site)+"_ECtot.png")
                plt.show()
                plt.close()
            ind_m+=1

        ind_m=0
        for mod_ls in model_list_imp:
            file_dir_mod = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Model_"+str(mod_ls)+"_eBC_"+str(date_sta)+"_"+str(site)+".pkl"
            df_out_mod = pd.read_pickle(file_dir_mod)

            #calculate scores for EC lf/sf
            df_out_obs_2= df_out_obs['lf']/df_out_obs['tot']*100
            df_out_obs_2 = np.where(df_out_obs_2<0.0,0.0,df_out_obs_2)
            df_out_mod_2 = df_out_mod['lf']/df_out_mod['tot']*100
            df_out_mod_2 = np.where(df_out_mod_2<0.0,0.0,df_out_mod_2)
            df_out_mod_3 = np.array(df_out_mod_2)
            df_out_obs_3 = np.array(df_out_obs_2)
            mask = ~np.isnan(df_out_obs_3)
            df_out_mod_3 = df_out_mod_3[mask]
            df_out_obs_3 = df_out_obs_3[mask]
            bias = np.round(np.nanmean(df_out_mod_3-df_out_obs_3),1)
            rmse = np.round(np.sqrt(np.nanmean((df_out_mod_3-df_out_obs_3)**2)),1)
            correl = np.round(np.corrcoef(df_out_mod_3,df_out_obs_3)[0,1],1)
            
            #plot results for EC lf/sf
            if mod_ls == 'ensemble':
                mod_ls_MAJ = 'ENS'
            else:
                mod_ls_MAJ = mod_ls.upper()
            ind_m_list = np.arange(0,np.size(model_list_imp))
            color_list = ['blue','green','cyan','olive','magenta','gold','teal','orchid','brown','deepskyblue','darkgrey','red']
            if ind_m==np.min(ind_m_list):
                fig = plt.figure(figsize=(12,6))
                plt.title("Site: "+str(site)+" | Date: "+str(date_imp)+"",fontsize=15)
                plt.plot(df_out_obs_2,'o-',color='black',linewidth=2,label = "Obs. AE33")
            if mod_ls == 'ensemble':
                plt.plot(df_out_mod_2,'o--',linewidth=2,color=color_list[ind_m],label ="Mod. "+str(mod_ls_MAJ)+" (Bias: "+str(bias)+" | RMSE: "+str(rmse)+" | R: "+str(correl)+")")
            else:
                plt.plot(df_out_mod_2,'o--',linewidth=0.8,color=color_list[ind_m],label ="Mod. "+str(mod_ls_MAJ)+" (Bias: "+str(bias)+" | RMSE: "+str(rmse)+" | R: "+str(correl)+")")
            if ind_m==np.max(ind_m_list):
                plt.ylabel("Source attrib. traffic [$\%$]",fontsize=12)
                plt.xlabel("Hourly average",fontsize=12)
                plt.xticks(np.arange(0,23,4),fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=10,bbox_to_anchor=(1,1))
                plt.grid()
                plt.savefig(""+str(dir_imp)+"/outputs/"+str(date_imp)+"/comp_timeseries_"+str(date_sta)+"_"+str(site)+"_ECfrac.png")
                plt.show()
                plt.close()
            ind_m+=1
    else:
        continue
#2-FAIRMODE target plot
print("- 2nd diagnostic (FAIRMODE target plot):")

step=0
for model in model_list_imp :

        dtoscan = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/"
        tofind = "Obs"
        count=0
        matches = [x for x in os.listdir(dtoscan) if tofind in x]
        if matches:
            for file in matches:
                count+=1
        last_ind=count-1       
    
        print("     for model: "+str(model)+"")
        mqi_vect = np.ma.zeros((count))
        MQI_vect = np.ma.zeros((count))
        Y_vect = np.ma.zeros((count))

        stat_ind=0
        for station in station_list_imp:
            file_dir_obs=""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Obs_AE33_eBC_"+str(date_sta)+"_"+str(station)+".pkl"

            if os.path.isfile(file_dir_obs):
                print("         and for site : "+str(station)+"")
                print("")

                #define obs. and mod. data
                obs_ind=pd.read_pickle(file_dir_obs)
                mod_ind=pd.read_pickle(""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Model_"+str(model)+"_eBC_"+str(date_sta)+"_"+str(station)+".pkl")
                df_out_obs_ = obs_ind['tot']
                df_out_mod_ = mod_ind['tot']
                df_out_mod_b = np.array(df_out_mod_,dtype='float')
                df_out_obs_b = np.array(df_out_obs_,dtype='float')
                mask_b = ~np.isnan(df_out_obs_b)
                sim = df_out_mod_b[mask_b]
                obs = df_out_obs_b[mask_b]
                obs_mean = np.nanmean(obs)
                
                #calculate metrics
                stdo = np.std(obs,axis=0)
                stds = np.std(sim,axis=0)
                PearsonR = scp.pearsonr(obs,sim)[0]
                CRMSE = np.sqrt(np.nanmean(((sim-np.nanmean(sim))-(obs-np.nanmean(obs)))**2))
                MeanBias = np.nanmean(sim - obs)
                mu = U*np.sqrt((1-alpha**2)*obs**2+(alpha*RV)**2)
                rmsu = np.sqrt(np.nanmean(mu**2))
                signx = (np.abs(stdo - stds) /
                        ((stdo*stds*2*(1-PearsonR))**0.5) > 1)*2 - 1
                x = (CRMSE/(beta*rmsu))*signx
                y = MeanBias/(beta*rmsu)
                stations = count
                valid_stations = stations
                mqi_prov = np.sqrt(x**2+y**2)
                mqi_vect[stat_ind] = mqi_prov
                rmse= np.sqrt(np.nanmean((sim-obs)**2))
                MQI = rmse/(beta*rmsu)
                MQI_vect[stat_ind] = MQI
                percent_obs = np.percentile(obs,90)
                percent_sim = np.percentile(sim,90)
                U95 = U*np.sqrt((1-alpha**2)*(obs_mean**2) / Np + (alpha**2)*(RV**2)/Nnp)
                RMSu_ = np.sqrt(np.nanmean(U95**2))
                Y = np.abs(MeanBias)/(beta*RMSu_) #Y is equal to  MQI when calculated for annual
                Y_vect[stat_ind] = Y

                #plot
                r = np.round(np.random.rand(),1)
                g = np.round(np.random.rand(),1)
                b = np.round(np.random.rand(),1)
                if stat_ind==0:
                        fig = plt.figure(figsize=(20,5))
                        ax = fig.add_subplot(111)

                        #axes
                        ax.set_xlim(-2, 2)
                        ax.set_ylim(-2, 2)
                        major_ticks = np.arange(-2, 3, 1)
                        minor_ticks = np.arange(-2, 2.1, 0.1)
                        ax.set_aspect('equal')
                        plt.xlabel("CRMSE / $\\beta RMS_U$",fontsize=18)
                        plt.ylabel("Mean Bias / $\\beta RMS_U$",fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)

                        #target lines
                        for coords in [(-2, 2, -2, 2), (-2, 2, 2, -2)]:
                                plt.plot([coords[0], coords[1]], [coords[2], coords[3]], 'k', lw=0.5)
                        plt.text(-1.9, 0, "R", color='grey', verticalalignment='center',
                                horizontalalignment='left', fontsize=18)
                        plt.text(1.9, 0, "SD", color='grey', verticalalignment='center',
                                horizontalalignment='right', fontsize=18)
                        plt.text(0, 1.9, "Mean bias > 0", color='grey', verticalalignment='top',
                                horizontalalignment='center', fontsize=18)
                        plt.text(0, -1.9, "Mean bias < 0", color='grey', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=18)

                        #target circles
                        smax = 1
                        levels = 2
                        xs, ys = np.meshgrid(np.linspace(-smax, smax), np.linspace(-smax, smax))
                        rms = np.sqrt(xs**2 + ys**2)
                        contours = ax.contour(xs, ys, rms, levels, colors='k',
                                        linestyles='dotted', linewidths=0.5)
                        circle = plt.Circle((0, 0), 1, color='#ECECEC', zorder=0)
                        ax.add_artist(circle)
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                ax.scatter(x,y,label=""+str(station)+"", color=[r,g,b])
                if stat_ind==0:
                        #text
                        plt.text(2.1, 1.9, "$\\alpha$ = {}".format(alpha), color='k',
                                verticalalignment='center', horizontalalignment='left',
                                fontsize=14)
                        plt.text(2.1, 1.6, "$\\beta$ = {}".format(beta), color='k',
                                verticalalignment='center', horizontalalignment='left',
                                 fontsize=14)
                        plt.text(2.1, 1.3, "RV = {}".format(RV), color='k',
                                verticalalignment='center', horizontalalignment='left',
                                 fontsize=14)
                        plt.text(2.1, 1.0, "$U^{RV}_{r}$ = " + "{}".format(U),
                                color='k', verticalalignment='center',
                                horizontalalignment='left',  fontsize=14)
                        plt.text(2.1, 0.5, ""+str(stations)+" stations in total", color='k',
                                verticalalignment='center', horizontalalignment='left',
                                 fontsize=14)

                #save figure
                if stat_ind==last_ind:
                        MQI90=np.percentile(MQI_vect,90)                        
                        Y90=np.percentile(Y_vect,90)                        
                        out_sta = np.nansum(np.where(mqi_vect > 1,1,0))
                        sep = [" ", "\n"]
                        plt.text(2.1, 0.2, ""+str(out_sta)+" station with MQI > 1", color='k',
                                verticalalignment='center', horizontalalignment='left',
                                 fontsize=14)
                        plt.text(2.1, -0.2, "$MQI_{90}$ = "+str(np.round(MQI90,1))+", $Y_{90}$ = "+str(np.round(Y90,1))+"", color='k',
                                verticalalignment='center', horizontalalignment='left',
                                 fontsize=14)
                        plt.title('Model '+str(model_list_imp[step])+' evaluation: eBC measurements', loc='center',color='blue',fontsize=18)
                        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.20),fontsize=11,ncol=3)
                        ax.add_artist(legend)
                        fig.savefig(""+str(dir_imp)+"/outputs/"+str(date_imp)+"/FM_target_diagram_plot_dev_"+str(model)+".png")
                        plt.tight_layout()
                        plt.show()
                        plt.close()
                stat_ind+=1
        step+=1
#3-FAIRMODE summary diagram, 

print("- 3rd diagnostic (FAIRMODE summary diagram):") #code adapted from Evatool

step=0
for model in model_list_imp :
        print("     for model: "+str(model)+"")

        from pathlib import Path
        dtoscan = ""+str(dir_imp)+"/inputs/"+str(date_imp)+"/"
        tofind = "Obs"
        count=0
        matches = [x for x in os.listdir(dtoscan) if tofind in x]
        if matches:
            for file in matches:
                count+=1
        last_ind=count-1

        correl_vec = np.ma.zeros((count))
        obs_std_vec = np.ma.zeros((count))
        sim_std_vec = np.ma.zeros((count))
        obs_mean_vec = np.ma.zeros((count))
        sim_mean_vec = np.ma.zeros((count))
        sum_treshold_o_vec = np.ma.zeros((count))
        x_vec = np.ma.zeros((count))
        x2_vec = np.ma.zeros((count))
        x3_vec = np.ma.zeros((count))
        x4_vec = np.ma.zeros((count))
        x5_vec = np.ma.zeros((count))
        Hperc_vec = np.ma.zeros((count))

        stat_ind=0
        for station in station_list_imp:
            file_dir_obs=""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Obs_AE33_eBC_"+str(date_sta)+"_"+str(station)+".pkl"

            if os.path.isfile(file_dir_obs):
                print("         and for site : "+str(station)+"")
                print("")

                #define obs. and mod. data
                obs_ind=pd.read_pickle(file_dir_obs)
                mod_ind=pd.read_pickle(""+str(dir_imp)+"/inputs/"+str(date_imp)+"/Model_"+str(model)+"_eBC_"+str(date_sta)+"_"+str(station)+".pkl")
                df_out_obs_ = obs_ind['tot']
                df_out_mod_ = mod_ind['tot']
                df_out_mod_b = np.array(df_out_mod_)
                df_out_obs_b = np.array(df_out_obs_)
                mask_b = ~np.isnan(df_out_obs_b)
                df_out_mod_b = df_out_mod_b[mask_b]
                df_out_obs_b = df_out_obs_b[mask_b]
                obs = np.array(df_out_obs_b,dtype='float')
                sim = np.array(df_out_mod_b,dtype='float')

                #calculate metrics
                obs_std = np.std(obs)
                obs_std_vec[stat_ind]=obs_std
                sim_std = np.std(sim)
                sim_std_vec[stat_ind]=sim_std
                obs_mean = np.nanmean(obs)
                obs_mean_vec[stat_ind]=obs_mean
                sim_mean = np.nanmean(sim)
                sim_mean_vec[stat_ind]=sim_mean
                PearsonR = scp.pearsonr(obs,sim)[0]
                correl_vec[stat_ind] = PearsonR
                MeanBias = np.nanmean(sim - obs)
                mu = U*np.sqrt((1-alpha**2)*obs**2+(alpha*RV)**2)
                rmsu = np.sqrt(np.nanmean(mu**2))
                sum_treshold_o = np.sum(np.where(obs > threshold,1,0)) 
                sum_treshold_o_vec[stat_ind]=sum_treshold_o
                percent_s = np.percentile(sim,perc*100)
                percent_o = np.percentile(obs,perc*100)
                stations = np.size(station_list_imp)
                valid_stations = stations
                x = MeanBias/(beta*rmsu)
                x_vec[stat_ind] = x
                x2 = (2.*obs_std*sim_std*(1. - PearsonR)) / (beta*rmsu)**2
                x2_vec[stat_ind] = x2
                x3 = (sim_std-obs_std)/(beta*rmsu)
                x3_vec[stat_ind] = x3
                U95 = U*np.sqrt((1-alpha**2)*(percent_o**2)+(alpha**2)*(RV**2))
                Hperc = (percent_s - percent_o) / (beta*U95)
                Hperc_vec[stat_ind] = Hperc

                if stat_ind==last_ind:
                        corr = ((np.nanmean((obs_mean_vec-np.nanmean(obs_mean_vec))*(sim_mean_vec-np.nanmean(sim_mean_vec)))) / (np.nanstd(obs_mean_vec)*np.nanstd(sim_mean_vec)))
                        U95_ = U*np.sqrt((1-alpha**2)*(obs_mean_vec**2)/Np + (alpha**2)*(RV**2)/Nnp)
                        RMSu_ = np.sqrt(np.nanmean(U95_**2))
                        x4 = ((2.*np.nanstd(obs_mean_vec)*np.nanstd(sim_mean_vec)*(1. - corr)) / (beta*RMSu_)**2)

                        x5 = (np.nanstd(sim_mean_vec)-np.nanstd(obs_mean_vec))/(beta*RMSu_)

                #plot
                if stat_ind==last_ind:
                        fig = plt.figure(figsize=(10,10))
                        title= 'Model '+str(model_list_imp[step])+' evaluation: eBC measurements'
                        plt.suptitle(title,color='blue',fontsize=22)
                        dot_col = "#1F77B4"
                        ymin = 0
                        ymax = 2

                        #obs mean
                        ax = fig.add_subplot(811)
                        common_params(ax, xmin=0, xmax=15, points=obs_mean_vec,
                                        sym=False)
                        ax.set_ylabel("Observed   \nmean ", labelpad=70,
                                        rotation='horizontal', verticalalignment='center',
                                        size='small')
                        ax.text(105, -2, r"$\mu gm^{-3}$")

                        #obs exceedences
                        ax = fig.add_subplot(812)
                        common_params(ax, xmin=0, xmax=50, points=sum_treshold_o_vec, sym=False)
                        ax.set_ylabel(
                        "Observed   \nexceedences \n(> " +
                        str(threshold) + r" $\mu gm^{-3}$)",
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center',
                        size='small')
                        ax.text(105, -2, "days")

                        #time bias norm
                        ax = fig.add_subplot(813)
                        common_params(ax, xmin=-2, xmax=2, points=x_vec, mqi=np.abs(x_vec))
                        ax.set_ylabel(
                        "Bias Norm",
                        size='small',
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center')
                        rect = plt.Rectangle(xy=(-1., 0.), width=2., height=2.,
                                        edgecolor='none', fc='#FFA500')
                        ax.add_patch(rect)
                        rect = plt.Rectangle(xy=(-.7, 0.), width=1.4, height=2.,
                                        edgecolor='none', fc='#4CFF00')
                        ax.add_patch(rect)

                        #time corr norm
                        ax = fig.add_subplot(814)
                        common_params(ax, xmin=0, xmax=2, points=x2_vec, mqi=x2_vec, sym=False)
                        ax.set_ylabel(
                        "1-R Norm",
                        size='small',
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center')
                        ax.text(-37/50.,-11, "----------- time -----------",
                                rotation='vertical')
                        rect = plt.Rectangle(xy=(0., 0.), width=1., height=2.,
                                                edgecolor='none', fc='#FFA500')
                        ax.add_patch(rect)
                        rect = plt.Rectangle(xy=(0., 0.), width=.5, height=2.,
                                                edgecolor='none', fc='#4CFF00')
                        ax.add_patch(rect)

                        #time stddev norm
                        ax = fig.add_subplot(815)
                        common_params(ax, xmin=-2, xmax=2, points=x3_vec, mqi=np.abs(x3_vec))
                        ax.set_yticks([1.])
                        ax.set_ylabel(
                        "StDev Norm",
                        size='small',
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center')
                        rect = plt.Rectangle(xy=(-1., 0.), width=2., height=2.,
                                                edgecolor='none', fc='#FFA500')
                        ax.add_patch(rect)
                        rect = plt.Rectangle(xy=(-.7, 0.), width=1.4, height=2.,
                                                edgecolor='none', fc='#4CFF00')
                        ax.add_patch(rect)

                        #hperc
                        ax = fig.add_subplot(816)
                        common_params(ax, xmin=-2, xmax=2, points=Hperc_vec,
                                        mqi=np.abs(Hperc_vec))
                        ax.set_ylabel(
                        "Hperc Norm",
                        size='small',
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center')
                        rect = plt.Rectangle(xy=(-1., 0.), width=2., height=2.,
                                                edgecolor='none', fc='#4CFF00')
                        ax.add_patch(rect)
                       
                        #Space corr norm
                        ax = fig.add_subplot(817)
                        common_params(ax, xmin=0, xmax=2, points=np.array([x4]), sym=False,
                                        mqi=np.array([x4]))
                        ax.set_ylabel(
                        "1-R Norm",
                        size='small',
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center')
                        ax.text(-37/50., -7, "--- space ---", rotation='vertical')
                        rect = plt.Rectangle(xy=(0., 0.), width=1., height=2.,
                                        edgecolor='none', fc='#FFA500')
                        ax.add_patch(rect)
                        rect = plt.Rectangle(xy=(0., 0.), width=.5, height=2.,
                                        edgecolor='none', fc='#4CFF00')
                        ax.add_patch(rect)

                        #space stdev norm
                        ax = fig.add_subplot(818)
                        common_params(ax, xmin=-2, xmax=2, points=np.array([x5]),
                                        mqi=np.abs(np.array([x5])))
                        ax.set_ylabel(
                        "StDev Norm",
                        size='small',
                        labelpad=70,
                        rotation='horizontal',
                        verticalalignment='center')
                        ax.annotate(
                        '({all} stations in total)'.format(all=stations),
                        xy=(1, -0.3), xycoords='axes fraction', fontsize='small',
                        xytext=(40, -30), textcoords='offset points', ha='right',
                        va='top')
                        rect = plt.Rectangle(xy=(-1., 0.), width=2., height=2.,
                                        edgecolor='none', fc='#FFA500')
                        ax.add_patch(rect)
                        rect = plt.Rectangle(xy=(-.7, 0.), width=1.4, height=2.,
                                        edgecolor='none', fc='#4CFF00')
                        ax.add_patch(rect)
                        
                        #save figure
                        fig.savefig(""+str(dir_imp)+"/outputs/"+str(date_imp)+"/FM_summary_diagram_plot_dev_"+str(model)+".png")
                        plt.show()
                        plt.close()

                stat_ind+=1

        step+=1
