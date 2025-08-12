#!/bin/bash

##################################################################
###       Prog. "EvaNRT-BC": A. Guion et al. (2025)            ###
###          Date of last update : 18/07/2025                  ###
##################################################################
echo ""
echo "--------------------"
echo "!!! JOB LAUNCHED !!!"
echo "--------------------"
echo ""
start=`date +%s`

###### Load modules ##############################################
##################################################################
module purge
module load R/4.2.0
module load python/3.11.4
module load netcdf-fortran/4.6.0
module load netcdf-c/4.9.0
module load nco/5.1.5
module load cdo/2.1.1

##### Main directory and parameters ##############################
##################################################################
dev_mod=0 #option for debug mode (O for inactivated and 1 for activated)

export DIR=$(pwd) #your current working directory

export DATE=20250301 #choose the processing date

#choose the stations from this list: '["Airparif_BpEst","Airparif_Chatelet","AMU_Marseille-Longchamp","APCG_Athens-Noa","APCG_Athens-Demokritos","CNR-ISAC_Bologna","CNR-ISAC_Milano","FMI_Helsinky","PSI_Zurich","RADO_Bucharest","SIRTA_Palaiseau"]'
export STATION_LIST='["Airparif_BpEst","APCG_Athens-Noa","RADO_Bucharest"]'

#choose the models from this list: '["chimere","dehm","emep","euradim","gemaq","lotos","match","minni","mocage","monarch","silam","ensemble"]'
export MODEL_LIST='["chimere","ensemble"]'

##### Main programs ##############################################
##################################################################
echo "PARAMETERS"
echo "----------"
echo "working directory: $DIR"
echo "chosen day: $DATE"
echo "chosen sites: "
for stat in "${STATION_LIST[*]}"; do
	echo $stat
done
echo "chosen models: "
for model in "${MODEL_LIST[*]}"; do
        echo $model
done

echo "----------"
echo ""
[ ! -d "inputs" ] && mkdir -p $DIR/inputs/
if [ -d "$DIR/inputs/$DATE" ]
then
       	echo "Data-sets already imported and prepared... -->  Let's jump to Task 4 !"
	echo ""
	case_data=1
else
	mkdir -p $DIR/inputs/$DATE
	case_data=0
fi

if [ $case_data -eq '0' ]
then
	[ -d "tmp$DATE" ] && rm -rf $DIR/tmp$DATE
	mkdir $DIR/tmp$DATE
	echo ">> TASK 1 - Simulations import - in progress... <<" 
	echo ""
	python scripts/1_Import_MOD_cams.py
	mv *.nc tmp$DATE/
	echo "|T1--> CAMS DATA WELL DOWNLOADED FROM ONLINE|"
	echo "---------------------------------------------"
	echo ""
fi

if [ $case_data -eq '0' ]
then
	echo ">> TASK 2 - Observations import - in progress... <<"
	echo ""
	python scripts/2_Import_OBS_ae33.py
	mv *.pkl tmp$DATE/
	echo "|T2--> AE33 DATA WELL IMPORTED FROM SFTP|"
        echo "-----------------------------------------"
	echo ""
fi

if [ $case_data -eq '0' ]
then
	echo ">> TASK 3 - Data preparation - in progress... <<"
	echo ""
	python scripts/3_Data_preparation.py
	echo "|T3--> DATA PREPARED|"
	echo ""
fi

##### Make outputs and clean directories #########################
##################################################################
echo ">> TASK 4 - Evaluation - in progress... <<"
echo ""
[ ! -d "$DIR/outputs" ] && mkdir -p $DIR/outputs/
[ ! -d "$DIR/outputs/$DATE" ] && mkdir -p $DIR/outputs/$DATE
python scripts/4_Evaluation.py
echo "|T4--> EVALUATION COMPLETED|"
echo ""

if [ $dev_mod -eq '0' ]
then
	rm -rf $DIR/tmp*
fi

##################################################################
echo ""
echo "--------------------"
echo "!!! JOB FINISHED !!!"
echo "--------------------"
end=`date +%s`
delta=$(expr $end - $start)
sec=60
delta_min=$(perl -e "print $delta/$sec")
round() {
  printf "%.${2}f" "${1}"
}
delta_min2=$(round ${delta_min} 2)
echo '[Execution time was '$delta_min2' minutes.]'
echo ""