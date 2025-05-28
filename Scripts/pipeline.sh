source /mnt/home/lzhao/EXPRES/env.sh
python ./0_mkTelluricMask.py --overwrite
python ./1_mkDataSet.py -d 1 -n 5 -v Nperday --validation-amount 2 -planet-file /mnt/home/lzhao/SolarComparison/DataSetPlanets/dataset_seq_1.csv
python ./2_runCCFs.py --overwrite
python ./3_fitIndicators.py --overwrite