#!/bin/bash
#python ./0_mkTelluricMask.py --overwrite
conda activate expres_pipeline
for dset in {0..9}; do
    planet_file="/Volumes/Hasbrouck/ceph/ESSP_Solar/4_DataSets/SuperSecretPlanets/dataset_seq_$dset.csv"
    python ./1_mkDataSet.py -d $dset --num-obs 5 --num-day 60 -v Nperday --validation-amount 2 -planet-file $planet_file
done
conda deactivate

# Code that doesn't need expres_pipeline or a specific data set
python ./1a_mergeSpec.py --overwrite
python ./2_runCCFs.py --overwrite --iccf
python ./3_fitIndicators.py --overwrite