#!/bin/bash
#python ./0_mkTelluricMask.py --overwrite
for dset in {0..9}; do
    conda activate expres_pipeline
    planet_file="/Volumes/Hasbrouck/ceph/ESSP_Solar/4_DataSets/SuperSecretPlanets/dataset_seq_$dset.csv"
    python ./1_mkDataSet.py -d $dset --num-obs 5 --num-day 60 -v Nperday --validation-amount 2 -planet-file $planet_file
    conda deactivate
    python ./1a_mergeSpec.py -d $dset --overwrite
    python ./2_runCCFs.py -d $dset --overwrite --iccf
    python ./3_fitIndicators.py -d $dset --overwrite
done