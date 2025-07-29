conda activate expres_pipeline
#python ./0_mkTelluricMask.py --overwrite
for dset in {0...9}; do
    planet_file="/Volumes/Hasbrouck/ceph/ESSP_Solar/4_DataSets/SuperSecretPlanetsdataset_seq_${dset}.csv"
    python ./1_mkDataSet.py -d $dset -n 5 -v Nperday --validation-amount 2 -planet-file "$planet_file"
    python ./1a_mergeSpec.py  -d $dset
    python ./2_runCCFs.py --overwrite --iccf
    python ./3_fitIndicators.py --overwrite
done