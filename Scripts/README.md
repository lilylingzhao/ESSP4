# Generate Data Sets
Scripts relevant to generating data sets start with numbers.

Script Name
- Description
- Is it run data set by data set?
- Relevant arguments
- Relevant default arguments

| Script Name          | Description |
| -------------------- | ----------- |
| `0_mkTelluricMask.py`| 
| `1_mkDataSet.py`     |
| `1a_mergeSpec.py`    |
| `2_runCCFs.py`       |
| `3_fitIndicators.py` |

# Post-Processing/Analysis Focused
Scripts relevant to post-processing and analysis of results start with letters (though there's no real reason in this case for them to be ordered)

### A - SDO Data
It's faster to download the needed SDO files separately to them being processed.  This also simplifies parallelizing the processing.  I would set the SDO download going, wait for a few days to download (depending on how many threads to be used in the processing), and then start the processing script going.

It slightly complicates what happens if there are errors or interuptions to a day.  There is therefore also a script that targets time stamps for which there are not saved values.  This is helpful to run after an initial download/processing of all days.

| Script Name          | Description |
| -------------------- | ----------- |
| `Aa_sdoDownload.py`  | Download all relevant SDO files a day at a time|
| `Ab_sdoProcessing.py`| Process all SDO files a day at a time|
| `Ac_sdoTouchUp.py`   | Check each day for time stamps without processed data and attempt to redownload/reprocess|


# Other
`pipeline.sh`: run through scripts to generate ESSP data sets with default arguments.

`results_reorg.py`: Re-organize submitted files directly downloaded from Box to standardize across groups and methods.