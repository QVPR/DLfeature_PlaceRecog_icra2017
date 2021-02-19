## Note
This fork is intended to generate HybridNet and AMOSNet descriptors given an image folder path. The codebase **needs significant cleaning**. But it serves the purpse for now. The original Readme is stored as README_original.md


# How To
Only HPC and CPU is supported for now.

## Setup HPC
Log on to HPC and run an interactive job (set parameters as per your requirements)

``` shell
qsub -I -l ncpus=1,mem=10gb,walltime=12:00:00
```

When a job is assigned:
``` shell
source /etc/profile.d/modules.sh
module load caffe/rc3-foss-2016a-7.5.18-python-2.7.11
cd /work/qvpr/workspace/DLfeature_PlaceRecog_icra2017/
```

## Extract features:
``` shell
python extract_feat_usingAMOS.py -p /work/qvpr/data/ready/gt_aligned/sample_2014-Multi-Lane-Road-Sideways-Camera/NIL/images/
```

Descriptors will be stored in your current directory. `-p <imgDirPath>` is from where images are read. Additionally, one can add `-u uniId` in the above command to include a uniqueStringId in the default savename. 

### Model choice
Defaul model is `HybridNet`, one can use `-m AmosNet` to use `AmosNet` model instead.

### Layer choice
By default, features from `fc7` layer will be extracted. Use `-l` option to specify another layer, say, `conv6` or `conv3`. 

Run `python extract_feat_usingAMOS.py -h` to know all the choices.
