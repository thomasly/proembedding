:: ampc cp3a4 cxcr4 gcr hivpr hivrt kif11
FOR %%s IN (akt1) DO (
    python pointnet_docking_dataset.py -i ..\data\docking\%%s\%%s.pointcloud -o ..\docking_pointnet_training_results -k 3 -b 128 -e 15 -d 0.5
    python pointnet_docking_dataset.py -i ..\data\docking\%%s\%%s.pointcloud -o ..\docking_pointnet_training_results -k 3 -b 128 -e 15 -d 0.5 -p
)