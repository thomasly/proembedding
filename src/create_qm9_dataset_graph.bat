@echo off

REM python -m create_qm9_dataset_graph ^
REM -e distance ^
REM -i ..\data\QM9\quantum-machine-9-aka-qm9 ^
REM -o ..\data\QM9\graphs\cutoff_5 ^
REM -c 5 ^
REM -d 6

python -m create_qm9_dataset_graph ^
-e bond ^
-i ..\data\QM9\mol2 ^
-o ..\data\QM9\graphs\bonds ^
-c 5 ^
-d 6
