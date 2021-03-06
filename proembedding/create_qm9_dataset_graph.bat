@echo off

python -m create_qm9_dataset_graph ^
-e distance ^
-i ..\data\QM9\quantum-machine-9-aka-qm9 ^
-o ..\data\QM9\graphs\excluded\graph_label_fixed\cutoff_3 ^
-x ..\data\QM9\excludedMolecules_MG.txt ^
-c 3 ^
-d 6

REM python -m create_qm9_dataset_graph ^
REM -e bond ^
REM -i ..\data\QM9\mol2 ^
REM -o ..\data\QM9\graphs\bonds_new ^
REM -c 5 ^
REM -d 6

REM python -m create_qm9_dataset_graph ^
REM -e bond ^
REM -i ..\data\QM9\mol2 ^
REM -o ..\data\QM9\graphs\excluded\graph_label_fixed\bonds ^
REM -x ..\data\QM9\excludedMolecules_MG.txt ^
REM -c 5 ^
REM -d 6

