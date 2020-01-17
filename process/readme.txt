These files process the raw model output (which is obtainable from the ESGF).  Raw files are not included in this archive because of their large size.  I am including these files in case people want to see how the calculations are done.

calc-MPI-historical.py: reads in MPI historical ensemble fields and writes out component fluxes; also generates fluxes for the 1% runs

calc-MPI-4xCO2-piCtrl-CRF.py: reads in abrupt4xCO2 files and writes out the '-crf' files

calc-MPI-4xCO2-piCtrl.py: reads in the averages files for the 4xCO2 periods and the average of the piControl run and calculates component fluxes

feedback-4xCO2-piCntrl-diff.ipynb: reads the CRF and flux fields for the 4xCO2 run and writes out the 4xCO2_output.p file

