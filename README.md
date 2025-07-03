# Tuned_PILOT
Code repository for the thesis: Extending PILOT by tuning the BIC penalty.

The plots from the simulation study are included in CodeExtension under experiments. 

How to replicate the results in: "Extending PILOT by tuning the BIC penalty"

In the folder PILOT-Extension_and_Replication the following files are important to recreate the results in the paper:
(It is important to read from top to bottom as this is the order in which you can exactly replicate all the results and follow which methods are used to make the plots).

benchmark_replication: 
This file should be run to train the model CART and PILOT (referred to as PILOT_Base in most of the code). This training can be done by including 'PILOT_Base' and 'CART' in the selected_keys list on line 29.

These models are then "configured" by using the dictionary of models and parameter grids in the file CodeExtension/dict.py
Make sure to check the file names in which the retrained models are saved (as a .pkl file) to make the plots for the replication with the next script:

make_plots_replication:
This file creates the plots shown in the replication section of the paper, simply check the saved model files and run the script to create all plots. This script mainly utilises the file paperplots.py, which contains the methods to make the plots. 

CodeExtension/

SimulationStudy:
The main file to run in order to obtain the results for the simulation study. The file creates plots, and a csv file in which all the scores of the models are saved of all runs. In turn, this file can be used by tables.py to create the exact tables shown in the paper. 

The file uses the simulation methods from the Simulation.py file, in this file the function to simulate the data is included as wel as a wrapper to get the original functions. 

This file should be run twice with the following settings to recreate the exact result of the paper:
First do a run solely for the MARS model for 250 replications. 
We perform both runs with the following rest of the settings: 

{
	n = [500], 
	random = 42, 
	n_iter= 75, 
	domain=6, 
	n_test = 5000,
 	indices= [1,2,3,4,5,6],
 	plot=True,
	valid_pairs = [(0, 0.0), (1, 0.1)]
}

To set the models simply put in the selected_keys list: 
'PILOT_Base','CART','PILOT','TrueDGP','LightGBMLinear', ('MARS')

Then, to create the tables run the file tables.py:

tables.py:
This file recreates the tables for the mse, and the wilcoxon tests. Make sure to reference the mars, and other csv document properly in the first 2 read_csv functions. 


Furthermore;

The original PILOT code is modified to be compatible with the Scikit-Learn methods that we use. Therefore, we implemented the code in the Pilot.py file, and stored the original code in originalPilot.py.





