# P6
This respository contans the script nessecary to compute all plots in chapter 5 of out bachelor project. 

The scripts are pationed such that the plots belonging to section 5.1 ends with _lung, and those belonging to section 5.2 with _stroke.

The 'Image_processor_lung' script contains the code nessecary to process the chest X-rays, and 'load_data_lung' for loading the images used in conformal prediction.

In general calculating the performance metrics as an average or just across a single relization can be changed by changing n_runs in the argument of the plotting function.
