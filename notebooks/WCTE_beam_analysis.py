'''## PID using the beam monitors
This code is the default way of identifying particles using the beam monitor. It is providing only a template event selection that is not optimised for any Physics analyses. It should serve as an exmple around which to build your own beam particle identification code. '''

#Step 0, import libraries
import numpy as np
import importlib
#this is the file with the necessary functions for performing PID 
import beam_monitors_pid as bm

#Step 0, decide what cut to apply:

#electron ACT35 cut
tag_electron_ACT35 =  False 
cut_line = 30 #PE
#choose the number of events to read in, set to -1 if you want to read all events
n_events = -1


#Step 1, read in the data 

#### Example 1: medium momentum negative polarity
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1478, -410, 1.01, 1.06, False

### Example 2: relatively high momentum, positive polarity
run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1610, 760, 1.01, 1.015, True


##### Example 3: relatively high momentum, positive polarity
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1602, 770, 1.01, 1.015, True

######## Example 4: relatively high momentum positive polarity
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1606,780, 1.01,1.015,True

###Example 5: low momentum, positive polarity 
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1308,220,1.01,1.15, False
#careful, run 1406 muons are below threshold, due to beam momentum biais at negative polarity
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1406,-220,1.01,1.15, False
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1419,-240,1.01,1.15, False

###Example 6: 
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 2077, 450, 1.01, 1.047, False

##example 7
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 2098, 530, 1.01, 1.03, False

##example 8
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1506, -560, 1.01, 1.03, False


##example 10
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 2241, 800, 1.01, 1.015, False

##example 11
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 2023, 370, 1.01, 1.06, False



#output_filename
filename = f"beam_analysis_output_R{run_number}_full.root"


#Set up a beam analysis class 
ana = bm.BeamAnalysis(run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5)

#Store into memory the number of events desired
ana.open_file(n_events)

#Step 2: Adjust the 1pe calibration: need to check the accuracy on the plots
# which are stored in plots/PID_run{run_number}_p{run_momentum}.pdf
ana.adjust_1pe_calibration()

#Step 3: proton and heavier particle tagging with T0-T1 TOF
#We need to tag protons before any other particles to avoid double-counting
ana.tag_protons_TOF()
#TODO: identify protons that produce knock-on electrons

#Step 4: tag electrons using ACT0-2 finding the minimum in the cut line
#If we want a tighter cut, add a coefficient of reduction of the optimal cut line (e.g. 5%) to remove more electrons (and also some more muons and pions) 
tightening_factor = 0 #in units of percent of the cut line, how much you want to reduce the cut position to increase the purity of the muon/pion sample
#this is interseting but not really resolving the issue of electron contamination: leave at 0% for now
ana.tag_electrons_ACT02(tightening_factor)

#instead use ACT35 to tag electrons (when depositing more than cutline PE, for now TBD by analyser)
if tag_electron_ACT35:
    ana.tag_electrons_ACT35(cut_line)

#Step 5: check visually that the electron and proton removal makes sense in ACT35
ana.plot_ACT35_left_vs_right()

#Step 6: make the muon/pion separation, using the muon tagger in case 
#at least 0.5% of muons and pions are above the cut line. This is necessary in case the 
#Number of particles is too high to clearly see a minimum between the muons and pions
#A more thorough analysis might want to remove events that are close to the cut line for a higher purity
ana.tag_muons_pions_ACT35()

# Study the number of particles produced per spill and per POT
#TODO: move to later on in the code. 
ana.plot_number_particles_per_POT()

#Step X: end_analysis, necessary to cleanly close files 
ana.end_analysis()

input("Wait")

#Step 7: estimate the momentum for each particle from the T0-T1 TOF
# first measure the particle TOF, make the plot
#This corrects any offset in the TOF (e.g. from cable length) that can cause the TOF 
#of electrons to be different from L/c This has to be calibrated to give meaningful momentum 
#estimates later on
ana.measure_particle_TOF()

###### Check the events that aren't tagged by ACT02 but that look electron-like in ACT35
#Useful for beam analyses, it is useful to already have computed the TOF to help PID, esp. at low momentum 
ana.study_electrons(45)


##### To speed up the checks, you can interrupt the analysis at this point and look at the plots, note that the analysis has to be "ended" for the plots to be visible
# ana.end_analysis()
# input("please check the plot")

##########################################################
#This function extimates both the mean momentum for each particle type and for each trigger
#We take the the error on the tof for each trigger is the resolution of the TS0-TS1 measurement
#Taken as the std of the gaussian fit to the electron TOF
#This is still a somewhat coarse way of estimating uncertainty... 
#This also saves the momentum after exiting the beam window, recosntructed using the same techinque
#Final momentum is after exiting through the beam pipe
#There is a little offset in the total length (1cm) found through tunning, needs more precise length calculation
ana.estimate_momentum(-1.012e-2, True)

############################################################
#Visually, it looks like all the particles reach the TOF
ana.plot_TOF_charge_distribution()


#Step X: end_analysis, necessary to cleanly close files 
ana.end_analysis()

#Output to a root file
ana.output_beam_ana_to_root(filename)

