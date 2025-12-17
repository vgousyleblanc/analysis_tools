'''## PID using the beam monitors
This code is the default way of identifying particles using the beam monitor. It is providing only a template event selection that is not optimised for any Physics analyses. It should serve as an exmple around which to build your own beam particle identification code. '''

#Step 0, import libraries
import numpy as np
import importlib
#this is the file with the necessary functions for performing PID 
import sys
#path to analysis tools - change with where your path is, though it might just work with mine on eos
sys.path.append("/eos/user/a/acraplet/analysis_tools/")
from analysis_tools import BeamAnalysis # as bm
# import cProfile


#choose the number of events to read in, set to -1 if you want to read all events
n_events = -1


#Step 1, read in the data 

#### Example 1: medium momentum negative polarity
run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1478, -410, 1.01, 1.06, False

### Example 2: relatively high momentum, positive polarity
# run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5 = 1610, 760, 1.01, 1.015, True


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
filename = f"../data/beam_data_analysis/beam_analysis_R{run_number}.root"


#Set up a beam analysis class 
ana = BeamAnalysis(run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5)


# cProfile.run("ana.open_file(n_events, require_t5 = True)", sort="cumtime")

#Store into memory the number of events desired
ana.open_file(n_events, require_t5 = True)



#Step 2: Adjust the 1pe calibration: need to check the accuracy on the plots
# which are stored in plots/PID_run{run_number}_p{run_momentum}.pdf
ana.adjust_1pe_calibration()

#Step 3: proton and heavier particle tagging with T0-T1 TOF
#We need to tag protons before any other particles to avoid double-counting
ana.tag_protons_TOF()
#TODO: identify protons that produce knock-on electrons


#Step 4: tag electrons using ACT0-2 finding the minimum in the cut line
ana.tag_electrons_ACT02()

#Step 5: check visually that the electron and proton removal makes sense in ACT35
ana.plot_ACT35_left_vs_right()

#Step 6: make the muon/pion separation, using the muon tagger in case 
#at least 0.5% of muons and pions are above the cut line. This is necessary in case the 
#Number of particles is too high to clearly see a minimum between the muons and pions
#A more thorough analysis might want to remove events that are close to the cut line for a higher purity
ana.tag_muons_pions_ACT35()

### here check the TOF distributions
ana.plot_all_TOFs()


#Step X: end_analysis, necessary to cleanly close files 
ana.end_analysis()

#Output to a root file
ana.output_to_root(filename)

