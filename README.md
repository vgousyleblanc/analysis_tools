## Contribution Rules
Main branch on WCTE/analysis_tools is protected - please open a pull request (either from your own branch or fork) to push changes to main branch

Installation:

git clone <git repo location>
cd analysis_tools
pip install -e .

The -e flag allows you to edit the package 
If using on lxplus you will need to setup this in a python virtual environment 

##  WaveformProcessing

Waveform processing contains a copy of the CFD used in the test stand repository
WaveformProcessingTeststand.cfd_teststand_method() processes the CFD using that method returning 
the charge and the time for a pulse in that waveform - including non-linearity corrections 
for both

Additionally the same CFD and charge calculation method used online by the mPMT is included 
in WaveformProcessingmPMT the versions to run on single waveforms and vectorised versions to run on arrays of waveforms are given 

## do_pulse_finding and do_pulse_finding_vect

Finds pulses in the waveforms using the same method as run online on the mPMT. do_pulse_finding_vect is
a vectorised version

## CalibrationDBInterface

Interfaces with the calibration database see more instruction here
https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation
Currently processed for the test database - to be updated when the production database 
is ready. The authentication requires a credential text file ./.wctecaldb.credential 
to be in the current working directory - more details in the database interface above

## PMTMapping 

PMTMapping is a class containing the mapping of the WCTE PMTs slot and position ids to the
electronics channel and mPMT card ids and vice versa
Usage:
mapping = PMTMapping()
mapping.get_slot_pmt_pos_from_card_pmt_chan(card_id,pmt_channel) returns the slot and pmt position
and 
mapping.get_card_pmt_chan_from_slot_pmt_pos(slot_id,pmt_position) returns the card and channel
The mapping json is located in the package

## DetectorGeometry

Class to load PMT positions, directions and calculate time of flight.

## Beam monitor PID

This code performs the 1pe calibration of the ACT PMTs as well as the *basic* event PID based on monitor information (TOF, 
charge deposited in ACTs, etc...). 
The beam PID code is called by the notebooks/WCTE_beam_analysis.ipynb notebook, which in turn calls 
the BeamAnalysis class living in the notebooks/beam_monitors_pid.py python script.  

All the plots needed for visualising the selection are saved under  
notebooks/plots and any user of the code should refer to them for sanity checks, an example is provided. The
outputs of the selection are saved in a separate root file called beam_analysis_output_R{run_number}.root

The code also calculates the mean momentum for each particle type before exiting the CERN beam pipe (upstream of T0) 
and right after exiting the WCTE beam window (i.e. into the tank). These momenta are also estimated for each trigger,
based on the estimated PID. Note the error on these momenta (propagated from the time of flight resultion, taken as the
standard deviation of the TOF distribution for electrons) is very large for slow particles. Protons and deuterons events are 
identified using their time of flight but 3He nuclei aren't. The total charge in the TOF detector is saved in the output file but
no cut is placed on it. The "is_kept" branch of the output file stores information on whether the trigger passes the basic beam
requirements (no hits above threshold in the hole counters, hits in all T0, T1, T2 PMTs etc...)

The next version of this code will include additional tools for performing PID, helpful for analyses with tighter PID requirements
and some form of PID likelihood useful for estimating the confidence we have in each particle identification. 
