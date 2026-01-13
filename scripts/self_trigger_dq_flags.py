## This script creates a new root file with data quality flags on a trigger by trigger 
import numpy as np
import uproot
import awkward as ak
import gc
import tracemalloc
import argparse
import os
import time
import json
from array import array
from analysis_tools import CalibrationDBInterface
from analysis_tools import PMTMapping
from enum import Flag, auto
import subprocess

class HitMask(Flag):
    STABLE_CHANNEL = 0    
    NO_TIMING_CONSTANT = 1 
    SLOW_CONTROL_EXCLUDED = 2 

class TriggerMask(Flag):
    STABLE_TRIGGER = 0    
    PERIODIC_67_ISSUE = 1 
    SLOW_CONTROL_EXCLUDED = 2 

def get_git_descriptor():
    try:
        # Get commit hash / tag
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--tags"],
            stderr=subprocess.STDOUT
        ).decode().strip()

        # Check if there are uncommitted changes (dirty repo)
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        if status:
            # print("Repository has uncommitted changes")
            raise Exception("Repository has uncommitted changes")
        return desc

    except subprocess.CalledProcessError as e:
        raise RuntimeError("Git command failed") from e

def get_run_database_data(json_path,run_number):

    with open(json_path, 'r') as f:
        data = json.load(f)

    run_key = str(run_number)
    if run_key not in data:
        raise ValueError(f"Run number {run_number} not found in the JSON data.")
    return data[run_key]    
       
def get_stable_mpmt_list_slow_control(run_data:dict,run_number:int):
    """
    Reads the slow control good run list JSON file to get the stable mPMT list for a given run number
    run_data - dict of data for specific run
    run_number - the run number integer
    returns a length-2 tuple
            set of enabled channels (in card-channel format)
            set of masked channels in card-channel format)
    """
    # with open(good_run_list_path, 'r') as f:
    #     data = json.load(f)

    # run_key = str(run_number)
    # print("run_number",run_number)
    # if run_key not in data:
    #     raise ValueError(f"Run number {run_number} not found in the JSON data.")

    enabled_channels = set(run_data["enabled_channels"])
    channel_mask = set(run_data["channel_mask"])

    return enabled_channels, channel_mask

def get_slow_control_trigger_mask(run_number_str:str, trigger_times:np.ndarray, run_data:dict):
    """
    Takes the run trigger times and applys the slow control data quality flags to return a mask of good triggers
    and a list of bad mPMTs to be excluded from the entire run - the list of bad mPMTs isn't used for now
    
    run_number -  a string for the run number
    trigger_time - a vector of trigger times in nanoseconds 
    run_data - json dictionary with run data for this run 
    returns 
            np.ndarray of Bools the same length as trigger time. Entries with "True" should be kept; entries with "False" should be discarded
    """
    
    # if run_number_str not in good_run_data:
    #     raise Exception("run_number not found",run_number_str,"in ")
    #     return np.zeros(len(trigger_times)).astype(bool), []
    
    # else:
    #     this_data = good_run_data[run_number_str]
    
    # very_bad = run_data["runtime"]<600
    # if very_bad:
    #     return np.zeros(len(trigger_times)).astype(bool), run_data["mpmts"]

    bad_mask = np.zeros(len(trigger_times)).astype(bool)
    bad_channel = []
    for problem in run_data["problems"]:
        # adding a 5 second window on either side 
        start   = (problem[0] - run_data["start"] -BUFFER)*(1e9),
        end     = (problem[1] - run_data["start"] +BUFFER)*(1e9),
        prob    = problem[2]
        if "dropped" in prob:
            # filter all trigger times within +/- buffer of the dropped packets 
            bad_mask = np.logical_or(bad_mask, np.logical_and(trigger_times>start, trigger_times<end) )

        elif ("no_data" in prob):
            this_mpmt = int(prob.split(":")[0][4:])
            for i in range(19):
                this_channel = this_mpmt*100 + i
                if this_channel not in bad_channel:
                    bad_channel.append( this_channel )
            
        elif ("Status." in prob):
            this_mpmt = int(prob.split(":")[0][4:])
            this_pmtno = int(prob.split(" ")[1][3:])
            this_channel = this_mpmt*100 + this_pmtno
            if this_channel not in bad_channel:
                bad_channel.append( this_channel )
            
        elif "bad_flow" in prob:
            pass 
        elif "crashed" in prob:
            bad_mask = np.logical_or(bad_mask, trigger_times>(run_data["end"]-30) )
        else:
            print(prob)
            raise ValueError("Unhandled problem!")
    return np.logical_not(bad_mask)

def get_67ms_mask(run_number_str:str, trigger_times:np.ndarray):
    """
    Takes the run trigger times and the run number and applies simple modulo division to determine whether 
    a trigger is effected by the 67ms missing trigger problem
    
    run_number -  a string for the run number
    trigger_time - a vector of trigger times in nanoseconds
    returns periodic_67ms_missing
            np.ndarray of Bools the same length as trigger time. Entries with "True" should be kept; entries with "False" should be discarded
    """
    if int(run_number_str)<1841:
        periodic_67ms_missing = trigger_times%67108864>1e7
    else:
        periodic_67ms_missing = np.ones(len(trigger_times))
    #these masks are 1 if no problem and 0 if issue
    return  periodic_67ms_missing

# def get_good_trigger_list_slow_control(good_run_list_path:str, data_root_file:str, run_number:int):
#     """
#     Wrapper function for get_trigger_mask
#     data_root_file - path to the WCTEReadoutWindows root file for the run
#     run_number - the run number integer
#     returns a length-2 tuple
#             np.ndarray of Bools the same length as trigger time. Entries with "True" should be kept; entries with "False" should be discarded
#             np.ndarray of Bools the same length as trigger time. Entries with "True" have no periodic 67ms issue; entries with "False" have the issue
#     """
    
#     with open(good_run_list_path, 'r') as f:
#         good_run_data = json.load(f)

#     BUFFER = 15 
    
#     with uproot.open(data_root_file) as file:
#         tree = file["WCTEReadoutWindows"]
#         trigger_times = tree["window_time"].array()
    
#     good_trigger_mask, _ = get_trigger_mask(str(run_number),trigger_times,good_run_data)
#     #for now only consider pre-hardware trigger installation runs
#     if int(run_number)<1841:
#         periodic_67ms_missing = trigger_times%67108864>1e7
#     else:
#         periodic_67ms_missing = np.ones(len(trigger_times))
#     #these masks are 1 if no problem and 0 if issue
#     return good_trigger_mask, periodic_67ms_missing

     
if __name__ == "__main__":
    
    git_hash = get_git_descriptor()

    parser = argparse.ArgumentParser(description="Add a new branch to a ROOT TTree in batches.")
    parser.add_argument("-i","--input_files",required=True, nargs='+', help="Path to WCTEReadoutWindows ROOT file")
    parser.add_argument("-c","--input_calibrated_file_directory",required=True, help="Path to WCTEReadoutWindows ROOT file")
    parser.add_argument("-r","--run_number",required=True, help="Run Number")
    parser.add_argument("-o","--output_dir",required=True, help="Directory to write output file")
    args = parser.parse_args()
    
    #check that the run number is correct 
    for input_file in args.input_files:
        if f"R{args.run_number}" not in input_file:
            raise Exception(f"Input file {input_file} does not match run number {args.run_number}")
    
    #make a list of calibrated input files - these are needed to determine for the channel list
    #of channels with calibration constants and for the hit list for which the mask is to be applied
    calibrated_input_files = []
    for input_file in args.input_files:
        base = os.path.splitext(os.path.basename(input_file))[0]
        calibrated_file_name = f"{base}_calibrated_hits.root" 
        calibrated_input_file_path = os.path.join(args.input_calibrated_file_directory, calibrated_file_name)
        calibrated_input_files.append(calibrated_input_file_path)
        if not os.path.exists(calibrated_input_file_path):
            raise Exception(f"Calibrated input file {calibrated_input_file} does not exist")
    
    #slow control file for good run list
    good_run_list_path = '/eos/experiment/wcte/configuration/slow_control_summary/good_run_list_v2.json' 
    #get run configuration from slow control
    run_data = get_run_database_data(good_run_list_path,args.run_number)
    run_configuration = run_data["trigger_name"]
    
    #get stable list of channels from slow control
    enabled_channels, channel_mask = get_stable_mpmt_list_slow_control(run_data,args.run_number)
    #the channels that are enabled less the channels that are determined as unstable
    stable_channels_card_chan = enabled_channels - channel_mask
    #map slow control channel list in card and channel to the mpmt slot and position
    mapping = PMTMapping()
    slow_control_stable_channels = [] #defined in terms of the slot id and pmt position 
    for ch in stable_channels_card_chan:
        card = ch // 100
        pmt_chan = ch % 100
        slot, pmt_pos = mapping.get_slot_pmt_pos_from_card_pmt_chan(card, pmt_chan)
        slow_control_stable_channels.append(100 * slot + pmt_pos)
    slow_control_stable_channels = np.array(slow_control_stable_channels)
    
    #loop over each file    
    first_file_pmts_with_timing_constant = None
    for readout_window_file_name, calibrated_input_file_name in zip(args.input_files, calibrated_input_files):
        
        with uproot.open(readout_window_file_name) as readout_window_file:
            with uproot.open(calibrated_input_file_name) as calibrated_file:
                
                config_tree = calibrated_file["Configuration"]
                pmts_with_timing_constant = config_tree["wcte_pmts_with_timing_constant"].array().to_numpy()[0]
                
                #this list should be the same for all files in the run
                if first_file_pmts_with_timing_constant is None:
                    first_file_pmts_with_timing_constant = pmts_with_timing_constant
                else:
                    if not np.array_equal(first_file_pmts_with_timing_constant, pmts_with_timing_constant):
                        raise Exception("PMTs with timing constants do not match between files in the same run")
                
                # Construct output path
                base = os.path.splitext(os.path.basename(readout_window_file_name))[0]
                new_filename = f"{base}_self_trigger_dq_flags.root" 
                os.makedirs(args.output_dir, exist_ok=True)
                output_file_name = os.path.join(args.output_dir, new_filename)

                #create the output file
                with uproot.recreate(output_file_name) as outfile:
            
                    config_tree = outfile.mktree("Configuration", {
                        "git_hash": "string",
                        "run_configuration": "string",
                        "good_wcte_pmts": "var * int32", #the global pmt id (slot*100 + pos) of good pmts with timing constants and stable in slow control
                        "wcte_pmts_with_timing_constant": "var * int32", #the global pmt id (slot*100 + pos) of pmts with timing constants
                        "wcte_pmts_slow_control_stable": "var * int32", #the revision id of the timing constant set used
                    })
                    print("There were",len(stable_channels_card_chan),"enabled channels not masked out")
                    print("There were",len(pmts_with_timing_constant),"channels with timing constants")
                    # print((set(pmts_with_timing_constant)- slow_control_stable_channels_set),"channels have timing constants but not stable in slow control")
                    print("In total there are",len(set(pmts_with_timing_constant) & set(slow_control_stable_channels)),"good channels with timing constants and stable in slow control")
                    
                    config_tree.extend({
                        "git_hash": [git_hash],
                        "run_configuration": [run_configuration],
                        "good_wcte_pmts": ak.Array([list(set(pmts_with_timing_constant) & set(slow_control_stable_channels))]), 
                        "wcte_pmts_with_timing_constant": ak.Array([pmts_with_timing_constant]),
                        "wcte_pmts_slow_control_stable": ak.Array([slow_control_stable_channels])
                    })
                    
                    # Create a TTree to store the flags
                    tree = outfile.mktree("DataQualityFlags", { #only WCTE detector hits are stored here (not trigger mainboard hits)
                        "hit_pmt_hit_pmt_readout_mask": "var * int32",
                        "window_data_quality_mask": "int32",
                        "readout_number": "int32" #the unique readout window number for this event in the run
                    })
                    
                    #batch load to get the
                    readout_window_tree_entries = readout_window_file["WCTEReadoutWindows"].num_entries 
                    calibrated_tree_entries = calibrated_file["CalibratedHits"].num_entries
                    if  readout_window_tree_entries!=calibrated_tree_entries:
                        print("Input file problem different number of entries between calibrated and original file")
                        # print("debug mode: override")
                        # readout_window_tree_entries = min(readout_window_tree_entries,calibrated_tree_entries)
                        raise Exception("Input file problem different number of entries between calibrated and original file")
                    
                    batch_size = 10_000 #can use large batches as only a couple of branches are loaded
                    for start in range(0, readout_window_tree_entries, batch_size):  
                        stop = min(start + batch_size, readout_window_tree_entries)
                        print(f"Loading entries {start} → {stop}")
                        branches_to_load = ["window_time","readout_number"]
                        readout_window_tree = readout_window_file["WCTEReadoutWindows"]
                        readout_window_events = readout_window_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                        
                        branches_to_load = ["readout_number","hit_mpmt_slot","hit_pmt_pos"]
                        calibrated_tree = calibrated_file["CalibratedHits"]
                        calibrated_file_events = calibrated_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                        
                        if not np.array_equal(readout_window_events["readout_number"].to_numpy(),calibrated_file_events["readout_number"].to_numpy()):
                            raise Exception("Batch start",start,"different events being compared between two files")
                        
                        #trigger level flags
                        sc_good_trigger_mask = get_slow_control_trigger_mask(args.run_number,readout_window_events["window_time"].to_numpy(),run_data)
                        periodic_67ms_missing_mask = get_67ms_mask(args.run_number,readout_window_events["window_time"].to_numpy())
                        
                        #make the trigger level bitmask 
                        trigger_mask = np.zeros_like(sc_good_trigger_mask, dtype=np.int32)                        
                        trigger_mask |= ~periodic_67ms_missing_mask * TriggerMask.PERIODIC_67_ISSUE.value
                        trigger_mask |= ~sc_good_trigger_mask * TriggerMask.SLOW_CONTROL_EXCLUDED.value
                         
                        #hit level flags
                        hit_global_id = (100*calibrated_file_events["hit_mpmt_slot"])+calibrated_file_events["hit_pmt_pos"]
                        hit_global_id_flat = ak.to_numpy(ak.flatten(hit_global_id))
                        
                        has_time_constant = np.isin(hit_global_id_flat,pmts_with_timing_constant)
                        is_sc_stable = np.isin(hit_global_id_flat,slow_control_stable_channels)
                        #make the trigger level bitmask 
                        hit_mask_flat = np.zeros_like(hit_global_id_flat, dtype=np.int32)
                        hit_mask_flat |= ~has_time_constant * HitMask.NO_TIMING_CONSTANT.value
                        hit_mask_flat |= ~is_sc_stable * HitMask.SLOW_CONTROL_EXCLUDED.value
                        hit_mask = ak.unflatten(hit_mask_flat,ak.num(hit_global_id))
                        
                        #append to tree
                        tree.extend({
                            "hit_pmt_hit_pmt_readout_mask": hit_mask,
                            "window_data_quality_mask": trigger_mask,
                            "readout_number": readout_window_events["readout_number"].to_numpy()
                        })
                        
                        print("Batch processed",np.sum(trigger_mask==0),"/",len(trigger_mask),"good triggers", f"{np.sum(trigger_mask==0)/len(trigger_mask):.2%}")
                        print("Processed",np.sum(hit_mask_flat==0),"/",len(hit_mask_flat),"good hits", f"{np.sum(hit_mask_flat==0)/len(hit_mask_flat):.2%}")