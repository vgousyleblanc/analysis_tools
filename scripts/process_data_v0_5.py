import ROOT
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

class HitMask(Flag):
    STABLE_CHANNEL = 0    
    NO_TIMING_CONSTANT = 1 
    SLOW_CONTROL_EXCLUDED = 2 

class TriggerMask(Flag):
    STABLE_TRIGGER = 0    
    PERIODIC_67_ISSUE = 1 
    SLOW_CONTROL_EXCLUDED = 2 

def get_run_database_data(run_number):

    json_path = '/eos/experiment/wcte/configuration/slow_control_summary/good_run_list.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    run_key = str(run_number)
    print("run_number",run_number)
    if run_key not in data:
        raise ValueError(f"Run number {run_number} not found in the JSON data.")
    return data[run_key]    
       
def get_stable_mpmt_list_slow_control(run_number):

    json_path = '/eos/experiment/wcte/configuration/slow_control_summary/good_run_list.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    run_key = str(run_number)
    print("run_number",run_number)
    if run_key not in data:
        raise ValueError(f"Run number {run_number} not found in the JSON data.")

    enabled_channels = set(data[run_key]["enabled_channels"])
    channel_mask = set(data[run_key]["channel_mask"])

    return enabled_channels, channel_mask

def get_good_trigger_list_slow_control(data_root_file, run_number):
    
    json_path = '/eos/experiment/wcte/configuration/slow_control_summary/good_run_list.json'

    with open(json_path, 'r') as f:
        good_run_data = json.load(f)

    BUFFER = 15 
    
    with uproot.open(data_root_file) as file:
        tree = file["WCTEReadoutWindows"]
        trigger_times = tree["window_time"].array()
    
    def get_trigger_mask(run_number_str:str, trigger_times:np.ndarray, good_run_data):
        """
        run_number -  a string for the run number
        trigger_time - a vector of trigger times in coarse counters  
        returns a length-2 tuple
                np.ndarray of Bools the same length as trigger time. Entries with "True" should be kept; entries with "False" should be discarded
                a vector of bad mPMTs which should be omitted from the entire run
        """
        
        if run_number_str not in good_run_data:
            raise Exception("run_number not found",run_number_str,"in run data")
            return np.zeros(len(trigger_times)).astype(bool), []
        
        else:
            this_data = good_run_data[run_number_str]
        
        very_bad = this_data["runtime"]<600
        if very_bad:
            return np.zeros(len(trigger_times)).astype(bool), this_data["mpmts"]

        bad_mask = np.zeros(len(trigger_times)).astype(bool)
        bad_channel = []
        for problem in this_data["problems"]:
            # adding a 5 second window on either side 
            start   = (problem[0] - this_data["start"] -BUFFER)*(1e9),
            end     = (problem[1] - this_data["start"] +BUFFER)*(1e9),
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
                bad_mask = np.logical_or(bad_mask, trigger_times>(this_data["end"]-30) )
            else:
                print(prob)
                raise ValueError("Unhandled problem!")
        return np.logical_not(bad_mask), this_data["channel_mask"] 
    
    good_trigger_mask, _ = get_trigger_mask(str(run_number),trigger_times,good_run_data)
    #for now only consider pre-hardware trigger installation runs
    if int(run_number)<1841:
        periodic_67ms_missing = trigger_times%67108864>1e7
    else:
        periodic_67ms_missing = np.ones(len(trigger_times))
    #these masks are 1 if no problem and 0 if issue
    return good_trigger_mask, periodic_67ms_missing
    

def process_data(input_file_names, run_number ,output_dir, timing_offsets_dict, timing_constant_set, slow_control_stable_channels_set):
    # get the timing constants from calibration database

    # make a fast lookup table for the offsets
    # Define a safe lookup function with default fallback
    DEFAULT_OFFSET = 0
    def safe_lookup(glb_pmt_pos_id):
        return timing_offsets_dict.get(glb_pmt_pos_id, DEFAULT_OFFSET)
    timing_offset_lookup = np.frompyfunc(safe_lookup, 1, 1)

    # Vectorized check whether a constant was found function
    def get_hit_mask_vectorized(card_ids: np.ndarray,
                            glb_pmt_pos_id: np.ndarray,
                            timing_constant_set: set,
                            slow_control_stable_channels_set: set) -> np.ndarray:
        """
        Vectorized computation of HitMask flags for arrays of hits.

        Parameters:
            card_ids (np.ndarray): Array of card IDs.
            glb_pmt_pos_id (np.ndarray): Array of global channel position IDs (100*slot + pmt).
            timing_constant_set (set[int]): Channels with timing constants.
            slow_control_stable_channels_set (set[int]): Channels marked stable by slow control.
        Returns:
            np.ndarray: Array of HitMask integer values.
        """
        # Start with STABLE_CHANNEL (0)
        mask = np.full(card_ids.shape, HitMask.STABLE_CHANNEL.value, dtype=np.uint8)
        # Only consider channels with card_id <= 120 (exclude trigger mainboard)
        is_data_channel = card_ids <= 120

        # Channels without timing constants
        no_timing = is_data_channel & ~np.isin(glb_pmt_pos_id, list(timing_constant_set))
        mask[no_timing] |= HitMask.NO_TIMING_CONSTANT.value

        # Channels excluded by slow control
        not_stable = is_data_channel & ~np.isin(glb_pmt_pos_id, list(slow_control_stable_channels_set))
        mask[not_stable] |= HitMask.SLOW_CONTROL_EXCLUDED.value
        # print("len no_timing",np.sum((~np.isin(glb_pmt_pos_id, timing_constant_set))==1),len(no_timing))
        # print("len not_stable",np.sum(not_stable==1),np.sum(is_data_channel==1),len(not_stable))
        return mask

    
    tree_name = "WCTEReadoutWindows"  # Replace with actual TTree name

    for input_file_name in input_file_names:
        # Construct output path
        filename = os.path.basename(input_file_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)

        #get mask of good triggers from slow control 
        good_trigger_mask, periodic_67ms_missing = get_good_trigger_list_slow_control(input_file_name, run_number)
        print("***** total triggers in file", len(good_trigger_mask), len(periodic_67ms_missing), "problem ones",np.sum(good_trigger_mask==0),np.sum(periodic_67ms_missing==0))
        input_file = ROOT.TFile.Open(input_file_name)
        tree = input_file.Get(tree_name)
        
        
        output_file = ROOT.TFile(output_file, "RECREATE")
        out_tree = tree.CloneTree(0)  # clone structure only, no entries
        
        window_data_quality = array('i', [0])  # 'i' = signed int (4 bytes)
        window_data_quality_branch = out_tree.Branch("window_data_quality", window_data_quality, "window_data_quality/I")

        hit_pmt_calibrated_times = ROOT.std.vector('double')()
        hit_pmt_calibrated_times_branch = out_tree.Branch("hit_pmt_calibrated_times", hit_pmt_calibrated_times)

        hit_pmt_readout_mask = ROOT.std.vector('int')()
        hit_pmt_readout_mask_branch = out_tree.Branch("hit_pmt_readout_mask", hit_pmt_readout_mask)

        for i, entry in enumerate(tree):
 
            slow_control_dq = good_trigger_mask[i]
            periodic_67ms_dq = periodic_67ms_missing[i]
            
            window_mask = TriggerMask.STABLE_TRIGGER
            if slow_control_dq == 0:
                window_mask |= TriggerMask.SLOW_CONTROL_EXCLUDED

            if periodic_67ms_dq == 0:
                window_mask |= TriggerMask.PERIODIC_67_ISSUE
            
            hit_pmt_calibrated_times.clear()
            hit_pmt_readout_mask.clear()
            
            if i%10_000==0:
                print("On event",i)

            window_data_quality[0] = window_mask.value

            hit_times = np.array(list(entry.hit_pmt_times))
            hit_mpmt_slot = np.array(list(entry.hit_mpmt_slot_ids))
            hit_pmt_pos = np.array(list(entry.hit_pmt_position_ids))
            hit_card_ids = np.array(list(entry.hit_mpmt_card_ids))
            glb_pmt_pos_id = hit_mpmt_slot * 100 + hit_pmt_pos
            timing_offsets = timing_offset_lookup(glb_pmt_pos_id)
            calibrated_times = hit_times - timing_offsets
            
            readout_mask = get_hit_mask_vectorized(hit_card_ids, glb_pmt_pos_id, timing_constant_set, slow_control_stable_channels_set)
            for time, mask in zip(calibrated_times, readout_mask):
                hit_pmt_calibrated_times.push_back(float(time))
                hit_pmt_readout_mask.push_back(bool(mask))
            out_tree.Fill()
        
        out_tree.Write()
        output_file.Close()
        input_file.Close()

    print(f"Finished writing output to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new branch to a ROOT TTree in batches.")
    parser.add_argument("-i","--input_files",nargs='+', help="Path to input ROOT file or files")
    parser.add_argument("-r","--run_number", help="Run Number")
    parser.add_argument("-o","--output_dir", help="Directory to write output file")
    args = parser.parse_args()
    
    #get calibration constants
    calibration_db_interface = CalibrationDBInterface()
    timing_offsets_list, timing_constant_revision_id, timing_constant_insert_time = calibration_db_interface.get_calibration_constants(args.run_number, 0, "timing_offsets", 1)
    timing_offsets_dict = {}
    #load into dict
    for offset in timing_offsets_list:
        timing_offsets_dict[offset['glb_pmt_id']]=offset['data']['timing_offset']
    #set of all channels with calibration constants 
    timing_constant_set = {offset['glb_pmt_id'] for offset in timing_offsets_list}
    
    #get stable list from slow control
    enabled_channels, channel_mask = get_stable_mpmt_list_slow_control(args.run_number)
    stable_channels_card_chan = enabled_channels - channel_mask
    #map slow control data to the 
    mapping = PMTMapping()
    slow_control_stable_channels_set = set() #defined in terms of the slot id and pmt position 
    for ch in stable_channels_card_chan:
        card = ch // 100
        pmt_chan = ch % 100
        slot, pmt_pos = mapping.get_slot_pmt_pos_from_card_pmt_chan(card, pmt_chan)
        slow_control_stable_channels_set.add(100 * slot + pmt_pos)
    
    print("len stable_channels",len(slow_control_stable_channels_set))  
    print("len cal constant channel",len(timing_constant_set))  
    print("Stable channels with no calibration constant",slow_control_stable_channels_set-timing_constant_set)
    print("Unstable channels with calibration constant",timing_constant_set-slow_control_stable_channels_set)
    run_data = get_run_database_data(args.run_number)
    meta_data_json = {}
    meta_data_json["run_number"] = args.run_number
    meta_data_json["run_configuration"] = run_data["trigger_name"]
    meta_data_json["good_wcte_pmts"] = list(timing_constant_set & slow_control_stable_channels_set)
    meta_data_json["wcte_pmts_with_timing_constant"] = list(timing_constant_set)
    meta_data_json["wcte_pmts_slow_control_stable"] = list(slow_control_stable_channels_set)
    meta_data_json["timing_constant_revision_id"] = timing_constant_revision_id
    meta_data_json["timing_constant_insert_time"] = timing_constant_insert_time
    with open(args.output_dir+"/run_"+str(args.run_number)+"_meta_data_json.json", "w") as f:
        json.dump(meta_data_json, f, indent=2)
    
    start = time.time()
    process_data(args.input_files, args.run_number, args.output_dir,timing_offsets_dict, timing_constant_set, slow_control_stable_channels_set)
    end = time.time()
    print(f"Elapsed time: {end - start:.3f} seconds")
