import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import glob
from matplotlib import colors
import matplotlib.colors as colors
import json
from analysis_tools import WaveformProcessingmPMT
from analysis_tools.pulse_finding import do_pulse_finding_vect
import time
import argparse
import os

def do_hit_processing(waveforms, waveform_times, waveform_cards, waveform_channels, wf_length):
    
    if not isinstance(waveforms, np.ndarray):
        print("waveforms are not being passed as numpy array",type(waveforms))
    #run on self-trigger data 

    #slice up the big waveform array into samples we are interested in
    #for each hit where do we want to start the sample in the waveform
    slice_len = 12
    peak_position = 8

    min_peak_sample = peak_position
    max_peak_sample = wf_length-(slice_len-peak_position)
    
    verbose = False
        
    # ============================================================
    # SECTION 1) do the pulse finding
    # ============================================================
    start_time = time.time()
    hit_indices = do_pulse_finding_vect(waveforms) ##indices of the pulse in each waveform
    hit_indices = ak.Array(hit_indices)
    #filter the found hits to avoid either end of the samples
    hit_indices = hit_indices[(hit_indices>=min_peak_sample) & (hit_indices<=max_peak_sample)]
    
    end_time = time.time()
    if verbose: print(f"Pulse finding took {end_time - start_time:.6f} seconds")

    # ============================================================
    # SECTION 2) now want to make an array of waveforms corresponding 
    #    to each hit with the info needed for CFD and charge calculation
    # ============================================================
    #get the index of the waveform in each case that corresponds to the hit and flatten
    row_wf_index = ak.local_index(hit_indices, axis=0) #returns the index of outer most array - the dimension of wfs
    # essentially[ wf_0, wf_1 wf_2 ...] since this is what was fed into the do_pulse_finding_vect
    hit_wf_index, _ = ak.broadcast_arrays(row_wf_index[:, None], hit_indices)
    # now using broadcast_arrays we broadcast that shape onto hit_indices to produce an array like [[wf_1,wf_1],[wf_2],..] with n hits as second dimension
    #now we flatten
    hit_wf_index = ak.to_numpy(ak.flatten(hit_wf_index)) # index of the waveform for each hit
    hit_indices_flat = ak.to_numpy(ak.flatten(hit_indices)) # list of where the hits are in the waveform - flat one for each hit

    #make a full array of the waveforms for each flat hit for later 
    each_hit_waveform = waveforms[hit_wf_index] #length of n hits the corresponding full waveform
    
    
    local_idx = np.arange(slice_len)
    start_sample = hit_indices_flat-peak_position
    
    #this makes a 2d array using broadcasting which sample to pick for each element 
    slice_idx = start_sample[:, None] + local_idx[None, :]  # shape (num_hits, slice_len)

    ## hit_idx is shape (n_hits, 1) slice_idx is shape (num_hits, slice_len), they are broadcasted to (num_hits, slice_len)
    # waveform_samples will be shape (num_hits, slice_len) each element waveform_samples[i, j] is each_hit_waveform[hit_idx[i, j],slice_idx[i,j]]
    hit_idx = np.arange(len(hit_indices_flat))[:, None]
    waveform_samples = each_hit_waveform[hit_idx, slice_idx]

    # ============================================================
    # SECTION 3) Now do the charge and time calculations
    # ============================================================
    offlineProcessing = WaveformProcessingmPMT()
    start_time = time.time()
    _,cfd_time_vector_corr,_,_ = offlineProcessing.cfd_vectorized(waveform_samples)
    found_cfd_time = start_sample + cfd_time_vector_corr #relative to start of the waveform
    found_hit_time = waveform_times[hit_wf_index] + (found_cfd_time*8.0) #relative to start of the window
    end_time = time.time()
    if verbose: print(f"Function vectorised took {end_time - start_time:.6f} seconds")

    found_hit_charge = offlineProcessing.charge_vectorized_mPMT_method(waveform_samples,peak_position)
    
    found_hit_card = waveform_cards[hit_wf_index]
    found_hit_chan = waveform_channels[hit_wf_index]

    return found_hit_charge, found_hit_time, found_hit_card, found_hit_chan, hit_wf_index, hit_indices_flat 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new branch to a ROOT TTree in batches.")
    parser.add_argument("-i","--input_files",nargs='+', help="Path to input ROOT file or files")
    # parser.add_argument("-r","--run_number", help="Run Number")
    parser.add_argument("-o","--output_dir", help="Directory to write output file")
    args = parser.parse_args()

    for input_file_name in args.input_files:
        # Construct output path
        base = os.path.splitext(os.path.basename(input_file_name))[0]
        new_filename = f"{base}_processed_waveforms.root" 
        os.makedirs(args.output_dir, exist_ok=True)
        output_file_name = os.path.join(args.output_dir, new_filename)
        # print(input_file_name, "->", output_file_name)
        
        with uproot.recreate(output_file_name) as outfile:
            # Create a TTree with two variable-length branches
            tree = outfile.mktree("ProcessedWaveforms", {
                "hit_time": "var * float64",
                "hit_charge": "var * float64",
                "hit_card": "var * int32",
                "hit_chan": "var * int32",
                "hit_waveform_index": "var * int32",
                "hit_peak_sample": "var * int32",
                "readout_number": "int32",
                "trigger_time": "float64",
                "missing_trigger_flag": "int32"
            })
            
            batch_size = 1000
            
            with uproot.open(input_file_name) as root_file:
                input_tree = root_file["WCTEReadoutWindows"]
                total_entries = input_tree.num_entries
                all_branches = input_tree.keys()
                
                #open the input file in batches
                for start in range(0, total_entries, batch_size):
                        
                    stop = min(start + batch_size, total_entries)
                    print(f"Loading entries {start} → {stop}")
                    start_batch = time.time()
                    #open the events in a batch
                    events = input_tree.arrays(all_branches,library="ak", entry_start=start, entry_stop=stop)
                    end = time.time()
                    print(f"Time load batch data: {end - start_batch:.6f} seconds")

                    batch_hit_time = []
                    batch_hit_charge = []
                    batch_hit_card = []
                    batch_hit_chan = []
                    batch_hit_waveform_index = []
                    batch_hit_peak_sample = []
                    batch_event_readout_number = [] 
                    batch_event_trigger_time = [] 
            
                    for iev, event in enumerate(events):
                        if iev%100==0:
                            print("On iev",iev)
                        
                        wf_start   = event["pmt_waveform_times"]
                        wf_waveforms    = event["pmt_waveforms"]
                        wf_end     = wf_start + 8.0 * ak.num(wf_waveforms)
                        wf_card    = event["pmt_waveform_mpmt_card_ids"]
                        wf_chan    = event["pmt_waveform_pmt_channel_ids"]
                        wf_card    = event["pmt_waveform_mpmt_card_ids"]
                        wf_chan    = event["pmt_waveform_pmt_channel_ids"]
                        event_readout_number = event["readout_number"]
                        
                        try:
                            wf_process = np.array(wf_waveforms)
                        except:
                            print("Problem converting waveforms to numpy array, probably caused by different lengths of waveforms in event ",iev)
                            wf_lengths_debug = ak.num(wf_waveforms).to_numpy()
                            debug_wf_process_card = np.array(wf_card)
                            print("Waveform lengths in event:",np.unique(wf_lengths_debug))
                            print("Cards ",np.unique(debug_wf_process_card[wf_lengths_debug!=64]),"have waveforms != 64 samples")
                            raise Exception("Cannot process event with different length waveforms")    
                        wf_length = wf_process.shape[1]
                        wf_process_start = np.array(wf_start)
                        wf_process_card = np.array(wf_card)
                        wf_process_chan = np.array(wf_chan)
                        
                        #outputs a list of hits and the wf_index of the hits hit_wf_index
                        start = time.time()
                        found_hit_charge, found_hit_time, found_hit_card, found_hit_chan, hit_wf_index, hit_local_indices = do_hit_processing(wf_process,wf_process_start,wf_process_card,wf_process_chan,wf_length)
                        end = time.time()
                        if iev==234: print(f"Estimated hit processing: {batch_size*(end - start):.6f} seconds")
                        
                        start = time.time()
                        batch_hit_time.append(found_hit_time.tolist())
                        batch_hit_charge.append(found_hit_charge.tolist())
                        batch_hit_card.append(found_hit_card.tolist())
                        batch_hit_chan.append(found_hit_chan.tolist())
                        batch_hit_waveform_index.append(hit_wf_index.tolist())
                        batch_hit_peak_sample.append(hit_local_indices.tolist())
                        batch_event_readout_number.append(int(event_readout_number))
                        
                        end = time.time()
                        if iev==234: print(f"Time append processing: {1000.0*(end - start):.6f} seconds")
                    #finished processing batch
                    start = time.time()
                    batch_hit_time = ak.Array(batch_hit_time)
                    batch_hit_charge = ak.Array(batch_hit_charge)
                    batch_hit_card = ak.Array(batch_hit_card)
                    batch_hit_chan = ak.Array(batch_hit_chan)
                    # print("Batch batch_hit_chan",ak.type(batch_hit_chan))
                    batch_hit_waveform_index = ak.Array(batch_hit_waveform_index)
                    batch_hit_peak_sample = ak.Array(batch_hit_peak_sample)
                    batch_event_readout_number = np.array(batch_event_readout_number, dtype=np.int32)

                    #find trigger time for each event
                    batch_event_trigger_time = ak.firsts(batch_hit_time[(batch_hit_card==131) & (batch_hit_chan==0)])
                    missing_trigger_mask = ak.is_none(batch_event_trigger_time).to_numpy()
                    if np.sum(missing_trigger_mask)>0:
                        print("Missing triggers",np.sum(missing_trigger_mask))
                    missing_trigger_flag = np.zeros_like(missing_trigger_mask)
                    missing_trigger_flag[missing_trigger_mask]=1
                    batch_event_trigger_time = ak.fill_none(batch_event_trigger_time, 0)
                    batch_hit_time = batch_hit_time - batch_event_trigger_time
                    batch_event_trigger_time = batch_event_trigger_time.to_numpy()
                    
                    
                    mid_time = time.time()
                    tree.extend({
                        "hit_time": (batch_hit_time),
                        "hit_charge": (batch_hit_charge),
                        "hit_card": (batch_hit_card),
                        "hit_chan": (batch_hit_chan),
                        "hit_waveform_index": (batch_hit_waveform_index),
                        "hit_peak_sample": (batch_hit_peak_sample),
                        "readout_number": batch_event_readout_number,
                        "trigger_time": batch_event_trigger_time,
                        "missing_trigger_flag": missing_trigger_flag
                    })
                    end = time.time()
                    print(f"Time ak.Array: {mid_time - start:.6f} seconds")
                    print(f"Time batch write uproot: {end - mid_time:.6f} seconds")
                    end_batch = time.time()
                    print(f"Time for full batch: {end_batch - start_batch:.6f} seconds")

        