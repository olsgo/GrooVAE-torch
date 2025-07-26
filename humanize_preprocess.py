import pandas as pd
import pretty_midi
import pickle
import os
import sys
import argparse
from drum_utils import *
from config import Config

def main():
    parser = argparse.ArgumentParser(description="Preprocess MIDI data for humanization")
    parser.add_argument("--dataset", default="humanize", help="Dataset name (default: humanize)")
    parser.add_argument("--data-dir", help="Override MIDI data directory")
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # Use configurable paths
    if args.data_dir:
        path = args.data_dir + '/'
        info_csv_path = os.path.join(args.data_dir, 'info.csv')
    else:
        path = config.MIDI_DATA_DIR + '/'
        info_csv_path = os.path.join(config.MIDI_DATA_DIR, 'info.csv')
    
    save_path = config.PROCESSED_DATA_DIR + '/'
    
    # Check if data exists
    if not os.path.exists(info_csv_path):
        print(f"Error: info.csv not found at {info_csv_path}")
        print("Please ensure the dataset is placed in the correct directory.")
        sys.exit(1)
    
    df = pd.read_csv(info_csv_path)
    
    train_data = []
    valid_data = []
    test_data = []
    
    # humanize version
    for _, row in df.iterrows():
        file_name = path + row['midi_filename']
        split = row['split']
        
        try:
            midi_data = pretty_midi.PrettyMIDI(file_name)
            inst = midi_data.instruments[0]
            start_time = midi_data.get_onsets()[0]
            beats = midi_data.get_beats(start_time)
            
            if len(beats) < 2:
                raise ValueError("Insufficient number of beats")
            
            fs = change_fs(beats, target_beats=16)
            
            seqs_tensor, input_tensor, combined_tensor = to_tensors(inst, fs, start_time, humanize=True)
            
            if combined_tensor is not None:  
                if split == 'train':
                    train_data.append(combined_tensor)
                elif split == 'validation':
                    valid_data.append(combined_tensor)
                else:
                    test_data.append(combined_tensor)
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    # Concatenate the tensors along the first dimension (number of windows)
    if train_data:
        train_data = torch.cat(train_data, dim=0)
    if valid_data:
        valid_data = torch.cat(valid_data, dim=0)
    if test_data:
        test_data = torch.cat(test_data, dim=0)
    
    # Save with dataset-specific names
    dataset_name = args.dataset
    
    with open(save_path + f'{dataset_name}_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(save_path + f'{dataset_name}_valid.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
    
    with open(save_path + f'{dataset_name}_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"{dataset_name.title()} preprocessing done!")
    print(f"Train: {len(train_data)} samples")
    print(f"Valid: {len(valid_data)} samples")
    print(f"Test: {len(test_data)} samples")

if __name__ == "__main__":
    main()



