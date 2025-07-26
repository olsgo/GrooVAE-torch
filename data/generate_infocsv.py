import os
import pandas as pd
import random
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate info.csv for MIDI dataset')
    parser.add_argument('midi_dir', nargs='?', default='midi_data/groove', 
                       help='Path to MIDI directory (default: midi_data/groove)')
    parser.add_argument('--dataset', help='Dataset name for output file naming')
    
    args = parser.parse_args()
    midi_dir = args.midi_dir
    
    # Find all MIDI files
    midi_files = []
    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                # Get relative path from the midi_dir
                rel_path = os.path.relpath(os.path.join(root, file), midi_dir)
                midi_files.append(rel_path)
    
    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return
    
    # Create train/validation/test splits (80/10/10)
    random.shuffle(midi_files)
    total = len(midi_files)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    data = []
    for i, filename in enumerate(midi_files):
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'validation'
        else:
            split = 'test'
        
        data.append({
            'midi_filename': filename,
            'split': split
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_file = os.path.join(midi_dir, 'info.csv')
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(midi_files)} MIDI files")

if __name__ == '__main__':
    main()