import os
import pandas as pd
import random

# Set the path to your MIDI directory
midi_dir = 'midi_data/groove'  # or 'groovae_data' if you chose option 2

# Find all MIDI files
midi_files = []
for root, dirs, files in os.walk(midi_dir):
    for file in files:
        if file.endswith('.mid') or file.endswith('.midi'):
            # Get relative path from the midi_dir
            rel_path = os.path.relpath(os.path.join(root, file), midi_dir)
            midi_files.append(rel_path)

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
df.to_csv(os.path.join(midi_dir, 'info.csv'), index=False)
print(f"Created info.csv with {len(midi_files)} MIDI files")