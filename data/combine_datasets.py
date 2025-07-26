import pickle
import torch
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def combine_datasets(dataset1_name, dataset2_name, output_name="combined"):
    """Combine two preprocessed datasets"""
    config = Config()
    processed_dir = config.PROCESSED_DATA_DIR
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        # Load both datasets
        file1 = f"{dataset1_name}_{split}.pkl"
        file2 = f"{dataset2_name}_{split}.pkl"
        
        path1 = os.path.join(processed_dir, file1)
        path2 = os.path.join(processed_dir, file2)
        
        data_combined = []
        
        # Load dataset 1
        if os.path.exists(path1):
            with open(path1, 'rb') as f:
                data1 = pickle.load(f)
                data_combined.extend(data1)
                print(f"Loaded {len(data1)} samples from {dataset1_name} {split}")
        
        # Load dataset 2
        if os.path.exists(path2):
            with open(path2, 'rb') as f:
                data2 = pickle.load(f)
                data_combined.extend(data2)
                print(f"Loaded {len(data2)} samples from {dataset2_name} {split}")
        
        # Shuffle combined data
        import random
        random.shuffle(data_combined)
        
        # Save combined dataset
        output_file = f"{output_name}_{split}.pkl"
        output_path = os.path.join(processed_dir, output_file)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_combined, f)
        
        print(f"Saved {len(data_combined)} combined samples to {output_file}")

if __name__ == "__main__":
    combine_datasets("toontrack", "qapt", "toontrack_qapt_combined")