import torch
import numpy as np
import os
import pickle
from pathlib import Path
import argparse
from datetime import datetime

# Import your modules
from model import Encoder, Decoder
from drum_utils import from_tensors_to_midi
from config import Config

class MIDIGenerator:
    def __init__(self, model_name="1st_humanize", device=None):
        self.config = Config()
        self.device = device or self.config.get_device()
        self.model_name = model_name
        
        # Load models
        self.encoder, self.decoder = self.load_models()
        
    def load_models(self):
        """Load the fine-tuned encoder and decoder models"""
        model_dir = Path("model")
        encoder_path = model_dir / f"{self.model_name}_encoder.pt"
        decoder_path = model_dir / f"{self.model_name}_decoder.pt"
        
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(f"Model files not found for {self.model_name}")
        
        # Initialize models with config parameters
        encoder = Encoder(
            self.config.ENC_INPUT_SIZE,
            self.config.ENC_HIDDEN_SIZE, 
            self.config.ENC_LATENT_DIM
        )
        decoder = Decoder(
            self.config.DEC_INPUT_SIZE,
            self.config.DEC_HIDDEN_SIZE,
            self.config.DEC_OUTPUT_SIZE
        )
        
        # Load weights
        encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        
        # Move to device and set to eval mode
        encoder = encoder.to(self.device).eval()
        decoder = decoder.to(self.device).eval()
        
        print(f"✅ Loaded models: {encoder_path.name} and {decoder_path.name}")
        return encoder, decoder
    
    def load_test_data(self, data_type="humanize"):
        """Load test data for input patterns"""
        data_path = Path(self.config.PROCESSED_DATA_DIR) / f"{data_type}_test.pkl"
        
        if not data_path.exists():
            print(f"⚠️  Test data not found: {data_path}")
            return None
            
        with open(data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"📁 Loaded {len(test_data)} test samples from {data_path.name}")
        return test_data
    
    def generate_from_latent(self, num_samples=10, seq_len=32, temperature=1.0, hit_threshold=0.5):
        """Generate MIDI from random latent vectors"""
        generated_patterns = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Sample random latent vector
                z = torch.randn(1, self.config.ENC_LATENT_DIM).to(self.device)
                
                # Generate pattern
                output, _, _, _ = self.decoder(
                    z, seq_len, 
                    temperature=temperature, 
                    hit_threshold=hit_threshold
                )
                
                pattern = output.cpu().numpy()[0]  # Remove batch dimension
                generated_patterns.append(pattern)
                
        return generated_patterns
    
    def generate_from_input(self, test_data, num_samples=10, temperature=1.0, hit_threshold=0.5):
        """Generate MIDI from input patterns (reconstruction + variation)"""
        if test_data is None:
            return []
            
        generated_patterns = []
        input_patterns = []
        
        # Select random samples from test data
        indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
        
        with torch.no_grad():
            for idx in indices:
                data = test_data[idx]
                
                if len(data.shape) == 2 and data.shape[1] == 54:  # Combined input+target format
                    input_data = data[:, :27]  # First 27 features are input
                else:
                    input_data = data  # Assume it's already input format
                
                # Encode input
                input_tensor = torch.tensor(input_data).unsqueeze(0).float().to(self.device)
                z, mu, std = self.encoder(input_tensor)
                
                # Generate with some variation by sampling from latent distribution
                z_varied = mu + torch.randn_like(std) * torch.exp(0.5 * std) * 0.5  # Reduced variance
                
                # Decode
                output, _, _, _ = self.decoder(
                    z_varied, input_tensor.size(1),
                    temperature=temperature,
                    hit_threshold=hit_threshold
                )
                
                pattern = output.cpu().numpy()[0]
                generated_patterns.append(pattern)
                input_patterns.append(input_data)
                
        return generated_patterns, input_patterns
    
    def interpolate_patterns(self, test_data, num_interpolations=5, steps=10):
        """Generate interpolations between two patterns"""
        if test_data is None or len(test_data) < 2:
            return []
            
        interpolated_patterns = []
        
        with torch.no_grad():
            for i in range(num_interpolations):
                # Select two random patterns
                idx1, idx2 = np.random.choice(len(test_data), 2, replace=False)
                
                data1 = test_data[idx1]
                data2 = test_data[idx2]
                
                if len(data1.shape) == 2 and data1.shape[1] == 54:
                    input1 = data1[:, :27]
                    input2 = data2[:, :27]
                else:
                    input1 = data1
                    input2 = data2
                
                # Encode both patterns
                input1_tensor = torch.tensor(input1).unsqueeze(0).float().to(self.device)
                input2_tensor = torch.tensor(input2).unsqueeze(0).float().to(self.device)
                
                z1, mu1, _ = self.encoder(input1_tensor)
                z2, mu2, _ = self.encoder(input2_tensor)
                
                # Interpolate in latent space
                for step in range(steps):
                    alpha = step / (steps - 1)
                    z_interp = (1 - alpha) * mu1 + alpha * mu2
                    
                    # Decode interpolated latent
                    output, _, _, _ = self.decoder(z_interp, input1_tensor.size(1))
                    pattern = output.cpu().numpy()[0]
                    interpolated_patterns.append(pattern)
                    
        return interpolated_patterns
    
    def save_midi_files(self, patterns, output_dir, prefix="generated", input_patterns=None):
        """Save patterns as MIDI files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        for i, pattern in enumerate(patterns):
            # Generate MIDI
            midi_data = from_tensors_to_midi(pattern, steps_per_quarter=4)
            
            # Save file
            filename = f"{prefix}_{timestamp}_{i+1:03d}.mid"
            filepath = output_path / filename
            midi_data.write(str(filepath))
            saved_files.append(str(filepath))
            
            # Also save input pattern if provided
            if input_patterns is not None and i < len(input_patterns):
                input_midi = from_tensors_to_midi(input_patterns[i], steps_per_quarter=4)
                input_filename = f"{prefix}_input_{timestamp}_{i+1:03d}.mid"
                input_filepath = output_path / input_filename
                input_midi.write(str(input_filepath))
                saved_files.append(str(input_filepath))
        
        return saved_files

def main():
    parser = argparse.ArgumentParser(description="Generate MIDI files from fine-tuned GrooVAE model")
    parser.add_argument("--model", default="1st_humanize", help="Model name (default: 1st_humanize)")
    parser.add_argument("--output-dir", default="generated_midi", help="Output directory")
    parser.add_argument("--num-random", type=int, default=20, help="Number of random generations")
    parser.add_argument("--num-variations", type=int, default=15, help="Number of input variations")
    parser.add_argument("--num-interpolations", type=int, default=10, help="Number of interpolation sequences")
    parser.add_argument("--data-type", default="humanize", help="Data type")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--hit-threshold", type=float, default=0.5, help="Hit threshold for drum hits")
    
    args = parser.parse_args()
    
    print(f"🎵 Starting MIDI generation with model: {args.model}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Initialize generator
    generator = MIDIGenerator(model_name=args.model)
    
    # Load test data
    test_data = generator.load_test_data(args.data_type)
    
    all_saved_files = []
    
    # 1. Generate from random latent vectors
    print(f"\n🎲 Generating {args.num_random} random patterns...")
    random_patterns = generator.generate_from_latent(
        num_samples=args.num_random,
        temperature=args.temperature,
        hit_threshold=args.hit_threshold
    )
    files = generator.save_midi_files(random_patterns, args.output_dir, "random")
    all_saved_files.extend(files)
    print(f"✅ Saved {len(random_patterns)} random patterns")
    
    # 2. Generate variations from input patterns
    if test_data is not None:
        print(f"\n🔄 Generating {args.num_variations} input variations...")
        variation_patterns, input_patterns = generator.generate_from_input(
            test_data,
            num_samples=args.num_variations,
            temperature=args.temperature,
            hit_threshold=args.hit_threshold
        )
        files = generator.save_midi_files(
            variation_patterns, args.output_dir, "variation", input_patterns
        )
        all_saved_files.extend(files)
        print(f"✅ Saved {len(variation_patterns)} variations (with input references)")
        
        # 3. Generate interpolations
        print(f"\n🌈 Generating {args.num_interpolations} interpolation sequences...")
        interpolated_patterns = generator.interpolate_patterns(
            test_data, num_interpolations=args.num_interpolations
        )
        files = generator.save_midi_files(interpolated_patterns, args.output_dir, "interpolation")
        all_saved_files.extend(files)
        print(f"✅ Saved {len(interpolated_patterns)} interpolated patterns")
    
    # Summary
    print(f"\n🎉 Generation complete!")
    print(f"📊 Total files generated: {len(all_saved_files)}")
    print(f"📁 Files saved to: {Path(args.output_dir).absolute()}")
    print(f"\n🎧 Ready for Ableton Live inspection!")
    
    # Show some example files
    print("\n📝 Sample generated files:")
    for file in all_saved_files[:5]:
        print(f"  • {Path(file).name}")
    if len(all_saved_files) > 5:
        print(f"  ... and {len(all_saved_files) - 5} more")

if __name__ == "__main__":
    main()