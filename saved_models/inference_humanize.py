import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pretty_midi
from pathlib import Path

from test import *
from model import *
from drum_utils import *
from config import Config

class HumanizeInference:
    def __init__(self, model_name="1st_humanize"):
        self.config = Config()
        self.device = self.config.get_device()
        self.model_name = model_name
        
        # Load models and data
        self.encoder, self.decoder = self.load_models()
        self.test_data = self.load_test_data()
        
    def load_models(self):
        """Load the humanize fine-tuned models"""
        # Try saved_models directory first (your current setup)
        model_paths = [
            Path("saved_models") / f"{self.model_name}_encoder.pth",
            Path("saved_models") / f"{self.model_name}_decoder.pth",
            Path("model") / f"{self.model_name}_encoder.pt",
            Path("model") / f"{self.model_name}_decoder.pt",
            Path("saved_models") / "current_best.pth"  # Your symbolic link
        ]
        
        encoder_path = None
        decoder_path = None
        
        # Find available model files
        for path in model_paths:
            if "encoder" in str(path) and path.exists():
                encoder_path = path
            elif "decoder" in str(path) and path.exists():
                decoder_path = path
            elif "current_best" in str(path) and path.exists():
                # Handle combined model file
                print(f"Found combined model: {path}")
                return self.load_combined_model(path)
        
        if not encoder_path or not decoder_path:
            print("Available model files:")
            for p in [Path("saved_models"), Path("model")]:
                if p.exists():
                    for f in p.glob("*.p*"):
                        print(f"  {f}")
            raise FileNotFoundError(f"Humanize model files not found for {self.model_name}")
        
        # Initialize models
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
        
        print(f"✅ Loaded humanize models:")
        print(f"  - Encoder: {encoder_path}")
        print(f"  - Decoder: {decoder_path}")
        
        return encoder, decoder
    
    def load_combined_model(self, model_path):
        """Load from a combined checkpoint file"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        
        # Load from checkpoint
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            # Assume it's a single model state dict
            encoder.load_state_dict(checkpoint)
            decoder.load_state_dict(checkpoint)
        
        encoder = encoder.to(self.device).eval()
        decoder = decoder.to(self.device).eval()
        
        return encoder, decoder
    
    def load_test_data(self):
        """Load humanize test data"""
        data_path = Path(self.config.PROCESSED_DATA_DIR) / "humanize_test.pkl"
        
        if not data_path.exists():
            print(f"⚠️  Humanize test data not found: {data_path}")
            print("Available data files:")
            data_dir = Path(self.config.PROCESSED_DATA_DIR)
            if data_dir.exists():
                for f in data_dir.glob("*.pkl"):
                    print(f"  {f}")
            return None
            
        with open(data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"📁 Loaded {len(test_data)} humanize test samples")
        return test_data
    
    def generate_single_sample(self, sample_idx=100, output_dir="generated_midi"):
        """Generate a single humanized drum pattern"""
        if self.test_data is None:
            print("❌ No test data available")
            return
        
        if sample_idx >= len(self.test_data):
            sample_idx = len(self.test_data) - 1
            print(f"⚠️  Using sample {sample_idx} instead")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get sample data
        data = self.test_data[sample_idx]  # (32, 54)
        input_data = data[:, :27]  # Original drum pattern
        target_data = data[:, 27:]  # Humanized target
        
        print(f"🎵 Generating humanized pattern from sample {sample_idx}")
        
        # Convert target to MIDI for comparison
        target_tensor = torch.tensor(target_data)
        target_midi = from_tensors_to_midi(target_tensor)
        target_path = Path(output_dir) / f"humanize_target_{sample_idx}.mid"
        target_midi.write(str(target_path))
        
        # Generate with model
        with torch.no_grad():
            input_tensor = torch.tensor(input_data).unsqueeze(0).to(self.device)  # (1, 32, 27)
            seq_len = input_tensor.size(1)
            
            # Encode
            z, mu, std = self.encoder(input_tensor)
            
            # Decode
            output, output_hits, output_velocities, output_offsets = self.decoder(z, seq_len)
        
        # Convert to MIDI
        output_numpy = output.cpu().numpy()
        generated_midi = from_tensors_to_midi(output_numpy[0])
        generated_path = Path(output_dir) / f"humanize_generated_{sample_idx}.mid"
        generated_midi.write(str(generated_path))
        
        # Also save original input
        input_midi = from_tensors_to_midi(input_data)
        input_path = Path(output_dir) / f"humanize_input_{sample_idx}.mid"
        input_midi.write(str(input_path))
        
        print(f"✅ Generated files:")
        print(f"  - Input: {input_path}")
        print(f"  - Generated: {generated_path}")
        print(f"  - Target: {target_path}")
        
        return {
            'input': str(input_path),
            'generated': str(generated_path),
            'target': str(target_path)
        }
    
    def generate_batch(self, num_samples=5, output_dir="generated_midi"):
        """Generate multiple humanized patterns"""
        results = []
        for i in range(num_samples):
            sample_idx = i * 20  # Spread out samples
            result = self.generate_single_sample(sample_idx, output_dir)
            if result:
                results.append(result)
        
        print(f"\n🎉 Generated {len(results)} humanized drum patterns!")
        return results

if __name__ == "__main__":
    # Initialize humanize inference
    humanizer = HumanizeInference()
    
    # Generate a single sample
    humanizer.generate_single_sample(sample_idx=100)
    
    # Or generate multiple samples
    # humanizer.generate_batch(num_samples=3)