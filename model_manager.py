#!/usr/bin/env python3
"""
GrooVAE Model Management Utility
"""

import os
import argparse
from config import MODEL_SAVE_DIR, MODEL_ALIASES, resolve_model_path

def list_models():
    """List all available models and aliases"""
    print("\n🎵 GrooVAE Model Registry\n")
    
    print("📁 Available Models:")
    for file in sorted(os.listdir(MODEL_SAVE_DIR)):
        if file.endswith('.pth'):
            path = os.path.join(MODEL_SAVE_DIR, file)
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  • {file:<35} ({size:.1f}MB)")
    
    print("\n🔗 Active Aliases:")
    for alias, target in MODEL_ALIASES.items():
        print(f"  • {alias:<15} → {target}")

def create_alias(alias_name, target_file):
    """Create a new symbolic link alias"""
    target_path = os.path.join(MODEL_SAVE_DIR, target_file)
    alias_path = os.path.join(MODEL_SAVE_DIR, f"{alias_name}.pth")
    
    if not os.path.exists(target_path):
        print(f"❌ Target file {target_file} does not exist")
        return False
    
    if os.path.exists(alias_path):
        print(f"⚠️  Alias {alias_name}.pth already exists")
        return False
    
    os.symlink(target_file, alias_path)
    print(f"✅ Created alias: {alias_name}.pth → {target_file}")
    return True

def test_model(model_name):
    """Test if a model can be loaded"""
    model_path = resolve_model_path(model_name)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loads successfully: {model_path}")
        
        if 'epoch' in checkpoint:
            print(f"   📊 Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"   📉 Loss: {checkpoint['loss']:.4f}")
            
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GrooVAE Model Manager")
    parser.add_argument('action', choices=['list', 'alias', 'test'], 
                       help='Action to perform')
    parser.add_argument('--alias', help='Alias name for creating links')
    parser.add_argument('--target', help='Target file for alias')
    parser.add_argument('--model', help='Model name to test')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_models()
    elif args.action == 'alias':
        if not args.alias or not args.target:
            print("❌ Both --alias and --target required")
        else:
            create_alias(args.alias, args.target)
    elif args.action == 'test':
        if not args.model:
            print("❌ --model required for testing")
        else:
            test_model(args.model)