"""
Create a deployment-only checkpoint (without optimizer state)
This reduces file size from ~1.4GB to ~500MB
"""

import torch
import os

def create_deployment_checkpoint(input_path, output_path):
    """
    Extract only model weights from full training checkpoint
    
    Args:
        input_path: Path to full checkpoint (with optimizer state)
        output_path: Path to save deployment checkpoint (model only)
    """
    print(f"Loading full checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Show what's in the checkpoint
    print(f"\nCheckpoint keys: {checkpoint.keys()}")
    
    # Original size
    original_size = os.path.getsize(input_path) / (1024**3)
    print(f"Original checkpoint size: {original_size:.2f} GB")
    
    # Create deployment checkpoint (only essentials)
    deployment_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'loss': checkpoint.get('loss', 0.0),
        'step': checkpoint.get('step', 0),
    }
    
    # Save deployment checkpoint
    print(f"\nSaving deployment checkpoint to: {output_path}")
    torch.save(deployment_checkpoint, output_path)
    
    # New size
    new_size = os.path.getsize(output_path) / (1024**3)
    print(f"Deployment checkpoint size: {new_size:.2f} GB")
    print(f"Size reduction: {original_size - new_size:.2f} GB ({(1 - new_size/original_size)*100:.1f}% smaller)")
    
    print("\n✅ Deployment checkpoint created successfully!")
    print(f"You can now use '{output_path}' for deployment")

if __name__ == "__main__":
    # Paths
    input_checkpoint = "huggingface_app/model_final.pt"
    output_checkpoint = "huggingface_app/model_deployment.pt"
    
    # Create deployment checkpoint
    create_deployment_checkpoint(input_checkpoint, output_checkpoint)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Test the deployment checkpoint:")
    print("   python test_deployment_checkpoint.py")
    print("\n2. Replace the large file:")
    print("   mv huggingface_app/model_deployment.pt huggingface_app/model_final.pt")
    print("\n3. Update git and push:")
    print("   git add huggingface_app/model_final.pt")
    print("   git commit -m 'Replace with deployment-only checkpoint (500MB)'")
    print("   git push")

