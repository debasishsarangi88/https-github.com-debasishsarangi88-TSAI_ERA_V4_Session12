"""
Test the deployment checkpoint to ensure it works correctly
"""

import torch
import sys
sys.path.append('huggingface_app')
from model import GPT, GPTConfig
import tiktoken

def test_checkpoint(checkpoint_path):
    """Test if checkpoint loads and generates text correctly"""
    
    print(f"Testing checkpoint: {checkpoint_path}")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    config = GPTConfig()
    model = GPT(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"   Step: {checkpoint.get('step', 'N/A')}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Test generation
    print("\n2. Testing text generation...")
    enc = tiktoken.get_encoding('gpt2')
    
    prompt = "First Citizen:"
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        # Generate 50 tokens
        for _ in range(50):
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / 0.8  # temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)
    
    generated_text = enc.decode(tokens[0].tolist())
    print(f"   ✅ Generation successful!")
    print(f"\n   Generated text:\n   {'-'*56}")
    print(f"   {generated_text}")
    print(f"   {'-'*56}")
    
    print("\n✅ All tests passed! Checkpoint is working correctly.")
    return True

if __name__ == "__main__":
    import os
    
    # Test deployment checkpoint
    deployment_path = "huggingface_app/model_deployment.pt"
    
    if os.path.exists(deployment_path):
        try:
            test_checkpoint(deployment_path)
        except Exception as e:
            print(f"\n❌ Error testing checkpoint: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Checkpoint not found: {deployment_path}")
        print("Run 'python create_deployment_checkpoint.py' first")

