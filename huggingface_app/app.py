"""
GPT-2 124M Shakespeare Text Generator - Hugging Face Spaces App
This app uses a trained GPT-2 model to generate Shakespearean text.
"""

import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig
import os

# Device setup
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using device: {device}")

# Load model
def load_model(checkpoint_path='model_final.pt'):
    """Load the trained model from checkpoint"""
    config = GPTConfig()
    model = GPT(config)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded! Training loss: {checkpoint['loss']:.6f}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
    
    model.to(device)
    model.eval()
    return model

# Initialize model and tokenizer
model = load_model()
enc = tiktoken.get_encoding('gpt2')

def generate_text(
    prompt,
    max_length=150,
    temperature=0.8,
    top_k=50,
    num_samples=1
):
    """
    Generate text based on the given prompt
    
    Args:
        prompt: Input text to start generation
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider for sampling
        num_samples: Number of samples to generate
    
    Returns:
        Generated text samples
    """
    if not prompt:
        return "Please provide a prompt!"
    
    # Encode the prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1)
    x = tokens.to(device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        while x.size(1) < max_length:
            # Forward pass
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
            
            # Sample from the filtered distribution
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            
            # Append to sequence
            x = torch.cat((x, xcol), dim=1)
    
    # Decode and return
    outputs = []
    for i in range(num_samples):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        outputs.append(decoded)
    
    return "\n\n" + "="*80 + "\n\n".join(outputs) if num_samples > 1 else outputs[0]

# Example prompts
examples = [
    ["First Citizen:", 150, 0.8, 50, 1],
    ["ROMEO:", 200, 0.9, 40, 1],
    ["To be, or not to be,", 150, 0.7, 50, 1],
    ["Once upon a time", 150, 1.0, 50, 1],
]

# Create Gradio interface
with gr.Blocks(title="GPT-2 Shakespeare Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 GPT-2 124M Shakespeare Text Generator
    
    This model was trained on Shakespeare's complete works to generate text in Shakespearean style.
    
    **Model Details:**
    - Architecture: GPT-2 (124M parameters)
    - Training: Decoder-only transformer
    - Dataset: Shakespeare's complete works
    - Training loss: < 0.1
    
    **Instructions:**
    1. Enter a prompt to start the generation
    2. Adjust parameters to control the output
    3. Click "Generate" to create text
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here (e.g., 'First Citizen:', 'ROMEO:', etc.)",
                lines=3
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_length_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Max Length",
                    info="Maximum number of tokens to generate"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random, Lower = more deterministic"
                )
                
                top_k_slider = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-K",
                    info="Number of top tokens to sample from"
                )
                
                num_samples_slider = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=1,
                    step=1,
                    label="Number of Samples",
                    info="Generate multiple variations"
                )
            
            generate_btn = gr.Button("Generate Text", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=20,
                max_lines=30
            )
    
    gr.Examples(
        examples=examples,
        inputs=[prompt_input, max_length_slider, temperature_slider, top_k_slider, num_samples_slider],
        label="Example Prompts"
    )
    
    gr.Markdown("""
    ---
    ### Tips for Best Results:
    - Use character names or dialogue markers (e.g., "ROMEO:", "First Citizen:")
    - Start with famous quotes or phrases from Shakespeare
    - Adjust temperature: 0.7-0.9 for coherent text, 1.0+ for creative variations
    - Lower top-k (20-30) for more focused text, higher (50-80) for diversity
    
    ### About the Model:
    This is a GPT-2 model with 124M parameters trained from scratch on Shakespeare's works.
    The model uses:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimensions
    - Causal self-attention mechanism
    """)
    
    # Connect the button to the function
    generate_btn.click(
        fn=generate_text,
        inputs=[
            prompt_input,
            max_length_slider,
            temperature_slider,
            top_k_slider,
            num_samples_slider
        ],
        outputs=output_text
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

