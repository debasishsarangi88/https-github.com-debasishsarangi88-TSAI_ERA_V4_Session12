---
title: GPT-2 Shakespeare Text Generator
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# 🎭 GPT-2 124M Shakespeare Text Generator

A GPT-2 language model (124M parameters) trained on Shakespeare's complete works to generate text in Shakespearean style.

## Model Details

- **Architecture**: GPT-2 decoder-only transformer
- **Parameters**: 124 million
- **Layers**: 12 transformer blocks
- **Attention Heads**: 12
- **Embedding Dimension**: 768
- **Vocabulary Size**: 50,257 (GPT-2 BPE tokenizer)
- **Context Length**: 1024 tokens

## Training

The model was trained from scratch on Shakespeare's complete works with the following configuration:

- **Training Loss**: < 0.1 (target achieved)
- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Batch Size**: 16
- **Sequence Length**: 128
- **Dataset**: Shakespeare's complete works (~1.1M characters)

## Deployment Instructions

### Step 1: Upload Model Weights

After training your model in Google Colab:

1. Download the `model_final.pt` or `model_best.pt` file from Colab
2. Upload it to this Hugging Face Space repository
3. Place it in the root directory alongside `app.py`

### Step 2: Verify Files

Ensure your Space has these files:
```
huggingface_app/
├── app.py              # Gradio application
├── model.py            # GPT-2 model architecture
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── model_final.pt     # Your trained model weights (upload after training)
```

### Step 3: Configure Space

1. Go to your Hugging Face Space settings
2. Set **SDK** to `gradio`
3. Set **Python version** to `3.10+`
4. The Space will automatically build and deploy

### Step 4: Usage

Once deployed, users can:
- Enter prompts to generate Shakespearean text
- Adjust temperature (0.1-2.0) for creativity control
- Modify top-k sampling (10-100) for diversity
- Generate multiple samples simultaneously
- Use pre-loaded example prompts

## Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Place your trained model checkpoint as model_final.pt

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

## Example Prompts

Try these prompts for best results:

- `First Citizen:`
- `ROMEO:`
- `To be, or not to be,`
- `Once upon a time`
- `All the world's a stage,`

## Parameters Guide

### Temperature
- **0.5-0.7**: More deterministic, coherent text
- **0.8-0.9**: Balanced creativity and coherence (recommended)
- **1.0-1.5**: More creative and diverse outputs
- **1.5+**: Very random and experimental

### Top-K
- **20-30**: Focused, consistent style
- **40-50**: Balanced (recommended)
- **60-100**: More diverse vocabulary

### Max Length
- Adjust based on desired output length
- Longer sequences may take more time
- Recommended: 100-200 tokens

## Technical Details

### Architecture Features
- **Causal Self-Attention**: Ensures autoregressive generation
- **Pre-Layer Normalization**: Improves training stability
- **Residual Connections**: Enables deep network training
- **Weight Tying**: Shares embeddings between input and output layers
- **Scaled Initialization**: Prevents activation explosion in deep networks

### Generation Strategy
- **Top-K Sampling**: Limits sampling to top-k most likely tokens
- **Temperature Scaling**: Controls randomness in generation
- **Autoregressive**: Generates one token at a time conditioned on previous tokens

## Model Performance

The model achieves:
- Training loss < 0.1
- Coherent text generation
- Good understanding of Shakespearean style and vocabulary
- Appropriate use of archaic English patterns

## Citation

If you use this model, please cite:

```
@misc{gpt2-shakespeare-124m,
  author = {Your Name},
  title = {GPT-2 124M Shakespeare Text Generator},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/your-username/gpt2-shakespeare}}
}
```

## License

This model is released for educational and research purposes. The training data (Shakespeare's works) is in the public domain.

## Acknowledgments

- Based on the GPT-2 architecture by OpenAI
- Implementation inspired by Andrej Karpathy's nanoGPT
- Training data: Shakespeare's complete works

