import gradio as gr
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

# Load BLIP model for initial caption generation
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load GPT-2 model for caption refinement
print("Loading GPT-2 model...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)
gpt2_model.to(device)
print(f"Using device: {device}")

def generate_caption(image):
    """Generate caption using BLIP and refine it with GPT-2"""
    start_time = time.time()
    
    # Process the image for BLIP
    inputs = blip_processor(image, return_tensors="pt").to(device)
    
    # Generate caption with BLIP
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=30)
        blip_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Prepare prompt for GPT-2 refinement
    prompt = f"An image shows {blip_caption}. A detailed description would be: "
    
    # Tokenize and generate refined caption with GPT-2
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
    
    refined_caption = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the enhanced part (after the prompt)
    enhanced_caption = refined_caption.split("A detailed description would be:")[-1].strip()
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return blip_caption, enhanced_caption, f"Processing time: {processing_time:.2f} seconds"

# Create Gradio interface
with gr.Blocks(title="AI Image Captioning App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI-Powered Image Captioning")
    gr.Markdown("This app generates captions for your images using BLIP for initial caption generation and GPT-2 for enhancement.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload an image")
            caption_button = gr.Button("Generate Captions", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Group():
                blip_caption_output = gr.Textbox(label="BLIP Caption (Base)")
                enhanced_caption_output = gr.Textbox(label="GPT-2 Enhanced Caption")
                processing_time = gr.Textbox(label="Performance")
    
    caption_button.click(
        generate_caption,
        inputs=[image_input],
        outputs=[blip_caption_output, enhanced_caption_output, processing_time]
    )
    
    gr.Markdown("## How it works")
    gr.Markdown("""
    1. **BLIP Model**: Extracts visual features and generates a basic caption
    2. **GPT-2 Model**: Enhances the basic caption to make it more detailed and descriptive
    3. **Both captions are displayed** so you can see the improvement from the enhancement process
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()