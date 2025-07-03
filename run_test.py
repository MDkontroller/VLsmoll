import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Load the SmolVLM model and processor  
model_name = "HuggingFaceTB/SmolVLM-Instruct"
print(f"Loading SmolVLM model: {model_name}")
#print("This will download the model files on first run (may take a few minutes)...")

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
processor = AutoProcessor.from_pretrained(model_name)

try:
    # Try with device_map (requires accelerate)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
except Exception as e:
    if "accelerate" in str(e):
        print("Note: device_map requires 'accelerate' package. Loading without it...")
        # Fallback: load without device_map
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        )
        model = model.to(DEVICE)
    else:
        raise e

print("SmolVLM model loaded successfully!")

def describe_image(image_path):
    """
    Generate a description for an image using SmolVLM
    """
    # Load the image
    if image_path.startswith("http"):
        image = load_image(image_path)
    else:
        image = Image.open(image_path)
    
    # Create the conversation format that SmolVLM expects
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process inputs - NOTE: Use text= and images= keyword arguments
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    # Generate description
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
    
    # Decode the generated text
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def ask_about_image(image_path, question):
    """
    Ask a specific question about an image using SmolVLM
    """
    # Load the image
    if image_path.startswith("http"):
        image = load_image(image_path)
    else:
        image = Image.open(image_path)
    
    # Create the conversation format
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process inputs - Use keyword arguments
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
    
    # Decode the generated text
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

# Test with different types of images
if __name__ == "__main__":
    # Example 1: Local image
    try:
        image_path = "RoboDog.jpg"  # Replace with your image path
        description = describe_image(image_path)
        print(f"Local Image Description: {description}")
    except Exception as e:
        print(f"Local image error: {e}")
    
    # Example 2: Online image  
    try:
        online_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        description = describe_image(online_image)
        print(f"\nOnline Image Description: {description}")
    except Exception as e:
        print(f"Online image error: {e}")
    
    # Example 3: Ask a specific question
    try:
        question = "What is the main subject of this image?"
        answer = ask_about_image(online_image, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    except Exception as e:
        print(f"Question error: {e}")

print("\n" + "="*50)
print("SmolVLM is ready! Try these examples:")
print("- describe_image('your_image.jpg')")
print("- ask_about_image('your_image.jpg', 'What colors do you see?')")
print("- Any image URL or local file path works!")
print("="*50)