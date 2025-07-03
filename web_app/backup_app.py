from flask import Flask, render_template, request, jsonify, Response
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer
from transformers.image_utils import load_image
import io
import base64
import json
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model (based on your template)
model = None
processor = None
DEVICE = None

def load_model():
    """Load SmolVLM model - exactly like your template"""
    global model, processor, DEVICE
    
    model_name = "HuggingFaceTB/SmolVLM-Instruct"
    print(f"Loading SmolVLM model: {model_name}")
    
    # Check if CUDA is available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load processor and model (your exact logic)
    processor = AutoProcessor.from_pretrained(model_name)
    
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None
        )
    except Exception as e:
        if "accelerate" in str(e):
            print("Note: device_map requires 'accelerate' package. Loading without it...")
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            )
            model = model.to(DEVICE)
        else:
            raise e
    
    print("SmolVLM model loaded successfully!")

def generate_streaming_response(image, question):
    """
    Generate streaming response - based on your template but with REAL streaming
    """
    # Create conversation format (exactly like your template)
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template (exactly like your template)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process inputs (exactly like your template)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    # Create streamer for REAL token streaming
    streamer = TextIteratorStreamer(
        processor.tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )
    
    # Generation parameters (same as your template + streamer)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "streamer": streamer,  # This enables REAL streaming
    }
    
    # Generate in background thread so streaming can work
    def generate():
        with torch.no_grad():
            model.generate(**generation_kwargs)
    
    # Start generation
    generation_thread = threading.Thread(target=generate)
    generation_thread.start()
    
    # Stream tokens as they're generated (REAL streaming!)
    for token_text in streamer:
        yield token_text
    
    # Wait for generation to complete
    generation_thread.join()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload - same as before"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process image (same as your template)
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64 for frontend
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Store image
        app.current_image = image
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_str}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/url_image', methods=['POST'])
def upload_url_image():
    """Handle image from URL - same as your template"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Load image from URL (exactly like your template)
        image = load_image(url)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64 for frontend
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Store image
        app.current_image = image
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_str}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error loading image from URL: {str(e)}'}), 500

@app.route('/generate')
def generate_text():
    """Stream text generation - REAL streaming based on your template"""
    prompt = request.args.get('prompt', 'Describe this image in detail.')
    
    if not hasattr(app, 'current_image') or app.current_image is None:
        return Response("Error: No image uploaded", mimetype='text/plain')
    
    def generate():
        try:
            # Use the streaming function based on your template
            for token_text in generate_streaming_response(app.current_image, prompt):
                yield f"data: {json.dumps({'text': token_text})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'complete': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# Non-streaming functions (exactly like your template) for testing
def describe_image(image_path):
    """Non-streaming version - exactly like your template"""
    # Load the image
    if image_path.startswith("http"):
        image = load_image(image_path)
    else:
        image = Image.open(image_path)
    
    # Create the conversation format (exactly like your template)
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    # Apply chat template (exactly like your template)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process inputs (exactly like your template)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    # Generate description (exactly like your template)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
    
    # Decode the generated text (exactly like your template)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

if __name__ == '__main__':
    # Load model on startup (exactly like your template)
    load_model()
    
    # Test with your local image if you want
    print("\n" + "="*50)
    print("Testing non-streaming with your RoboDog.jpg:")
    try:
        description = describe_image("RoboDog.jpg")
        print(f"Description: {description}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Initialize current image
    app.current_image = None
    
    print("\n🌐 Starting SmolVLM Web Demo with REAL streaming...")
    print("📸 Upload images and get real-time AI descriptions!")
    print("🔗 Open your browser to: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)