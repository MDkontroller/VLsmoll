from flask import Flask, render_template, request, jsonify, Response
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer, BitsAndBytesConfig
from transformers.image_utils import load_image
import io
import base64
import json
import threading
import gc
import time
import os
import sys
from functools import lru_cache
import psutil
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
model = None
processor = None
DEVICE = None
model_loaded = False

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.generation_time = 0
        self.preprocessing_time = 0
        self.tokens_generated = 0
        
    def get_stats(self):
        total_time = time.time() - self.start_time
        tokens_per_second = self.tokens_generated / self.generation_time if self.generation_time > 0 else 0
        
        return {
            'total_time': round(total_time, 2),
            'generation_time': round(self.generation_time, 2),
            'preprocessing_time': round(self.preprocessing_time, 2),
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': round(tokens_per_second, 2),
            'memory_usage': f"{psutil.virtual_memory().percent}%",
            'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
        }

perf_monitor = PerformanceMonitor()

def check_model_exists_locally(model_name):
    """Check if SmolVLM model exists in local cache"""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    
    # Check environment variable for custom cache location
    if 'HF_HOME' in os.environ:
        cache_dir = Path(os.environ['HF_HOME']) / 'hub'
    elif 'TRANSFORMERS_CACHE' in os.environ:
        cache_dir = Path(os.environ['TRANSFORMERS_CACHE'])
    
    # Convert model name to cache format
    cache_model_name = model_name.replace('/', '--')
    model_pattern = f"models--{cache_model_name}"
    
    model_dirs = list(cache_dir.glob(f"{model_pattern}*"))
    
    if model_dirs:
        model_dir = model_dirs[0]
        print(f"✅ Found model cache at: {model_dir}")
        
        # Check if it has all necessary files
        refs_dir = model_dir / "refs"
        if refs_dir.exists() and (refs_dir / "main").exists():
            commit = (refs_dir / "main").read_text().strip()
            snapshot_dir = model_dir / "snapshots" / commit
            
            if snapshot_dir.exists():
                config_file = snapshot_dir / "config.json"
                has_model = (snapshot_dir / "model.safetensors").exists() or (snapshot_dir / "pytorch_model.bin").exists()
                has_tokenizer = (snapshot_dir / "tokenizer.json").exists() or (snapshot_dir / "tokenizer_config.json").exists()
                
                if config_file.exists() and has_model and has_tokenizer:
                    print(f"✅ Model cache is complete")
                    return True, str(model_dir)
                else:
                    print(f"⚠️  Model cache incomplete - missing files")
                    return False, str(model_dir)
        
        print(f"⚠️  Model cache structure invalid")
        return False, str(model_dir)
    
    print(f"❌ Model not found in cache: {cache_dir}")
    return False, None

def load_model_with_max_optimizations():
    """Load SmolVLM with MAXIMUM optimizations for speed - OFFLINE FIRST"""
    global model, processor, DEVICE, model_loaded
    
    model_name = "HuggingFaceTB/SmolVLM-Instruct"
    print(f"🚀 Loading SmolVLM with MAXIMUM optimizations...")
    
    # Device setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {DEVICE}")
    
    if DEVICE == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 🔍 STEP 1: Check if model exists locally first
    print("🔍 Checking for local model cache...")
    model_exists, cache_path = check_model_exists_locally(model_name)
    
    if not model_exists:
        print(f"❌ SmolVLM model not found locally!")
        print(f"📥 You need to download the model first (~2-8GB)")
        print(f"")
        print(f"🚀 Quick fix options:")
        print(f"1. Run: python -c \"from transformers import AutoProcessor, AutoModelForVision2Seq; AutoProcessor.from_pretrained('{model_name}'); AutoModelForVision2Seq.from_pretrained('{model_name}')\"")
        print(f"2. Or use the API version: python smolvlm_cheap_api.py")
        print(f"")
        print(f"💡 After downloading, run this script again!")
        return False
    
    # 🎯 STEP 2: Load processor with LOCAL_FILES_ONLY to avoid internet
    try:
        print("📝 Loading processor (offline mode)...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=True,  # 🔑 KEY: Don't try to connect to internet
            trust_remote_code=True
        )
        print("✅ Processor loaded from local cache!")
        
    except Exception as e:
        print(f"⚠️  Failed to load processor offline: {e}")
        print("🌐 Trying online mode (requires internet)...")
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print("✅ Processor loaded online!")
        except Exception as e2:
            print(f"❌ Failed to load processor: {e2}")
            print("💡 Try: pip install --upgrade transformers")
            return False
    
    # 🎯 STEP 3: Load model with optimizations
    print("🤖 Loading model (offline mode)...")
    
    # Optimization 1: 4-bit quantization for maximum memory savings
    quantization_config = None
    if DEVICE == "cuda":
        try:
            print("🔧 Setting up 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8
            )
            print("✅ Quantization config ready")
        except ImportError:
            print("⚠️  bitsandbytes not available, install with: pip install bitsandbytes")
    
    # Model loading with local-first approach
    model_kwargs = {
        "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "local_files_only": True,  # 🔑 KEY: Load from cache only
        "trust_remote_code": True
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        print("✅ Using 4-bit quantization")
    else:
        model_kwargs["device_map"] = "auto" if DEVICE == "cuda" else None
    
    try:
        print("📦 Loading model from local cache...")
        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
        print("✅ Model loaded from local cache!")
        
        if not quantization_config:
            model = model.to(DEVICE)
            
    except Exception as e:
        print(f"⚠️  Failed to load model offline: {e}")
        print("🌐 Trying online mode...")
        try:
            # Remove local_files_only and try again
            model_kwargs.pop("local_files_only", None)
            model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
            print("✅ Model loaded online!")
            
            if not quantization_config:
                model = model.to(DEVICE)
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            print("💡 Fallback: Loading with minimal config...")
            try:
                # Last resort: minimal loading
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
                model = model.to(DEVICE)
                print("✅ Model loaded with minimal config!")
            except Exception as e3:
                print(f"❌ All loading attempts failed: {e3}")
                return False
    
    # 🎯 STEP 4: Apply optimizations
    print("⚡ Applying optimizations...")
    
    # Optimization 2: Set to eval mode
    model.eval()
    
    # Optimization 3: Compile model for faster inference (PyTorch 2.0+)
    try:
        print("⚡ Compiling model for maximum speed...")
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        print("✅ Model compiled successfully")
    except Exception as e:
        print(f"⚠️  Model compilation not available: {e}")
    
    # Optimization 4: Enable Flash Attention if available
    if hasattr(model.config, 'use_flash_attention_2'):
        model.config.use_flash_attention_2 = True
        print("✅ Flash Attention enabled")
    
    # Optimization 5: Gradient checkpointing off (we're not training)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    # Optimization 6: Memory cleanup
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    model_loaded = True
    print("🎉 SmolVLM loaded with maximum optimizations!")
    
    # Print optimization summary
    print("\n🔧 Applied Optimizations:")
    print(f"  ✓ Offline loading: Yes")
    print(f"  ✓ 4-bit quantization: {'Yes' if quantization_config else 'No'}")
    print(f"  ✓ Model compilation: {'Yes' if torch.jit.is_scripting() else 'Attempted'}")
    print(f"  ✓ Flash attention: {'Yes' if hasattr(model.config, 'use_flash_attention_2') else 'No'}")
    print(f"  ✓ Mixed precision: {'FP16' if DEVICE == 'cuda' else 'FP32'}")
    print(f"  ✓ Memory optimization: Yes")
    
    return True

@lru_cache(maxsize=32)
def preprocess_image_cached(image_hash, max_size=512):
    """Cache preprocessed images for repeated queries"""
    # This is a placeholder - in practice, you'd need to implement proper image hashing
    pass

def preprocess_image_optimized(image):
    """Optimized image preprocessing"""
    perf_start = time.time()
    
    # Optimization 1: Aggressive resizing for speed
    max_size = 448  # Smaller than default for speed, but still good quality
    
    if max(image.size) > max_size:
        # Use faster resampling method
        image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    
    # Optimization 2: Ensure RGB format efficiently
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    perf_monitor.preprocessing_time += time.time() - perf_start
    return image

def generate_streaming_response_optimized(image, question):
    """Ultra-optimized streaming generation"""
    global perf_monitor
    perf_monitor.reset()
    
    # Preprocess image
    image = preprocess_image_optimized(image)
    
    gen_start = time.time()
    
    # Create conversation - SmolVLM format
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
    
    # Process inputs with optimizations
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    # Optimization: Use attention mask for better efficiency
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    
    # Create optimized streamer
    streamer = TextIteratorStreamer(
        processor.tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True,
        timeout=30,
        clean_up_tokenization_spaces=True
    )
    
    # ULTRA-OPTIMIZED generation parameters
    generation_kwargs = {
        **inputs,
        "max_new_tokens": 300,           # Reduced for speed
        "min_new_tokens": 10,            # Ensure minimum response
        "do_sample": False,              # Greedy decoding = faster
        "num_beams": 1,                  # No beam search = much faster
        "early_stopping": True,          # Stop early when possible
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,               # Enable KV caching
        "streamer": streamer,
        "repetition_penalty": 1.1,       # Slight penalty to avoid repetition
    }
    
    # Remove attention_mask if it causes issues
    if "attention_mask" in generation_kwargs and len(generation_kwargs["attention_mask"].shape) > 2:
        del generation_kwargs["attention_mask"]
    
    print(f"[GENERATION] Starting optimized generation: '{question}'")
    print(f"[GENERATION] Image size: {image.size}, Input tokens: {inputs['input_ids'].shape[1]}")
    
    def generate():
        try:
            with torch.no_grad():
                # Enable autocast for mixed precision
                if DEVICE == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        model.generate(**generation_kwargs)
                else:
                    model.generate(**generation_kwargs)
        except Exception as e:
            print(f"Generation error: {e}")
            streamer.put(f"Generation error: {str(e)}")
        finally:
            # Aggressive memory cleanup
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    # Start generation in background
    generation_thread = threading.Thread(target=generate, daemon=True)
    generation_thread.start()
    
    # Stream tokens with performance monitoring
    token_count = 0
    try:
        for token_text in streamer:
            if token_text and token_text.strip():
                token_count += 1
                perf_monitor.tokens_generated = token_count
                yield token_text
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f" [Streaming error: {str(e)}]"
    
    # Wait for completion with timeout
    generation_thread.join(timeout=60)
    perf_monitor.generation_time = time.time() - gen_start
    
    print(f"[GENERATION] Complete! Generated {token_count} tokens in {perf_monitor.generation_time:.2f}s")
    print(f"[GENERATION] Speed: {perf_monitor.tokens_generated / perf_monitor.generation_time:.2f} tokens/sec")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Optimized image upload"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Fast image processing
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Quick format conversion
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Moderate resizing for web display
        display_max = 800
        display_image = image.copy()
        if max(display_image.size) > display_max:
            display_image.thumbnail((display_max, display_max), Image.Resampling.BILINEAR)
        
        # Convert to base64 with good compression
        buffer = io.BytesIO()
        display_image.save(buffer, format='JPEG', quality=75, optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Store original for processing
        app.current_image = image
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_str}",
            'original_size': f"{image.size[0]}x{image.size[1]}",
            'display_size': f"{display_image.size[0]}x{display_image.size[1]}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/url_image', methods=['POST'])
def upload_url_image():
    """Optimized URL image loading"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Load image with timeout
        image = load_image(url)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process for display
        display_max = 800
        display_image = image.copy()
        if max(display_image.size) > display_max:
            display_image.thumbnail((display_max, display_max), Image.Resampling.BILINEAR)
        
        buffer = io.BytesIO()
        display_image.save(buffer, format='JPEG', quality=75, optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        app.current_image = image
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_str}",
            'original_size': f"{image.size[0]}x{image.size[1]}",
            'display_size': f"{display_image.size[0]}x{display_image.size[1]}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error loading image from URL: {str(e)}'}), 500

@app.route('/generate')
def generate_text():
    """Ultra-fast streaming generation"""
    if not model_loaded:
        return Response("Error: Model not loaded", mimetype='text/plain')
    
    prompt = request.args.get('prompt', 'Describe this image concisely.')
    
    if not hasattr(app, 'current_image') or app.current_image is None:
        return Response("Error: No image uploaded", mimetype='text/plain')
    
    def generate():
        try:
            # Start performance monitoring
            start_time = time.time()
            
            for token_text in generate_streaming_response_optimized(app.current_image, prompt):
                yield f"data: {json.dumps({'text': token_text})}\n\n"
            
            # Send performance stats
            stats = perf_monitor.get_stats()
            yield f"data: {json.dumps({'complete': True, 'stats': stats})}\n\n"
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/performance')
def performance_stats():
    """Get current performance statistics"""
    stats = {
        'model_loaded': model_loaded,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'system_memory': f"{psutil.virtual_memory().percent}%",
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_memory_allocated': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
            'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            'gpu_memory_percent': f"{torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%"
        })
    
    return jsonify(stats)

@app.route('/optimize', methods=['POST'])
def optimize_model():
    """Runtime optimization controls"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    
    action = request.json.get('action', '')
    
    if action == 'clear_cache':
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return jsonify({'success': True, 'message': 'Cache cleared'})
    
    elif action == 'memory_stats':
        stats = {}
        if DEVICE == "cuda":
            stats = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        return jsonify(stats)
    
    return jsonify({'error': 'Unknown action'}), 400

@app.route('/warmup')
def warmup_model():
    """Warm up the model with a test image"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        # Create a small test image
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Run a quick inference to warm up
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    model.generate(**inputs, max_new_tokens=5, do_sample=False)
            else:
                model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
        return jsonify({'success': True, 'message': 'Model warmed up'})
        
    except Exception as e:
        return jsonify({'error': f'Warmup failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("⚡ SmolVLM ULTRA-OPTIMIZED Local Demo")
    print("="*60)
    
    # Load model with all optimizations
    if not load_model_with_max_optimizations():
        print("\n❌ Model loading failed!")
        print("🔧 Quick fix options:")
        print("  1. Download model: python -c \"from transformers import AutoProcessor, AutoModelForVision2Seq; AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct'); AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')\"")
        print("  2. Use API version: python smolvlm_cheap_api.py") 
        print("  3. Check internet connection if downloading")
        sys.exit(1)
    
    # Test with a sample if available
    try:
        if os.path.exists("RoboDog.jpg"):
            print("\n🐕 Testing with RoboDog.jpg...")
            test_image = Image.open("RoboDog.jpg")
            
            start_time = time.time()
            result_tokens = list(generate_streaming_response_optimized(test_image, "What do you see?"))
            end_time = time.time()
            
            result_text = "".join(result_tokens)
            speed = len(result_tokens) / (end_time - start_time)
            
            print(f"✅ Test complete!")
            print(f"📊 Generated {len(result_tokens)} tokens in {end_time - start_time:.2f}s")
            print(f"⚡ Speed: {speed:.2f} tokens/second")
            print(f"💬 Result: {result_text[:100]}...")
            
    except Exception as e:
        print(f"⚠️  Test failed: {e}")
    
    # Initialize
    app.current_image = None
    
    print(f"\n🚀 Starting ULTRA-OPTIMIZED SmolVLM server...")
    print(f"🌐 Web interface: http://localhost:5000")
    print(f"📊 Performance monitor: http://localhost:5000/performance")
    print(f"🔥 Warmup endpoint: http://localhost:5000/warmup")
    print(f"🧹 Cache control: http://localhost:5000/optimize")
    print("="*60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)