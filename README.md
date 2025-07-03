# SmolVLM Vision Demo

<p align="center">
  <img src="assets/Bürgerfest.png" alt="Demo Example" width="500"/>
</p>

<p align="center"><em>Example output from SmolVLM Vision Demo</em></p>

This repository provides a web demo for the SmolVLM vision-language model, allowing you to upload or link to an image and receive real-time AI-generated descriptions and answers to your questions about the image.

## Features
- Upload images or provide image URLs
- Ask questions or request detailed descriptions about the image
- Real-time streaming responses
- Performance monitoring endpoints

## Prerequisites
- Python 3.10+
- pip (Python package manager)
- (Recommended) A CUDA-capable GPU for best performance

## Installation
1. **Clone the repository:**
   ```bash
   git clone <this-repo-url>
   cd VLsmoll
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Download the SmolVLM model in advance:**
   This is recommended for offline/fast startup. Run:
   ```bash
   python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct'); AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')"
   ```

## Usage
1. **Start the web app:**
   ```bash
   python web_app/app.py
   ```
2. **Open your browser and go to:**
   [http://localhost:5000](http://localhost:5000)

3. **Upload an image or paste an image URL, enter a prompt, and click "Generate Response".**

## Performance & Monitoring
- Visit [http://localhost:5000/performance](http://localhost:5000/performance) for live stats.
- Use `/warmup` and `/optimize` endpoints for advanced control (see `app.py` for details).

## Notes
- The first run may take a while as the model loads into memory.
- For best results, use a machine with a modern GPU and sufficient RAM.
- If you encounter issues with model loading, check the console output for troubleshooting tips.

## License
This project is for research and demo purposes. See individual model and dependency licenses for details. 