# Vision Caption App (Google AI Studio + Gradio)

A minimal VS Code–ready sample that captions images with the Gemini multimodal model from Google AI Studio.

## Features
- Uses `google-generativeai` to call the `gemini-1.5-flash` model for vision captioning.
- Gradio front end for drag-and-drop image uploads and instant captions.
- `.vscode` launch configuration so you can run and debug directly from VS Code.

## Prerequisites
- Python 3.10+
- A Google AI Studio API key stored in the `GOOGLE_API_KEY` environment variable.

## Setup
```bash
cd examples/vision_caption_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
export GOOGLE_API_KEY="<your-api-key>"
python app.py
```
Then open the Gradio URL printed in the terminal.

## VS Code debugging
1. Open the `examples/vision_caption_app` folder in VS Code.
2. Copy your API key into a `.env` file (ignored by Git) in this folder:
   ```env
   GOOGLE_API_KEY=your-key
   ```
3. Press `F5` or select **Run and Debug** → **Vision Caption App**. The launch config loads `.env` automatically.

## How it works
`app.py` lazily builds a Gemini client when the first image is uploaded, sends the image plus a concise captioning prompt, and streams the response text back to the UI.
