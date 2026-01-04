"""
Vision Caption App using Google AI Studio (Gemini) and Gradio.

Run with:
    GOOGLE_API_KEY="your-key" python app.py
"""
import os
from typing import Optional

import gradio as gr
import google.generativeai as gen


def _build_model() -> gen.GenerativeModel:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is required to call Google AI Studio."
        )

    gen.configure(api_key=api_key)
    return gen.GenerativeModel("gemini-1.5-flash")


model: Optional[gen.GenerativeModel] = None


def caption_image(image) -> str:
    """Generate a concise caption for the uploaded image.

    The image is passed directly to Gemini along with an instruction prompt.
    """
    global model

    if image is None:
        return "Please upload an image to generate a caption."

    if model is None:
        model = _build_model()

    prompt = (
        "You are a precise photo captioning assistant."
        " Summarize the key objects, actions, and context in under 50 words."
    )
    response = model.generate_content([prompt, image])
    return response.text


def _demo() -> gr.Interface:
    return gr.Interface(
        fn=caption_image,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=gr.Textbox(label="Generated Caption", lines=4),
        title="Vision Caption App (Gemini)",
        description=(
            "Upload a photo and Gemini will describe what it sees using the"
            " multimodal vision model from Google AI Studio."
        ),
        allow_flagging="never",
        examples=[
            ["https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/COCO/000000039769.png"],
            ["https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/COCO/000000436483.png"],
        ],
    )


def main() -> None:
    demo = _demo()
    demo.launch()


if __name__ == "__main__":
    main()
