# ─────────────────────────────────────────────────────────────
#  NutriVision — Gradio Demo App
#  Three input modes: Image | Audio | Video walkthrough
#
#  Install:
#    pip install gradio "unsloth[colab-new]" pillow librosa
#    pip install opencv-python  # for video tab
#
#  Run:
#    python gradio_app.py
#    # or to share publicly:
#    python gradio_app.py --share
# ─────────────────────────────────────────────────────────────

import os
import json
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image as PILImage


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

ADAPTER_DIR   = "nutrivision_adapter"     # your fine-tuned model
BASE_MODEL    = "unsloth/gemma-4-E4B-it"  # fallback if adapter not found
MAX_SEQ_LEN   = 2048
LOAD_IN_4BIT  = True
MAX_NEW_TOKENS= 512

# Gemma 4 recommended inference settings
GEN_PARAMS = dict(
    max_new_tokens = MAX_NEW_TOKENS,
    use_cache      = True,
    temperature    = 1.0,
    top_p          = 0.95,
    top_k          = 64,
    do_sample      = True,
)


# ──────────────────────────────────────────────
#  MODEL LOADING (cached at startup)
# ──────────────────────────────────────────────

_model     = None
_processor = None

def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from unsloth import FastVisionModel  # noqa

    model_path = ADAPTER_DIR if Path(ADAPTER_DIR).exists() else BASE_MODEL
    print(f"Loading model from: {model_path}")

    _model, _processor = FastVisionModel.from_pretrained(
        model_name             = model_path,
        max_seq_length         = MAX_SEQ_LEN,
        load_in_4bit           = LOAD_IN_4BIT,
        use_gradient_checkpointing = "unsloth",
    )
    FastVisionModel.for_inference(_model)
    print("Model loaded.")
    return _model, _processor


# ──────────────────────────────────────────────
#  INFERENCE CORE
# ──────────────────────────────────────────────

def run_gemma4(content: list[dict]) -> str:
    """
    Sends a multimodal content list to Gemma 4 and returns the response string.
    content format: [{"type": "image"|"audio"|"text", ...}, ...]
    IMPORTANT: image/audio must come BEFORE text (Gemma 4 requirement).
    """
    model, processor = get_model()

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize              = True,
        return_dict           = True,
        return_tensors        = "pt",
    ).to("cuda")

    output_ids = model.generate(**inputs, **GEN_PARAMS)
    # Decode only newly generated tokens (skip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_len:]
    response   = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return response.strip()


def load_audio_array(audio_path: str) -> np.ndarray:
    """Resamples any audio file to 16kHz mono float32."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                ratio = 16000 / sr
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, int(len(audio) * ratio)),
                    np.arange(len(audio)),
                    audio
                )
            return audio.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Cannot load audio. Install librosa: pip install librosa\n{e}")


def format_nutrition_output(raw_text: str) -> tuple[str, str]:
    """
    Parses raw model output into two parts:
      1. Formatted markdown for display
      2. Raw JSON string for the JSON tab
    """
    # Try to extract JSON from the response
    json_data = None
    try:
        # Model may output raw JSON or JSON wrapped in markdown code blocks
        text = raw_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        json_data = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        # Fallback: try to find the first { ... } block
        import re
        matches = re.findall(r'\{.*\}', raw_text, re.DOTALL)
        for m in matches:
            try:
                json_data = json.loads(m)
                break
            except json.JSONDecodeError:
                continue

    if json_data is None:
        return raw_text, raw_text   # show raw if parse fails

    # ── Build markdown display ──────────────────
    food     = json_data.get("food", "Unknown")
    serving  = json_data.get("serving", "—")
    calories = json_data.get("calories", "—")
    macros   = json_data.get("macros", {})
    score    = json_data.get("health_score", "—")
    tip      = json_data.get("tip", "")

    stars = "⭐" * min(int(score), 10) if isinstance(score, (int, float)) else ""

    md = f"""## {food}

**Serving size:** {serving}
**Calories:** {calories} kcal

---

### Macronutrients
| Nutrient | Amount |
|----------|--------|
| Protein  | {macros.get("protein_g", "—")} g |
| Carbs    | {macros.get("carbs_g", "—")} g |
| Fat      | {macros.get("fat_g", "—")} g |
| Fiber    | {macros.get("fiber_g", "—")} g |

---

### Health score: {score}/10 {stars}

💡 **{tip}**
"""
    raw_json = json.dumps(json_data, indent=2)
    return md, raw_json


# ──────────────────────────────────────────────
#  TAB 1 — IMAGE
# ──────────────────────────────────────────────

IMAGE_PROMPTS = [
    "Analyze this food image and return a JSON object with: food name, serving size, calories, macros (protein_g, carbs_g, fat_g, fiber_g), health_score (1-10), and a brief tip.",
    "What is this food? Give me a complete nutritional breakdown as JSON.",
    "Identify this dish and estimate its calories, macros, and health score as JSON.",
]

def analyze_image(image, custom_prompt: str, prompt_choice: str) -> tuple[str, str]:
    if image is None:
        return "Please upload a food image.", ""

    prompt = custom_prompt.strip() if custom_prompt.strip() else (
        IMAGE_PROMPTS[["Default analysis", "Quick ID", "Detailed estimate"].index(prompt_choice)]
        if prompt_choice in ["Default analysis", "Quick ID", "Detailed estimate"] else IMAGE_PROMPTS[0]
    )

    # Convert to PIL if numpy (Gradio may pass numpy array)
    if isinstance(image, np.ndarray):
        pil_image = PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
    elif isinstance(image, PILImage.Image):
        pil_image = image.convert("RGB")
    else:
        return "Invalid image format.", ""

    content = [
        {"type": "image", "image": pil_image},  # image FIRST
        {"type": "text",  "text": prompt},
    ]

    try:
        raw = run_gemma4(content)
        return format_nutrition_output(raw)
    except Exception as e:
        return f"Error: {e}", ""


# ──────────────────────────────────────────────
#  TAB 2 — AUDIO
# ──────────────────────────────────────────────

AUDIO_PROMPT = (
    "This is a voice recording of a meal log. "
    "Extract all food items mentioned, estimate total calories and macros, "
    "compute a health score (1-10), and return a JSON object with: "
    "food, serving, calories, macros (protein_g, carbs_g, fat_g, fiber_g), health_score, tip."
)

def analyze_audio(audio_path, custom_prompt: str) -> tuple[str, str]:
    if audio_path is None:
        return "Please record or upload an audio clip.", ""

    prompt = custom_prompt.strip() if custom_prompt.strip() else AUDIO_PROMPT

    try:
        audio_array = load_audio_array(audio_path)

        # Enforce 30s max (Gemma 4 E4B audio limit)
        max_samples = 30 * 16000
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
            gr.Warning("Audio trimmed to 30 seconds (Gemma 4 E4B limit).")

        content = [
            {"type": "audio", "audio": audio_array},  # audio FIRST
            {"type": "text",  "text": prompt},
        ]
        raw = run_gemma4(content)
        return format_nutrition_output(raw)
    except Exception as e:
        return f"Error: {e}", ""


# ──────────────────────────────────────────────
#  TAB 3 — VIDEO (plate walkthrough)
# ──────────────────────────────────────────────

VIDEO_PROMPT = (
    "This is a short video walkthrough of a meal plate. "
    "Analyze each frame and provide a complete JSON nutrition analysis: "
    "food name, serving estimate, calories, macros (protein_g, carbs_g, fat_g, fiber_g), "
    "health_score (1-10), and a health tip."
)

MAX_FRAMES = 10   # keep under 60s at 1fps; more frames = more VRAM

def extract_frames(video_path: str, n_frames: int = MAX_FRAMES) -> list[PILImage.Image]:
    """Extracts evenly-spaced frames from a video file."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV not installed. Run: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps

    if duration > 60:
        gr.Warning(f"Video is {duration:.0f}s — only first 60s used (Gemma 4 limit).")
        total_frames = int(60 * fps)

    indices = np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PILImage.fromarray(rgb).resize((336, 336)))
    cap.release()
    return frames


def analyze_video(video_path, custom_prompt: str) -> tuple[str, str]:
    if video_path is None:
        return "Please upload a video.", ""

    prompt = custom_prompt.strip() if custom_prompt.strip() else VIDEO_PROMPT

    try:
        frames = extract_frames(video_path)
        if not frames:
            return "Could not extract frames from video.", ""

        # Video format: frames first, then text instruction
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})

        raw = run_gemma4(content)
        return format_nutrition_output(raw)
    except Exception as e:
        return f"Error: {e}", ""


# ──────────────────────────────────────────────
#  GRADIO UI
# ──────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue   = gr.themes.colors.emerald,
    secondary_hue = gr.themes.colors.sky,
    neutral_hue   = gr.themes.colors.slate,
    font          = gr.themes.GoogleFont("Inter"),
)

DESCRIPTION = """
### Snap · Speak · Record → Get your nutrition breakdown instantly

NutriVision uses a fine-tuned Gemma 4 E4B multimodal model to analyze food across three input types:
- 📸 **Image** — take or upload a photo of your meal
- 🎤 **Audio** — speak a quick voice log ("I had dal and rice for lunch")
- 🎥 **Video** — record a short walkthrough of your plate
"""

def build_ui():
    with gr.Blocks(theme=THEME, title="NutriVision") as demo:

        gr.Markdown("# 🥗 NutriVision")
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():

            # ── Tab 1: Image ──────────────────────────
            with gr.TabItem("📸 Image"):
                gr.Markdown("**Upload or snap a photo of your meal.** Supports JPEG, PNG, WebP.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label  = "Food image",
                            type   = "numpy",
                            height = 300,
                        )
                        img_prompt_radio = gr.Radio(
                            choices = ["Default analysis", "Quick ID", "Detailed estimate"],
                            value   = "Default analysis",
                            label   = "Analysis type",
                        )
                        img_custom_prompt = gr.Textbox(
                            label       = "Custom prompt (overrides analysis type if filled)",
                            placeholder = "e.g. How many calories is this for a diabetic diet?",
                            lines       = 2,
                        )
                        img_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=1):
                        img_markdown_out = gr.Markdown(label="Nutrition summary")
                        img_json_out     = gr.Code(language="json", label="Raw JSON")

                img_btn.click(
                    fn      = analyze_image,
                    inputs  = [img_input, img_custom_prompt, img_prompt_radio],
                    outputs = [img_markdown_out, img_json_out],
                )
                gr.Examples(
                    examples        = [["examples/pizza.jpg", "", "Default analysis"]],
                    inputs          = [img_input, img_custom_prompt, img_prompt_radio],
                    outputs         = [img_markdown_out, img_json_out],
                    fn              = analyze_image,
                    cache_examples  = False,
                    label           = "Example inputs",
                )

            # ── Tab 2: Audio ──────────────────────────
            with gr.TabItem("🎤 Audio"):
                gr.Markdown(
                    "**Record a voice log or upload an audio file** (MP3, WAV, M4A). "
                    "Max 30 seconds. Speak naturally: *'I had two chapatis and dal for lunch.'*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label  = "Voice meal log",
                            type   = "filepath",   # Gradio returns file path for .from_pretrained()
                            format = "wav",
                        )
                        audio_custom_prompt = gr.Textbox(
                            label       = "Custom prompt (optional)",
                            placeholder = "e.g. Log this for a 1500-calorie daily target.",
                            lines       = 2,
                        )
                        audio_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=1):
                        audio_markdown_out = gr.Markdown(label="Nutrition summary")
                        audio_json_out     = gr.Code(language="json", label="Raw JSON")

                audio_btn.click(
                    fn      = analyze_audio,
                    inputs  = [audio_input, audio_custom_prompt],
                    outputs = [audio_markdown_out, audio_json_out],
                )

            # ── Tab 3: Video ──────────────────────────
            with gr.TabItem("🎥 Video"):
                gr.Markdown(
                    "**Record or upload a short plate walkthrough** (MP4, MOV, AVI). "
                    "Max 60 seconds. Slowly move the camera over your entire plate."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label = "Plate walkthrough video",
                        )
                        video_custom_prompt = gr.Textbox(
                            label       = "Custom prompt (optional)",
                            placeholder = "e.g. I am vegetarian. Analyze all visible dishes.",
                            lines       = 2,
                        )
                        video_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=1):
                        video_markdown_out = gr.Markdown(label="Nutrition summary")
                        video_json_out     = gr.Code(language="json", label="Raw JSON")

                video_btn.click(
                    fn      = analyze_video,
                    inputs  = [video_input, video_custom_prompt],
                    outputs = [video_markdown_out, video_json_out],
                )

        # ── Footer ───────────────────────────────
        gr.Markdown(
            "---\n"
            "Built with [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) fine-tuned via "
            "[Unsloth](https://unsloth.ai) · "
            "Nutritional estimates are approximations — consult a dietitian for medical advice."
        )

    return demo


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriVision Gradio App")
    parser.add_argument("--share",     action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port",      type=int, default=7860, help="Port to serve on")
    parser.add_argument("--no-preload",action="store_true",  help="Skip model preload at startup")
    args = parser.parse_args()

    if not args.no_preload:
        print("Pre-loading model at startup (avoids cold-start on first request)...")
        try:
            get_model()
        except Exception as e:
            print(f"  Model preload failed: {e}")
            print("  Will load on first request instead.")

    demo = build_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = args.share,
        favicon_path= None,
    )
