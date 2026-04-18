# ─────────────────────────────────────────────────────────────
#  NutriVision — Google Colab Setup Notebook (paste into Colab)
#  One-shot setup: install → build data → train → launch demo
# ─────────────────────────────────────────────────────────────
#
#  Paste each cell below into a new Colab notebook.
#  Runtime: GPU → T4 (free tier works for E4B in 4-bit)

# ── CELL 1: Install dependencies ──────────────────────────────
"""
!pip install "unsloth[colab-new]" -q
!pip install trl transformers accelerate datasets -q
!pip install gradio pillow librosa gtts opencv-python tqdm -q
"""

# ── CELL 2: Clone / upload your scripts ───────────────────────
"""
# Option A — from GitHub (after you push your repo):
# !git clone https://github.com/YOUR_USERNAME/nutrivision.git
# %cd nutrivision

# Option B — upload files manually to Colab:
# Upload dataset_builder.py, train_nutrivision.py, gradio_app.py
# via the Files panel on the left, then:
# import os; os.chdir('/content')
"""

# ── CELL 3: Build dataset ──────────────────────────────────────
"""
!python dataset_builder.py
"""

# ── CELL 4: Fine-tune ──────────────────────────────────────────
"""
!python train_nutrivision.py
"""

# ── CELL 5: Launch Gradio demo (public share link) ────────────
"""
!python gradio_app.py --share
"""

# ── CELL 6: Quick inline inference test ───────────────────────
"""
from train_nutrivision import run_inference
run_inference(
    image_path  = "nutrivision_data/images/food_0000.jpg",
    adapter_dir = "nutrivision_adapter",
    prompt      = "Analyze this food and return calories, macros, and health score as JSON.",
)
"""

# ── CELL 7: Push adapter to HuggingFace Hub ───────────────────
"""
from huggingface_hub import login
login()   # paste your HF token

from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained(
    "nutrivision_adapter", load_in_4bit=True
)
model.push_to_hub("your-hf-username/nutrivision-gemma4")
processor.push_to_hub("your-hf-username/nutrivision-gemma4")
"""
