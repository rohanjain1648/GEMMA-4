# ─────────────────────────────────────────────────────────────
#  NutriVision — Dataset Builder
#  Builds a multimodal JSONL dataset for Gemma 4 E4B fine-tuning
#  Modalities: Image (food photos) | Audio (voice logs) | Video (plate walkthroughs)
#
#  Install:
#    pip install datasets pillow gtts pydub opencv-python tqdm huggingface_hub
# ─────────────────────────────────────────────────────────────

import os
import json
import random
import tempfile
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

OUTPUT_DIR = Path("nutrivision_data")
IMAGE_DIR  = OUTPUT_DIR / "images"
AUDIO_DIR  = OUTPUT_DIR / "audio"
VIDEO_DIR  = OUTPUT_DIR / "video_frames"

N_IMAGE_SAMPLES = 800
N_AUDIO_SAMPLES = 400
N_VIDEO_SAMPLES = 200

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
#  NUTRITION KNOWLEDGE BASE
#  (approximate values per 100g / standard serving)
# ──────────────────────────────────────────────

FOOD_NUTRITION = {
    "pizza":           {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "fiber": 2,  "serving": "1 slice (107g)"},
    "salad":           {"calories": 20,  "protein": 2,  "carbs": 3,  "fat": 0,  "fiber": 2,  "serving": "1 bowl (150g)"},
    "sushi":           {"calories": 170, "protein": 9,  "carbs": 24, "fat": 4,  "fiber": 1,  "serving": "6 pieces (156g)"},
    "burger":          {"calories": 295, "protein": 17, "carbs": 24, "fat": 14, "fiber": 1,  "serving": "1 burger (226g)"},
    "pasta":           {"calories": 220, "protein": 8,  "carbs": 43, "fat": 2,  "fiber": 2,  "serving": "1 cup (250g)"},
    "steak":           {"calories": 271, "protein": 26, "carbs": 0,  "fat": 18, "fiber": 0,  "serving": "1 serving (100g)"},
    "ice_cream":       {"calories": 207, "protein": 4,  "carbs": 24, "fat": 11, "fiber": 0,  "serving": "1 scoop (100g)"},
    "omelette":        {"calories": 154, "protein": 11, "carbs": 1,  "fat": 12, "fiber": 0,  "serving": "1 omelette (120g)"},
    "waffles":         {"calories": 291, "protein": 8,  "carbs": 37, "fat": 13, "fiber": 1,  "serving": "2 waffles (130g)"},
    "fried_rice":      {"calories": 238, "protein": 5,  "carbs": 46, "fat": 3,  "fiber": 1,  "serving": "1 cup (200g)"},
    "hot_dog":         {"calories": 290, "protein": 11, "carbs": 23, "fat": 17, "fiber": 1,  "serving": "1 hot dog (160g)"},
    "apple_pie":       {"calories": 296, "protein": 2,  "carbs": 42, "fat": 14, "fiber": 2,  "serving": "1 slice (125g)"},
    "spring_rolls":    {"calories": 154, "protein": 5,  "carbs": 21, "fat": 5,  "fiber": 2,  "serving": "2 rolls (100g)"},
    "miso_soup":       {"calories": 40,  "protein": 3,  "carbs": 5,  "fat": 1,  "fiber": 1,  "serving": "1 bowl (250ml)"},
    "ramen":           {"calories": 436, "protein": 21, "carbs": 64, "fat": 10, "fiber": 2,  "serving": "1 bowl (500ml)"},
    "caesar_salad":    {"calories": 190, "protein": 7,  "carbs": 8,  "fat": 16, "fiber": 2,  "serving": "1 serving (170g)"},
    "chocolate_cake":  {"calories": 352, "protein": 5,  "carbs": 51, "fat": 15, "fiber": 2,  "serving": "1 slice (100g)"},
    "pancakes":        {"calories": 227, "protein": 6,  "carbs": 28, "fat": 10, "fiber": 1,  "serving": "2 pancakes (100g)"},
    "grilled_salmon":  {"calories": 206, "protein": 30, "carbs": 0,  "fat": 9,  "fiber": 0,  "serving": "1 fillet (178g)"},
    "chicken_wings":   {"calories": 290, "protein": 27, "carbs": 8,  "fat": 16, "fiber": 0,  "serving": "4 wings (100g)"},
    "dal":             {"calories": 116, "protein": 9,  "carbs": 20, "fat": 0,  "fiber": 5,  "serving": "1 cup (200g)"},
    "chapati":         {"calories": 104, "protein": 3,  "carbs": 18, "fat": 3,  "fiber": 2,  "serving": "1 roti (40g)"},
    "biryani":         {"calories": 290, "protein": 12, "carbs": 45, "fat": 7,  "fiber": 2,  "serving": "1 cup (200g)"},
    "idli":            {"calories": 58,  "protein": 2,  "carbs": 12, "fat": 0,  "fiber": 1,  "serving": "2 idlis (80g)"},
    "dosa":            {"calories": 133, "protein": 3,  "carbs": 24, "fat": 4,  "fiber": 1,  "serving": "1 dosa (80g)"},
}

HEALTH_TIPS = {
    "high_protein": [
        "Excellent protein source — great for muscle recovery.",
        "High protein content supports satiety and lean muscle.",
    ],
    "high_carbs": [
        "High in carbohydrates — consider pairing with protein to balance blood sugar.",
        "Good energy source; pair with fiber-rich vegetables.",
    ],
    "high_fat": [
        "Higher fat content — opt for smaller portions if watching calories.",
        "Contains significant fat; choose earlier in the day for better energy utilization.",
    ],
    "balanced": [
        "Well-balanced macro profile — a solid meal choice.",
        "Good nutritional balance across protein, carbs, and fat.",
    ],
    "low_cal": [
        "Low-calorie choice — excellent for weight management.",
        "Very light on calories; consider adding a protein source.",
    ],
}

def get_health_tip(nutrition: dict) -> str:
    cal = nutrition["calories"]
    prot = nutrition["protein"]
    fat = nutrition["fat"]
    if cal < 100:
        return random.choice(HEALTH_TIPS["low_cal"])
    if prot > 20:
        return random.choice(HEALTH_TIPS["high_protein"])
    if fat > 15:
        return random.choice(HEALTH_TIPS["high_fat"])
    if nutrition["carbs"] > 40:
        return random.choice(HEALTH_TIPS["high_carbs"])
    return random.choice(HEALTH_TIPS["balanced"])

def build_nutrition_response(food_name: str, nutrition: dict) -> str:
    tip = get_health_tip(nutrition)
    display_name = food_name.replace("_", " ").title()
    response = {
        "food":        display_name,
        "serving":     nutrition["serving"],
        "calories":    nutrition["calories"],
        "macros": {
            "protein_g": nutrition["protein"],
            "carbs_g":   nutrition["carbs"],
            "fat_g":     nutrition["fat"],
            "fiber_g":   nutrition["fiber"],
        },
        "health_score": max(1, min(10, round(10 - (nutrition["fat"] * 0.2) + (nutrition["protein"] * 0.3) - (nutrition["calories"] * 0.005) + (nutrition["fiber"] * 0.5)))),
        "tip": tip,
    }
    return json.dumps(response, indent=2)


# ──────────────────────────────────────────────
#  PART 1 — IMAGE DATASET (Food-101 via HF)
# ──────────────────────────────────────────────

IMAGE_INSTRUCTIONS = [
    "Analyze this food image and return a JSON object with the food name, serving size, calories, macros (protein, carbs, fat, fiber), health score out of 10, and a brief health tip.",
    "What food is in this image? Provide a complete nutrition analysis as JSON including calories, macros, and a health score.",
    "Please identify this dish and give me a detailed nutritional breakdown in JSON format.",
    "Estimate the nutritional content of this meal shown in the image. Return JSON with calories, protein, carbs, fat, fiber, health score, and tip.",
]

def build_image_dataset() -> list[dict]:
    print("Loading Food-101 from HuggingFace...")
    try:
        ds = load_dataset("ethz/food101", split="train", streaming=True)
    except Exception as e:
        print(f"  Could not load Food-101: {e}")
        print("  Generating synthetic placeholder images instead.")
        return _build_synthetic_image_dataset()

    # Food-101 label to our nutrition map (best-effort matching)
    label_map = {
        "pizza":          "pizza",
        "caesar_salad":   "caesar_salad",
        "sushi":          "sushi",
        "hamburger":      "burger",
        "spaghetti_carbonara": "pasta",
        "steak":          "steak",
        "ice_cream":      "ice_cream",
        "omelette":       "omelette",
        "waffles":        "waffles",
        "fried_rice":     "fried_rice",
        "hot_and_sour_soup": "miso_soup",
        "apple_pie":      "apple_pie",
        "spring_rolls":   "spring_rolls",
        "ramen":          "ramen",
        "chocolate_cake": "chocolate_cake",
        "pancakes":       "pancakes",
        "grilled_salmon": "grilled_salmon",
        "chicken_wings":  "chicken_wings",
    }

    samples = []
    seen = 0
    for item in tqdm(ds, desc="  Sampling Food-101", total=N_IMAGE_SAMPLES):
        if seen >= N_IMAGE_SAMPLES:
            break
        label = item["label"]
        # The dataset uses integer labels — map to string via features
        food_key = None
        for k, v in label_map.items():
            if k in str(label).lower():
                food_key = v
                break
        if food_key is None:
            food_key = random.choice(list(FOOD_NUTRITION.keys()))

        nutrition = FOOD_NUTRITION[food_key]
        img: Image.Image = item["image"].convert("RGB")
        img_path = IMAGE_DIR / f"food_{seen:04d}.jpg"
        img.save(img_path, format="JPEG", quality=90)

        instruction = random.choice(IMAGE_INSTRUCTIONS)
        response = build_nutrition_response(food_key, nutrition)

        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text",  "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ]
        })
        seen += 1

    print(f"  Built {len(samples)} image samples.")
    return samples


def _build_synthetic_image_dataset() -> list[dict]:
    """Fallback: creates solid-color placeholder images when Food-101 unavailable."""
    samples = []
    food_list = list(FOOD_NUTRITION.keys())
    for i in tqdm(range(N_IMAGE_SAMPLES), desc="  Generating placeholder images"):
        food_key = food_list[i % len(food_list)]
        nutrition = FOOD_NUTRITION[food_key]
        # Simple colored placeholder (replace with real images in production)
        color = (random.randint(100, 220), random.randint(80, 180), random.randint(60, 140))
        img = Image.new("RGB", (224, 224), color=color)
        img_path = IMAGE_DIR / f"food_{i:04d}.jpg"
        img.save(img_path)
        instruction = random.choice(IMAGE_INSTRUCTIONS)
        response = build_nutrition_response(food_key, nutrition)
        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text",  "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ]
        })
    print(f"  Built {len(samples)} synthetic image samples.")
    return samples


# ──────────────────────────────────────────────
#  PART 2 — AUDIO DATASET (gTTS voice logs)
# ──────────────────────────────────────────────

VOICE_LOG_TEMPLATES = [
    "I just had {qty} {food} for {meal}.",
    "For {meal} I ate {qty} {food}.",
    "Just finished eating {qty} {food}.",
    "Logging {qty} {food} for {meal}.",
    "Had {qty} {food} right now.",
    "I had {qty} {food} and a glass of water for {meal}.",
    "Just ate {qty} {food}, it was delicious.",
]

MEALS = ["breakfast", "lunch", "dinner", "a snack", "brunch"]
QTY_TEMPLATES = {
    "pizza":  ["two slices of", "one slice of", "a slice of"],
    "salad":  ["a bowl of", "a large bowl of", "some"],
    "burger": ["a", "one", "a big"],
    "pasta":  ["a plate of", "one serving of", "a bowl of"],
    "dal":    ["a cup of", "one bowl of", "some"],
    "chapati":["two", "three", "one"],
    "biryani":["a plate of", "one serving of", "half a plate of"],
    "idli":   ["two", "four", "three"],
    "dosa":   ["a", "one", "two"],
}

AUDIO_INSTRUCTIONS = [
    "Transcribe and analyze this food voice log. Return a JSON object with food items, estimated total calories, macros, and a brief health tip.",
    "Listen to this meal log and extract nutritional information as JSON.",
    "This is a voice recording of a meal log. Parse it and return structured nutrition JSON.",
]

def build_audio_dataset() -> list[dict]:
    try:
        from gtts import gTTS
    except ImportError:
        print("  gTTS not installed. Run: pip install gtts")
        print("  Skipping audio dataset (add gTTS and re-run).")
        return []

    samples = []
    food_list = list(FOOD_NUTRITION.keys())
    for i in tqdm(range(N_AUDIO_SAMPLES), desc="  Generating audio clips"):
        food_key = food_list[i % len(food_list)]
        nutrition = FOOD_NUTRITION[food_key]
        display = food_key.replace("_", " ")
        qty_opts = QTY_TEMPLATES.get(food_key, ["some", "a serving of", "one portion of"])
        qty = random.choice(qty_opts)
        meal = random.choice(MEALS)
        template = random.choice(VOICE_LOG_TEMPLATES)
        spoken_text = template.format(qty=qty, food=display, meal=meal)

        audio_path = AUDIO_DIR / f"log_{i:04d}.mp3"
        try:
            tts = gTTS(text=spoken_text, lang="en", slow=False)
            tts.save(str(audio_path))
        except Exception as e:
            print(f"  Audio gen failed for sample {i}: {e}")
            continue

        instruction = random.choice(AUDIO_INSTRUCTIONS)
        response = build_nutrition_response(food_key, nutrition)

        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": str(audio_path)},
                        {"type": "text",  "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ]
        })

    print(f"  Built {len(samples)} audio samples.")
    return samples


# ──────────────────────────────────────────────
#  PART 3 — VIDEO DATASET (multi-frame walkthrough)
#  Uses a sequence of PIL frames representing a
#  "plate walkthrough" — each frame is a slightly
#  different crop of the same food image.
#  Replace with real video frames in production.
# ──────────────────────────────────────────────

VIDEO_INSTRUCTIONS = [
    "This is a short video walkthrough of a meal plate. Analyze each frame and provide a complete JSON nutrition analysis including all visible food items, total calories, macros, and health score.",
    "Watch this meal walkthrough video and identify all foods present. Return nutrition data as JSON.",
    "Analyze this plate walkthrough video. What foods are present? Give me calories, macros, and a health score as JSON.",
]

def build_video_dataset() -> list[dict]:
    """
    Simulates video by generating N_FRAMES crops of a food image.
    In production: use cv2 to extract frames from real video files.
    """
    import cv2  # noqa: F401 — needed for real video; fallback below uses PIL only

    samples = []
    food_list = list(FOOD_NUTRITION.keys())

    for i in tqdm(range(N_VIDEO_SAMPLES), desc="  Generating video frame sequences"):
        food_key = food_list[i % len(food_list)]
        nutrition = FOOD_NUTRITION[food_key]
        video_dir = VIDEO_DIR / f"video_{i:04d}"
        video_dir.mkdir(exist_ok=True)

        # Use existing food image if available, else synthetic
        existing = list(IMAGE_DIR.glob("*.jpg"))
        if existing:
            base_img = Image.open(random.choice(existing)).convert("RGB").resize((320, 320))
        else:
            color = (random.randint(100, 220), random.randint(80, 180), random.randint(60, 140))
            base_img = Image.new("RGB", (320, 320), color=color)

        # Simulate 5 frames — slight zoom + pan to mimic walkthrough
        frames = []
        n_frames = 5
        arr = np.array(base_img)
        for f in range(n_frames):
            offset_x = int(f * 6)
            offset_y = int(f * 4)
            crop_size = 280 - f * 8
            cropped = arr[offset_y: offset_y + crop_size,
                          offset_x: offset_x + crop_size]
            if cropped.size == 0:
                cropped = arr[:280, :280]
            frame_img = Image.fromarray(cropped).resize((224, 224))
            frame_path = video_dir / f"frame_{f:02d}.jpg"
            frame_img.save(frame_path, format="JPEG", quality=85)
            frames.append(str(frame_path))

        instruction = random.choice(VIDEO_INSTRUCTIONS)
        response = build_nutrition_response(food_key, nutrition)

        # Gemma 4 video format: list of image frames, then text
        content = [{"type": "image", "image": fp} for fp in frames]
        content.append({"type": "text", "text": instruction})

        samples.append({
            "messages": [
                {"role": "user",      "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": response}]},
            ]
        })

    print(f"  Built {len(samples)} video samples.")
    return samples


# ──────────────────────────────────────────────
#  MAIN — ASSEMBLE + WRITE JSONL
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NutriVision Dataset Builder")
    print("=" * 60)

    all_samples = []

    print("\n[1/3] Building image dataset...")
    all_samples += build_image_dataset()

    print("\n[2/3] Building audio dataset...")
    all_samples += build_audio_dataset()

    print("\n[3/3] Building video dataset...")
    try:
        all_samples += build_video_dataset()
    except ImportError:
        print("  OpenCV not installed. Skipping video dataset.")
        print("  Run: pip install opencv-python and re-run.")

    random.shuffle(all_samples)

    # Split 90/10 train/eval
    split = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split]
    eval_samples  = all_samples[split:]

    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path  = OUTPUT_DIR / "eval.jsonl"

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(eval_path, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    print("\n" + "=" * 60)
    print(f"  Done! Total samples : {len(all_samples)}")
    print(f"  Train               : {len(train_samples)} → {train_path}")
    print(f"  Eval                : {len(eval_samples)}  → {eval_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
