# ─────────────────────────────────────────────────────────────
#  NutriVision — Fine-tuning Script
#  Fine-tunes Gemma 4 E4B-it via Unsloth for multimodal
#  nutrition analysis (image + audio + video)
#
#  Hardware requirement: ~12GB VRAM (free Colab T4 works)
#
#  Install:
#    pip install "unsloth[colab-new]" trl transformers accelerate
#    pip install datasets pillow
# ─────────────────────────────────────────────────────────────
import unsloth  # MUST be first — before trl, transformers, peft
import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerState, TrainerControl


# ──────────────────────────────────────────────
#  CONFIG  (edit these before running)
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Model ──────────────────────────────────
    model_name: str        = "unsloth/gemma-4-E4B-it"   # or google/gemma-4-E4B-it
    output_dir: str        = "nutrivision_checkpoints"
    adapter_save_dir: str  = "nutrivision_adapter"
    hf_repo_id: str        = ""                         # e.g. "your-hf-user/nutrivision-gemma4"
    push_to_hub: bool      = False

    # ── Data ───────────────────────────────────
    train_jsonl: str       = "nutrivision_data/train.jsonl"
    eval_jsonl: str        = "nutrivision_data/eval.jsonl"

    # ── LoRA ───────────────────────────────────
    lora_rank: int         = 16
    lora_alpha: int        = 16
    lora_dropout: float    = 0.0
    finetune_vision: bool  = True    # set False to save VRAM (text-only fine-tune)
    finetune_language: bool= True
    finetune_attention: bool= True
    finetune_mlp: bool     = True

    # ── Training ───────────────────────────────
    max_seq_length: int    = 2048
    load_in_4bit: bool     = True     # QLoRA — set False for 16-bit LoRA (needs more VRAM)
    per_device_batch: int  = 1
    grad_accum: int        = 4        # effective batch = 4
    warmup_steps: int      = 5
    max_steps: int         = 100      # increase to 300-500 for production
    learning_rate: float   = 2e-4
    logging_steps: int     = 5
    save_steps: int        = 50
    eval_steps: int        = 50
    seed: int              = 42


CFG = TrainConfig()


# ──────────────────────────────────────────────
#  LOAD MODEL + PROCESSOR via Unsloth
# ──────────────────────────────────────────────

def load_model_and_processor():
    print(f"Loading {CFG.model_name} via Unsloth FastVisionModel...")
    from unsloth import FastVisionModel  # noqa

    model, processor = FastVisionModel.from_pretrained(
        model_name              = CFG.model_name,
        max_seq_length          = CFG.max_seq_length,
        load_in_4bit            = CFG.load_in_4bit,
        use_gradient_checkpointing = "unsloth",   # saves VRAM on long contexts
    )
    return model, processor


def attach_lora_adapters(model):
    from unsloth import FastVisionModel  # noqa

    print("Attaching LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers    = CFG.finetune_vision,
        finetune_language_layers  = CFG.finetune_language,
        finetune_attention_modules= CFG.finetune_attention,
        finetune_mlp_modules      = CFG.finetune_mlp,
        r                         = CFG.lora_rank,
        lora_alpha                = CFG.lora_alpha,
        lora_dropout              = CFG.lora_dropout,
        bias                      = "none",
        target_modules            = "all-linear",
        use_rslora                = True,   # rank-stabilised LoRA — more stable training
        random_state              = CFG.seed,
    )
    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


# ──────────────────────────────────────────────
#  DATASET LOADING
#  Reads the JSONL produced by dataset_builder.py
#  and resolves image/audio paths to actual objects
# ──────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def resolve_content(content_list: list[dict]) -> list[dict]:
    """
    Converts file-path strings in the dataset to PIL Images / numpy arrays
    so the Unsloth collator can process them.
    """
    from PIL import Image as PILImage

    resolved = []
    for item in content_list:
        if item["type"] == "image":
            path = item["image"]
            if isinstance(path, str) and Path(path).exists():
                img = PILImage.open(path).convert("RGB")
                resolved.append({"type": "image", "image": img})
            else:
                resolved.append(item)
        elif item["type"] == "audio":
            path = item["audio"]
            if isinstance(path, str) and Path(path).exists():
                # Load as numpy float32 array at 16kHz (expected by Gemma 4 audio)
                audio_array = _load_audio_as_array(path)
                resolved.append({"type": "audio", "audio": audio_array})
            else:
                resolved.append(item)
        else:
            resolved.append(item)
    return resolved


def _load_audio_as_array(path: str):
    """Loads an audio file and resamples to 16kHz mono numpy float32."""
    try:
        import librosa  # pip install librosa
        audio, _ = librosa.load(path, sr=16000, mono=True)
        return audio
    except ImportError:
        # Fallback: use soundfile if librosa not available
        try:
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Simple linear resample to 16kHz
            if sr != 16000:
                ratio = 16000 / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)
            return audio.astype(np.float32)
        except ImportError:
            print("  WARNING: neither librosa nor soundfile found.")
            print("  Install: pip install librosa")
            import numpy as np
            return np.zeros(16000, dtype=np.float32)  # 1s silence placeholder


def prepare_dataset(jsonl_path: str) -> Dataset:
    """Loads JSONL and resolves all media file paths to objects."""
    raw = load_jsonl(jsonl_path)
    resolved = []
    for sample in raw:
        new_messages = []
        for msg in sample["messages"]:
            new_content = resolve_content(msg["content"])
            new_messages.append({"role": msg["role"], "content": new_content})
        resolved.append({"messages": new_messages})
    return Dataset.from_list(resolved)


# ──────────────────────────────────────────────
#  FORMATTING FUNCTION
#  Applies the Gemma 4 chat template to each sample.
#  IMPORTANT: image/audio MUST come before text in
#  the user turn (Gemma 4 requirement).
# ──────────────────────────────────────────────

def make_formatting_func(processor):
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = processor.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = False,
            )
            texts.append(text)
        return {"text": texts}
    return formatting_func


# ──────────────────────────────────────────────
#  CALLBACKS
# ──────────────────────────────────────────────

class LossLogger(TrainerCallback):
    """Logs training loss and warns if it looks unusually high."""
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            step = state.global_step
            # For Gemma 4 multimodal: loss of 13-15 at start is NORMAL
            # If > 50 consistently, gradient accumulation may be misconfigured
            if step <= 10 and loss > 50:
                print(f"\n  WARNING: loss={loss:.2f} at step {step} is unusually high.")
                print("  This may indicate a gradient accumulation bug.")
                print("  Ref: https://unsloth.ai/blog/gradient\n")
            elif step > 20 and loss > 30:
                print(f"\n  WARNING: loss={loss:.2f} at step {step} — not converging as expected.")


# ──────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NutriVision Fine-tuning — Gemma 4 E4B via Unsloth")
    print("=" * 60)

    # ── GPU check ──────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Use Colab (T4 or better) or a local NVIDIA GPU.")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ─────────────────────────────
    model, processor = load_model_and_processor()
    model = attach_lora_adapters(model)

    # ── Load datasets ──────────────────────────
    print(f"\nLoading datasets...")
    if not Path(CFG.train_jsonl).exists():
        raise FileNotFoundError(
            f"Train JSONL not found at {CFG.train_jsonl}.\n"
            "Run dataset_builder.py first."
        )

    train_dataset = prepare_dataset(CFG.train_jsonl)
    eval_dataset  = prepare_dataset(CFG.eval_jsonl) if Path(CFG.eval_jsonl).exists() else None
    print(f"  Train: {len(train_dataset)} samples")
    if eval_dataset:
        print(f"  Eval:  {len(eval_dataset)} samples")

    # ── Data collator ──────────────────────────
    # UnslothVisionDataCollator handles the multimodal tensor preparation
    from unsloth.trainer import UnslothVisionDataCollator  # noqa
    data_collator = UnslothVisionDataCollator(model, processor)

    # ── Precision flags ────────────────────────
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # ── Training arguments ─────────────────────
    training_args = SFTConfig(
        output_dir                  = CFG.output_dir,
        per_device_train_batch_size = CFG.per_device_batch,
        gradient_accumulation_steps = CFG.grad_accum,
        warmup_steps                = CFG.warmup_steps,
        max_steps                   = CFG.max_steps,
        learning_rate               = CFG.learning_rate,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        logging_steps               = CFG.logging_steps,
        save_steps                  = CFG.save_steps,
        eval_steps                  = CFG.eval_steps if eval_dataset else None,
        eval_strategy               = "steps" if eval_dataset else "no",
        seed                        = CFG.seed,
        report_to                   = "none",          # set "wandb" if you want W&B logging
        dataset_text_field          = "text",
        dataset_kwargs              = {"skip_prepare_dataset": True},
        remove_unused_columns       = False,           # REQUIRED for multimodal
    )

    # ── Trainer ────────────────────────────────
    trainer = SFTTrainer(
        model          = model,
        tokenizer      = processor,
        train_dataset  = train_dataset,
        eval_dataset   = eval_dataset,
        data_collator  = data_collator,
        formatting_func= make_formatting_func(processor),
        args           = training_args,
        max_seq_length = CFG.max_seq_length,
        callbacks      = [LossLogger()],
    )

    # ── Show GPU memory before training ────────
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    max_memory = round(gpu_stats.total_memory / 1024 ** 3, 3)
    print(f"\n  GPU memory reserved: {start_gpu_memory} GB / {max_memory} GB")
    print(f"  Starting training for {CFG.max_steps} steps...\n")

    # ── Train ──────────────────────────────────
    trainer_stats = trainer.train()

    # ── Memory summary ─────────────────────────
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    used_pct    = round(100 * used_memory / max_memory, 3)
    print(f"\n  Peak GPU memory: {used_memory} GB ({used_pct}% of {max_memory} GB)")
    print(f"  Training time  : {trainer_stats.metrics['train_runtime']:.1f}s")

    # ── Save LoRA adapter ──────────────────────
    print(f"\nSaving LoRA adapter to {CFG.adapter_save_dir}...")
    model.save_pretrained(CFG.adapter_save_dir)
    processor.save_pretrained(CFG.adapter_save_dir)
    print("  Adapter saved.")

    # ── Push to HuggingFace Hub (optional) ─────
    if CFG.push_to_hub and CFG.hf_repo_id:
        print(f"\nPushing to HuggingFace Hub: {CFG.hf_repo_id}...")
        model.push_to_hub(CFG.hf_repo_id)
        processor.push_to_hub(CFG.hf_repo_id)
        print("  Pushed successfully.")

    # ── Export to GGUF (for llama.cpp / Ollama) ─
    print("\nExporting to GGUF (Q4_K_M quantization)...")
    try:
        model.save_pretrained_gguf(
            "nutrivision_gguf",
            processor,
            quantization_method="q4_k_m",
        )
        print("  GGUF saved to nutrivision_gguf/")
    except Exception as e:
        print(f"  GGUF export skipped: {e}")

    print("\n" + "=" * 60)
    print("  Fine-tuning complete!")
    print(f"  LoRA adapter → {CFG.adapter_save_dir}/")
    print("  Load with: FastVisionModel.from_pretrained(adapter_save_dir)")
    print("=" * 60)


# ──────────────────────────────────────────────
#  INFERENCE HELPER (for quick testing post-train)
# ──────────────────────────────────────────────

def run_inference(
    image_path: str | None       = None,
    audio_path: str | None       = None,
    adapter_dir: str             = "nutrivision_adapter",
    prompt: str                  = "Analyze this food and return nutrition JSON.",
    max_new_tokens: int          = 512,
):
    """Quick inference test after training."""
    from unsloth import FastVisionModel  # noqa
    from PIL import Image as PILImage

    print(f"Loading fine-tuned model from {adapter_dir}...")
    model, processor = FastVisionModel.from_pretrained(
        model_name   = adapter_dir,
        max_seq_length = CFG.max_seq_length,
        load_in_4bit = CFG.load_in_4bit,
    )
    FastVisionModel.for_inference(model)

    content = []
    if image_path:
        img = PILImage.open(image_path).convert("RGB")
        content.append({"type": "image", "image": img})
    if audio_path:
        from dataset_builder import _load_audio_as_array
        audio = _load_audio_as_array(audio_path)
        content.append({"type": "audio", "audio": audio})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize              = True,
        return_dict           = True,
        return_tensors        = "pt",
    ).to("cuda")

    from transformers import TextStreamer
    streamer = TextStreamer(processor, skip_prompt=True)

    print("\n── Model output ──────────────────────────")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        use_cache      = True,
        temperature    = 1.0,    # Gemma 4 recommended defaults
        top_p          = 0.95,
        top_k          = 64,
        streamer       = streamer,
    )
    print("\n──────────────────────────────────────────")


if __name__ == "__main__":
    main()
