"""
phase3_vision_model/hf_skills/eval_entry.py

Entry script executed *inside* an HF Skills job container for Phase 3 evaluation.

Responsibilities:
    1. Download the multimodal dataset revision from the Hub.
    2. Download the trained adapter (decoder_lora + projector.pt) from the Hub.
    3. Build the base CodeVisionModel and run baseline eval.
    4. Reload as fine-tuned and run post-FT eval against the same split.
    5. Push baseline.json + finetuned.json + summary.json to the adapter repo
       under eval/ so the comparison lives next to the model.

Expected env vars inside the job:
    HF_TOKEN            — read access to dataset, write access to adapter_repo
    PHASE3_PARAMS_JSON  — serialized A100VisionConfig (set by launcher)
    PHASE3_ADAPTER_REPO — e.g. cmndcntrlcyber/code-trainer-vision-adapter
    EVAL_NUM_SAMPLES    — int, defaults to 200
    EVAL_SPLIT          — "test" | "validation", defaults to "test"
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase3_vision_model.architecture.vision_model import CodeVisionModel
from src.phase3_vision_model.evaluation.evaluator import VisionModelEvaluator
from src.phase3_vision_model.training.collator import ScreenshotCodeCollator
from src.phase3_vision_model.training.dataset import ScreenshotCodeDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")


def _download_dataset_to_disk(dataset_id, revision, local_dir, limit=None):
    from datasets import load_dataset

    logger.info(f"Downloading dataset {dataset_id}@{revision} → {local_dir}")
    ds = load_dataset(dataset_id, revision=revision)
    if limit is not None:
        logger.info(f"Slicing each split to first {limit} rows")
        ds = ds.__class__({
            split: ds[split].select(range(min(limit, len(ds[split]))))
            for split in ds.keys()
        })
    local_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(local_dir))
    return local_dir


def _download_adapter(repo_id, local_dir, token):
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading adapter {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        repo_type="model",
    )
    return local_dir


def _run_eval(model, dataset_dir, split, num_samples, max_seq_length, run_name, output_dir):
    from torch.utils.data import DataLoader

    val_ds = ScreenshotCodeDataset(
        str(dataset_dir),
        split=split,
        tokenizer=model.tokenizer,
        feature_extractor=model.vision_encoder.feature_extractor,
        max_seq_length=max_seq_length,
    )
    loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=ScreenshotCodeCollator(pad_token_id=model.tokenizer.pad_token_id or 0),
    )
    evaluator = VisionModelEvaluator(
        model, model.tokenizer, model.vision_encoder.feature_extractor
    )
    metrics = evaluator.evaluate_from_dataloader(
        loader, num_samples=num_samples, run_name=run_name
    )
    output_path = Path(output_dir) / f"{run_name}.json"
    evaluator.save_results(metrics, output_path)
    logger.info(f"[{run_name}] {metrics}")
    return metrics


def _push_results(folder, repo_id, token):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo="eval",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Phase 3 eval: baseline + finetuned metrics",
    )
    logger.info(f"Eval results pushed: https://huggingface.co/{repo_id}/tree/main/eval")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 HF Skills eval entry")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-revision", default="v2-multimodal")
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--split", default=None, help="test | validation")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/phase3-eval")
    parser.add_argument("--local-dataset-dir", default="/tmp/phase3-eval-dataset")
    parser.add_argument("--local-adapter-dir", default="/tmp/phase3-eval-adapter")
    parser.add_argument("--device-profile", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    params = json.loads(os.environ.get("PHASE3_PARAMS_JSON", "{}"))

    dataset_id = args.dataset_id or params.get("dataset_id") or "cmndcntrlcyber/code-trainer-offsec-dataset"
    dataset_revision = args.dataset_revision or params.get("dataset_revision") or "v2-multimodal"
    adapter_repo = args.adapter_repo or os.environ.get("PHASE3_ADAPTER_REPO") or params.get("adapter_repo")
    split = args.split or os.environ.get("EVAL_SPLIT") or "test"
    num_samples = args.num_samples or int(os.environ.get("EVAL_NUM_SAMPLES") or 200)
    device_profile = args.device_profile or params.get("device_profile", "a100")

    if not adapter_repo:
        raise RuntimeError("adapter_repo not set (env PHASE3_ADAPTER_REPO or --adapter-repo)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dataset_dir = Path(args.local_dataset_dir)
    local_adapter_dir = Path(args.local_adapter_dir)

    logger.info("=" * 60)
    logger.info("PHASE 3 — HF Skills Vision Eval")
    logger.info(f"  dataset:     {dataset_id}@{dataset_revision} (split={split}, n={num_samples})")
    logger.info(f"  adapter:     {adapter_repo}")
    logger.info(f"  output_dir:  {output_dir}")
    logger.info("=" * 60)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required to download adapter / push eval results")

    # 1. Download dataset
    _download_dataset_to_disk(dataset_id, dataset_revision, local_dataset_dir, limit=args.limit)

    # 2. Download adapter
    _download_adapter(adapter_repo, local_adapter_dir, token)

    vision_id = params.get("vision_encoder", "microsoft/swin-base-patch4-window7-224")
    decoder_id = params.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    max_seq_length = params.get("max_seq_length", 2048)

    # 3. Baseline eval — fresh model, random LoRA init
    logger.info(">>> Baseline eval (no fine-tuning)")
    base_model = CodeVisionModel(
        vision_model_id=vision_id,
        decoder_model_id=decoder_id,
        lora_r=params.get("lora_r", 16),
        lora_alpha=params.get("lora_alpha", 32),
        lora_dropout=params.get("lora_dropout", 0.05),
        device_profile=device_profile,
    )
    baseline_metrics = _run_eval(
        base_model, local_dataset_dir, split, num_samples, max_seq_length, "baseline", output_dir
    )
    del base_model
    import torch
    torch.cuda.empty_cache()

    # 4. Fine-tuned eval — load adapter checkpoint
    logger.info(">>> Fine-tuned eval (Phase 3 adapter)")
    ft_model = CodeVisionModel.from_pretrained(
        str(local_adapter_dir),
        vision_model_id=vision_id,
        decoder_model_id=decoder_id,
        lora_r=params.get("lora_r", 16),
        lora_alpha=params.get("lora_alpha", 32),
        lora_dropout=params.get("lora_dropout", 0.05),
        device_profile=device_profile,
    )
    finetuned_metrics = _run_eval(
        ft_model, local_dataset_dir, split, num_samples, max_seq_length, "finetuned", output_dir
    )

    # 5. Summary
    summary = {
        "dataset": f"{dataset_id}@{dataset_revision}",
        "adapter": adapter_repo,
        "split": split,
        "num_samples": num_samples,
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Summary:\n{json.dumps(summary, indent=2, default=str)}")

    if args.skip_push:
        logger.info("--skip-push set; not uploading.")
        return

    _push_results(output_dir, adapter_repo, token)
    logger.info("Phase 3 eval job complete.")


if __name__ == "__main__":
    main()
