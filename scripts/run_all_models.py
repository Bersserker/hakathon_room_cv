import subprocess
from pathlib import Path
from datetime import datetime

CONFIGS = [
    "class_weights_effective_num_0995.yaml",
    "class_weights_sqrt_inv.yaml",
    "cv03_baseline_ce.yaml",
    "cv03_ratio_weighting.yaml",
    "cv03_balanced_sampler.yaml",
    "cv03_weighted_ce.yaml",
    "cv04_convnext_small_earlystop_gpu.yaml",
    "cv05_clip_vit_b32.yaml",
    "cv06_clip_vit_l14_laion.yaml",
    "image_albumentations_v1.yaml",
    "image_baseline_v1.yaml",
    "image_safe_aug_v1.yaml",
    "imbalance_rc1.yaml",
    "model2_v1.yaml",
    "sampler_class_aware_l05.yaml",
    "weak_images_v1.yaml",
]

FOLDS = [0, 1, 2, 3, 4]

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def run_config_fold(cfg_name: str, fold: int):
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {cfg_name} | fold={fold}")
    print(f"{'=' * 60}")

    script = (
        "src/training/train_image_clip.py"
        if "clip" in cfg_name.lower()
        else "src/training/train_image.py"
    )

    log_file = LOG_DIR / f"{cfg_name.replace('.yaml', '')}_fold{fold}.log"

    cmd = [
        "uv",
        "run",
        "python",
        script,
        "--config",
        f"configs/model/{cfg_name}",
        "--fold",
        str(fold),
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        process.wait()

    if process.returncode == 0:
        print(f"✅ SUCCESS: {cfg_name} | fold={fold}")
    else:
        print(f"❌ FAILED: {cfg_name} | fold={fold}")

    return process.returncode


def main():
    start = datetime.now()

    for cfg in CONFIGS:
        for fold in FOLDS:
            code = run_config_fold(cfg, fold)

            if code != 0:
                print("🛑 STOPPING due to error")
                return

    print("\n========================")
    print("DONE")
    print(f"Total time: {datetime.now() - start}")
    print("========================")


if __name__ == "__main__":
    main()