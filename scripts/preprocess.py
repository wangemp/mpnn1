import os
from collections import Counter, defaultdict

from configs import Config
from utils import get_logger, set_seed
from preprocess.build_dataset import build_and_save_dataset


def _is_php(lang: str) -> bool:
    return lang == "php"


def _is_asp(lang: str) -> bool:
    return lang.startswith("asp")


def _count_raw_files(raw_dir: str, exts):
    """
    Count language distribution in raw data folders.
    We use file extension heuristic here (robust and fast).
    """
    counts = {
        "webshell": {"php": 0, "asp": 0, "other": 0},
        "normal": {"php": 0, "asp": 0, "other": 0},
    }

    for label in ["webshell", "normal"]:
        root = os.path.join(raw_dir, label)
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                fp = os.path.join(dirpath, fn)
                if ext == ".php":
                    counts[label]["php"] += 1
                elif ext in (".asp", ".aspx"):
                    counts[label]["asp"] += 1
                elif ext in exts:
                    counts[label]["other"] += 1
                else:
                    # ignore unrelated files
                    pass
    return counts


def _count_pt_lang(data_list):
    """
    Count (label, lang) distribution in processed graphs.
    Expects each Data has:
      - y: tensor([0/1])
      - lang: str ('php' / 'asp_vb' / 'asp_js')
    """
    # label_name -> lang_group -> count
    out = {
        "webshell": {"php": 0, "asp": 0, "other": 0},
        "normal": {"php": 0, "asp": 0, "other": 0},
    }

    for d in data_list:
        y = int(d.y.view(-1)[0].item())
        label = "webshell" if y == 1 else "normal"
        lang = getattr(d, "lang", "other")

        if _is_php(lang):
            out[label]["php"] += 1
        elif _is_asp(lang):
            out[label]["asp"] += 1
        else:
            out[label]["other"] += 1

    return out


def _fmt_counts(title: str, c: dict) -> str:
    lines = [title]
    for label in ["webshell", "normal"]:
        php = c[label]["php"]
        asp = c[label]["asp"]
        oth = c[label]["other"]
        total = php + asp + oth
        lines.append(
            f"  - {label}: total={total} | php={php} | asp={asp} | other={oth}"
        )
    return "\n".join(lines)


def main():
    cfg = Config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    logger = get_logger("preprocess", os.path.join(cfg.log_dir, "preprocess.log"))
    logger.info("Start preprocessing...")
    logger.info(f"Raw dir: {cfg.data_raw_dir}")
    logger.info(f"Processed dir: {cfg.data_processed_dir}")

    # ===== Raw distribution (by extension) =====
    raw_counts = _count_raw_files(cfg.data_raw_dir, cfg.exts)
    logger.info(_fmt_counts("[RAW] file distribution (by extension)", raw_counts))

    # ===== Build dataset =====
    meta = build_and_save_dataset(
        raw_dir=cfg.data_raw_dir,
        processed_dir=cfg.data_processed_dir,
        cache_dir=cfg.data_cache_dir,
        exts=cfg.exts,
        behavior_types=cfg.behavior_types,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed
    )

    logger.info(f"Build done. Meta: {meta}")

    # ===== Processed distribution (by stored d.lang) =====
    import torch

    train_list = torch.load(os.path.join(cfg.data_processed_dir, "train.pt"))
    val_list = torch.load(os.path.join(cfg.data_processed_dir, "val.pt"))
    test_list = torch.load(os.path.join(cfg.data_processed_dir, "test.pt"))

    logger.info(_fmt_counts("[PROCESSED] train distribution (by d.lang)", _count_pt_lang(train_list)))
    logger.info(_fmt_counts("[PROCESSED] val distribution (by d.lang)", _count_pt_lang(val_list)))
    logger.info(_fmt_counts("[PROCESSED] test distribution (by d.lang)", _count_pt_lang(test_list)))

    logger.info("Preprocess finished.")


if __name__ == "__main__":
    main()
