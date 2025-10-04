# tools/migrate_layout.py
import argparse, re, json, shutil
from pathlib import Path

FILE_PAT = re.compile(r"""
    ^sign_                             # prefix
    (?P<gloss>.+?)                     # GLOSS (non-greedy)
    __(?P<emotion>[^_]+)__             # __EMOTION__
    (?P<ts>[^_]+)                      # timestamp like 2025-10-02T12-30-11
    _r(?P<rep>\d{2})                   # _r01
    \.(?P<ext>mp4|mov|avi|mkv)$        # extension
""", re.VERBOSE | re.IGNORECASE)

def norm_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in s.strip()).strip("_")

def find_videos_emotion_first(videos_root: Path):
    """
    Ищет файлы по маске:
    videos/<split>/<participant>/emotion_<emotion>/sign_...mp4
    Возвращает список путей.
    """
    for split in ("train", "val", "test"):
        split_dir = videos_root / split
        if not split_dir.exists():
            continue
        for participant_dir in split_dir.iterdir():
            if not participant_dir.is_dir():
                continue
            for emotion_dir in participant_dir.glob("emotion_*"):
                if not emotion_dir.is_dir():
                    continue
                for f in emotion_dir.rglob("sign_*.*"):
                    if f.is_file():
                        yield split_dir, participant_dir, emotion_dir, f

def compute_destination(videos_root: Path, split_dir: Path, participant_dir: Path, file_path: Path, gloss: str, emotion: str):
    """
    Новая цель: videos/<split>/<participant>/<GLOSS>/<EMOTION>/<filename>
    """
    gloss_n = norm_name(gloss)
    emotion_n = norm_name(emotion)
    dest_dir = videos_root / split_dir.name / participant_dir.name / gloss_n / emotion_n
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / file_path.name

def migrate_files(videos_root: Path, dry_run: bool = True):
    moved, skipped, errors = 0, 0, 0
    for split_dir, participant_dir, emotion_dir, f in find_videos_emotion_first(videos_root):
        m = FILE_PAT.match(f.name)
        if not m:
            print(f"[SKIP] filename not matched: {f}")
            skipped += 1
            continue
        gloss = m.group("gloss")
        emotion_from_name = m.group("emotion")
        # Если эмоция в пути «emotion_<emotion>» не совпадает с из имени файла — логируем, но верим имени файла
        emotion_from_dir = emotion_dir.name.replace("emotion_", "", 1)
        if norm_name(emotion_from_dir) != norm_name(emotion_from_name):
            print(f"[WARN] emotion mismatch dir({emotion_from_dir}) vs name({emotion_from_name}) -> using name")
        dest = compute_destination(videos_root, split_dir, participant_dir, f, gloss, emotion_from_name)
        if dest.exists():
            print(f"[SKIP] already exists: {dest}")
            skipped += 1
            continue
        print(f"[MOVE] {f} -> {dest}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
        moved += 1
        # Пустые папки после перемещения почистим в конце
    return moved, skipped, errors

def prune_empty_dirs(root: Path):
    # Удаляет пустые «emotion_*» каталоги (и их родителей, если опустели)
    removed = 0
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        try:
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
                removed += 1
        except Exception:
            pass
    return removed

def fix_metadata(meta_path: Path, videos_root: Path, dry_run: bool = True):
    """
    Переписывает file_path в metadata.jsonl согласно новой схеме.
    Ожидает, что file_path указывает на относительный путь внутри корня датасета.
    Логику извлекаем из имени файла — это надёжнее.
    """
    if not meta_path.exists():
        print(f"[INFO] metadata not found: {meta_path}")
        return 0, 0
    temp_out = meta_path.with_suffix(".jsonl.tmp")
    updated, kept = 0, 0

    with meta_path.open("r", encoding="utf-8") as fin, temp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                print(f"[META:SKIP] bad json: {line[:120]}...")
                kept += 1
                fout.write(line + "\n")
                continue

            old_path = rec.get("file_path", "")
            # пробуем понять из имени файла gloss/emotion
            fname = Path(old_path).name
            m = FILE_PAT.match(fname)
            if not m:
                # оставим как есть
                kept += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            gloss = m.group("gloss")
            emotion = m.group("emotion")
            # split и participant восстановим из старого пути, если там есть
            # ожидается старый вид: videos/<split>/<participant>/emotion_<emotion>/filename
            parts = Path(old_path).parts
            try:
                idx = parts.index("videos")
            except ValueError:
                # если путь не начинается с videos/ — оставим как есть
                kept += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            split = parts[idx+1] if len(parts) > idx+1 else rec.get("split", "train")
            participant = parts[idx+2] if len(parts) > idx+2 else rec.get("participant_id", "participant_unknown")

            new_rel = Path("videos") / split / participant / norm_name(gloss) / norm_name(emotion) / fname
            if old_path != new_rel.as_posix():
                rec["file_path"] = new_rel.as_posix()
                updated += 1
            else:
                kept += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if dry_run:
        temp_out.unlink(missing_ok=True)
        print(f"[META] dry-run: {updated} would update, {kept} keep")
        return updated, kept
    else:
        backup = meta_path.with_suffix(".jsonl.bak")
        if backup.exists():
            backup.unlink()
        meta_path.rename(backup)
        temp_out.rename(meta_path)
        print(f"[META] updated: {updated}, kept: {kept}, backup: {backup.name}")
        return updated, kept

def main():
    ap = argparse.ArgumentParser(description="Migrate videos from emotion_first to gloss_first layout and fix metadata.")
    ap.add_argument("--root", default=".", help="Корень датасета (по умолчанию текущая папка)")
    ap.add_argument("--dry_run", type=int, default=1, help="1=только показать, 0=выполнить перенос")
    ap.add_argument("--fix_metadata", type=int, default=1, help="1=поправить paths в annotations/metadata/metadata.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    videos_root = root / "videos"
    meta_path = root / "annotations" / "metadata" / "metadata.jsonl"

    print(f"[INFO] root={root.resolve()}")
    print("[STEP] scan & migrate files...")
    moved, skipped, errors = migrate_files(videos_root, dry_run=bool(args.dry_run))
    print(f"[FILES] moved={moved}, skipped={skipped}, errors={errors}")

    if not args.dry_run:
        print("[STEP] prune empty dirs...")
        removed = prune_empty_dirs(videos_root)
        print(f"[CLEAN] removed empty dirs: {removed}")

    if args.fix_metadata:
        print("[STEP] fix metadata.jsonl...")
        fix_metadata(meta_path, videos_root, dry_run=bool(args.dry_run))

    print("[DONE]")

if __name__ == "__main__":
    main()
