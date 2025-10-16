# tools/record_session.py
import argparse, cv2, json, time, datetime
from pathlib import Path
from typing import List, Dict, Optional

# ---------- утилиты ----------
def iso_now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_list_from_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                items.append(s)
    return items

def cycle_idx(cur: int, total: int, step: int = 1) -> int:
    return (cur + step) % total

def draw_hud(frame, text_lines: List[str]):
    y = 30
    for t in text_lines:
        cv2.putText(frame, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        y += 32

# простая транслитерация (каз/рус -> латиница) + нормализация
TRANSLIT = str.maketrans({
    'ә':'a','і':'i','ң':'n','ғ':'g','ү':'u','ұ':'u','қ':'q','ө':'o','һ':'h','ё':'e',
    'Ә':'A','І':'I','Ң':'N','Ғ':'G','Ү':'U','Ұ':'U','Қ':'Q','Ө':'O','Һ':'H','Ё':'E',
    'я':'ya','ю':'yu','ч':'ch','ш':'sh','щ':'sch','ж':'zh','ъ':'','ь':'','ы':'y','э':'e','й':'i',
    'Я':'Ya','Ю':'Yu','Ч':'Ch','Ш':'Sh','Щ':'Sch','Ж':'Zh','Ъ':'','Ь':'','Ы':'Y','Э':'E','Й':'I'
})
def norm_name(s: str) -> str:
    s = s.translate(TRANSLIT).lower()
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in s.strip()).strip("_")

def build_video_filename(participant: str, gloss: str, emotion: str, rep: int, ts: str, ext: str) -> str:
    # Пример: sign_salem__happy__participant_001__2025-10-02T12-30-11_r03.mp4
    ext = ext.lstrip(".")
    return f"sign_{gloss}__{emotion}__{participant}__{ts}_r{rep:02d}.{ext}"

def write_metadata_line(meta_path: Path, meta: Dict):
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

# ---------- аргументы ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Isolated recorder: videos/<split>/<emotion>/<gloss>/*.mp4")
    ap.add_argument("--root", default=".", help="Корень датасета (по умолчанию текущая папка)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Куда писать клипы")
    ap.add_argument("--participant", required=True, help="ID участника (например, participant_001)")
    ap.add_argument("--glosses", nargs="*", help="Список глоссов через пробел")
    ap.add_argument("--glosses_file", help="Путь к .txt со списком глоссов (по одному в строке)")
    ap.add_argument("--emotions", nargs="+", required=True, help="Список эмоций (например: neutral happy angry sad)")
    ap.add_argument("--repeats", type=int, default=6, help="Сколько повторов на комбинацию (глосс×эмоция)")
    ap.add_argument("--pause_sec", type=float, default=1.0, help="Буферная пауза между клипами (сек)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--angle", default="frontal")
    ap.add_argument("--lighting", default="indoor")
    ap.add_argument("--codec", default="mp4v", help="fourcc кодек (mp4v, avc1 и т.п.)")
    ap.add_argument("--out_ext", default="mp4", help="Расширение файла (mp4/mov/avi)")
    # новые:
    ap.add_argument("--countdown_sec", type=float, default=2.0,
                    help="задержка перед стартом записи после SPACE (сек)")
    ap.add_argument("--record_sec", type=float, default=0.0,
                    help="фиксированная длительность записи (сек); 0 = без автостопа")
    return ap.parse_args()

# ---------- основной цикл ----------
def main():
    args = parse_args()

    glosses_from_file = load_list_from_file(args.glosses_file)
    if glosses_from_file is not None:
        glosses = glosses_from_file
    else:
        if not args.glosses:
            raise SystemExit("Нужно указать --glosses или --glosses_file")
        glosses = args.glosses

    emotions = args.emotions
    repeats = max(1, args.repeats)

    root = Path(args.root)
    videos_root = root / "videos" / args.split   # <root>/videos/<split>/<emotion>/<gloss>/*.mp4
    meta_dir = root / "annotations" / "metadata"
    ensure_dir(videos_root); ensure_dir(meta_dir)
    meta_path = meta_dir / "metadata.jsonl"

    session_id = f"session_{iso_now()}"
    print(f"[i] Session: {session_id}")
    print("[keys] SPACE: rec on/off | N: next combo | G: next gloss | E: next emotion | P: pause | S: skip combo | Q: quit")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[!] Не удалось открыть камеру")
        return

    # попытка выставить желаемые параметры
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or args.fps
    try:
        fps_cam = float(fps_cam)
    except Exception:
        fps_cam = float(args.fps)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    # пробный writer с откатом на avc1 при неудаче
    test_path = Path("tmp_codec_test.mp4")
    test_w = cv2.VideoWriter(str(test_path), fourcc, fps_cam, (width, height))
    if not test_w.isOpened():
        print("[!] Кодек", args.codec, "не открылся, пробую 'avc1'")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    test_w.release()
    if test_path.exists(): test_path.unlink()

    g_idx, e_idx, r_idx = 0, 0, 0
    recording = False
    writer = None
    start_time = None
    stop_time = None          # <-- для автостопа по --record_sec
    current_path = None

    # Пауза-буфер перед первым стартом
    next_ready_time = time.time() + args.pause_sec

    def combo_label():
        return f"{glosses[g_idx]} × {emotions[e_idx]} (rep {r_idx+1}/{repeats})"

    def do_stop_record(save_meta=True, quit_flag=False):
        nonlocal recording, writer, start_time, stop_time, width, height, g_idx, e_idx, r_idx, next_ready_time
        if not recording:
            return
        recording = False
        if writer is not None:
            writer.release()
        dur = time.time() - start_time
        if save_meta:
            gloss_raw = glosses[g_idx]
            emotion_raw = emotions[e_idx]
            gloss_n = norm_name(gloss_raw)
            emotion_n = norm_name(emotion_raw)
            ts_part = Path(current_path).stem.split("__")[-1]  # 2025-..._rXX
            vid_id  = f"{args.participant}_{gloss_n}_{emotion_n}_{ts_part}"
            meta = {
                "video_id": vid_id,
                "participant_id": args.participant,
                "file_path": str(current_path.relative_to(root).as_posix()),
                "gloss": gloss_raw,
                "emotion": emotion_raw,
                "repeat_index": r_idx + 1,
                "duration_sec": round(dur, 3),
                "fps": float(fps_cam),
                "resolution": {"w": width, "h": height},
                "session_id": session_id,
                "camera_angle": args.angle,
                "lighting": args.lighting,
                "recording_date": datetime.date.today().isoformat(),
                "split": args.split
            }
            write_metadata_line(meta_path, meta)
            print(f"[REC] Stop {gloss_raw} | {emotion_raw} | rep {r_idx+1} ({dur:.2f}s)")
        stop_time = None
        if not quit_flag:
            # авто-переход к следующей комбинации
            r_idx += 1
            if r_idx >= repeats:
                r_idx = 0
                e_idx += 1
                if e_idx >= len(emotions):
                    e_idx = 0
                    g_idx += 1
                    if g_idx >= len(glosses):
                        print("[i] Комбинации закончились")
                        return
            print(f"[NEXT] -> {combo_label()}")
            next_ready_time = time.time() + args.pause_sec

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Кадр не получен")
            break

        # авто-стоп по таймеру
        if recording and stop_time is not None and time.time() >= stop_time:
            do_stop_record(save_meta=True, quit_flag=False)

        # HUD
        extra = []
        if recording and stop_time is not None:
            extra.append(f"Time left: {max(0.0, stop_time - time.time()):.1f}s")
        hud = [
            f"Participant: {args.participant} | Split: {args.split} | Session: {session_id}",
            f"Combo: {combo_label()} | REC: {'ON' if recording else 'OFF'}",
            f"Next delay: {max(0, next_ready_time-time.time()):.1f}s | Resolution: {width}x{height}@{fps_cam:.1f}"
        ] + extra
        overlay = frame.copy()
        draw_hud(overlay, hud)
        cv2.imshow("sign-recorder (split → emotion → gloss)", overlay)

        if recording and writer is not None:
            # защита: если вдруг размер кадра изменился — переоткрыть writer
            h, w = frame.shape[:2]
            if (w, h) != (width, height):
                writer.release()
                width, height = w, h
                writer = cv2.VideoWriter(str(current_path), fourcc, fps_cam, (width, height))
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF

        # --- выход ---
        if key in (ord('q'), ord('Q')):
            if recording:
                do_stop_record(save_meta=True, quit_flag=True)
            break

        # --- старт/стоп записи ---
        elif key == ord(' '):  # SPACE toggle rec
            now = time.time()
            if not recording:
                if now < next_ready_time:
                    continue

                # --- COUNTDOWN перед стартом ---
                if args.countdown_sec > 0:
                    t_end = time.time() + args.countdown_sec
                    while time.time() < t_end:
                        ok, f = cap.read()
                        if not ok: break
                        remain = max(0.0, t_end - time.time())
                        g = f.copy()
                        cv2.putText(g, f"Start in {remain:.1f}s",
                                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,0), 3)
                        cv2.imshow("sign-recorder (split → emotion → gloss)", g)
                        cv2.waitKey(1)

                gloss_raw = glosses[g_idx]
                emotion_raw = emotions[e_idx]
                gloss_n = norm_name(gloss_raw)
                emotion_n = norm_name(emotion_raw)
                part_n  = norm_name(args.participant)

                ts = iso_now()
                fname = build_video_filename(part_n, gloss_n, emotion_n, r_idx+1, ts, args.out_ext)

                # Иерархия: videos/<split>/<emotion>/<gloss>/
                out_dir = videos_root / emotion_n / gloss_n
                ensure_dir(out_dir)
                current_path = out_dir / fname

                writer = cv2.VideoWriter(str(current_path), fourcc, fps_cam, (width, height))
                if not writer.isOpened():
                    print("[!] VideoWriter не открылся")
                    break
                recording = True
                start_time = time.time()
                stop_time = (start_time + args.record_sec) if args.record_sec > 0 else None
                print(f"[REC] Start {gloss_raw} | {emotion_raw} | rep {r_idx+1} -> {current_path.name}")
            else:
                # ручной стоп
                do_stop_record(save_meta=True, quit_flag=False)

        # --- сервисные клавиши ---
        elif key in (ord('p'), ord('P')):
            next_ready_time = time.time() + args.pause_sec
            print(f"[PAUSE] next record available after {args.pause_sec}s")

        elif key in (ord('s'), ord('S')):  # skip текущую комбинацию
            if recording:
                print("[!] Сначала останови запись (SPACE)")
            else:
                print("[SKIP] combo skipped")
                r_idx += 1
                if r_idx >= repeats:
                    r_idx = 0
                    e_idx += 1
                    if e_idx >= len(emotions):
                        e_idx = 0
                        g_idx += 1
                        if g_idx >= len(glosses):
                            print("[i] Комбинации закончились")
                            break
                print(f"[NEXT] -> {combo_label()}")
                next_ready_time = time.time() + args.pause_sec

        elif key in (ord('n'), ord('N')):  # next combo (вручную)
            if recording:
                print("[!] Сначала останови запись (SPACE)")
            else:
                r_idx += 1
                if r_idx >= repeats:
                    r_idx = 0
                    e_idx += 1
                    if e_idx >= len(emotions):
                        e_idx = 0
                        g_idx += 1
                        if g_idx >= len(glosses):
                            print("[i] Комбинации закончились")
                            break
                print(f"[NEXT] -> {combo_label()}")
                next_ready_time = time.time() + args.pause_sec

        elif key in (ord('g'), ord('G')):  # принудительно следующий глосс
            if recording:
                print("[!] Сначала останови запись (SPACE)")
            else:
                g_idx = cycle_idx(g_idx, len(glosses), 1)
                r_idx = 0
                print(f"[GLOSS] -> {combo_label()}")
                next_ready_time = time.time() + args.pause_sec

        elif key in (ord('e'), ord('E')):  # принудительно следующая эмоция
            if recording:
                print("[!] Сначала останови запись (SPACE)")
            else:
                e_idx = cycle_idx(e_idx, len(emotions), 1)
                r_idx = 0
                print(f"[EMOTION] -> {combo_label()}")
                next_ready_time = time.time() + args.pause_sec

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
