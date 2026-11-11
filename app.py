# -*- coding: utf-8 -*-
"""
Озвучка книги — живые нейросетевые голоса (бесплатно, через edge-tts).
Голоса: Анна, Дмитрий (чистые нейроголоса) и Вера (тёплый обработанный голос).
Генерация фрагментов — ПАРАЛЛЕЛЬНАЯ (несколько запросов к edge-tts одновременно),
с сохранением исходного порядка и добором неудавшихся фрагментов.
"""
import os, re, asyncio, tempfile, subprocess
import edge_tts
import gradio as gr

# --- Обработка голоса «Вера» (ffmpeg, -af): ТОЛЬКО естественные фильтры — тепло снизу,
#     немного «тела», мягкая компрессия (слышнее дыхание), де-эссер против шипения,
#     короткие паузы, ровная громкость. Без эксайтера/сатурации (они давали «металл»). ---
VERA_AF = (
    "highpass=f=70,"
    "bass=g=2.5:f=170,"
    "equalizer=f=450:t=q:w=1.2:g=1.5,"
    "equalizer=f=6800:t=q:w=1.5:g=-3.5,"
    "acompressor=threshold=0.08:ratio=2:attack=20:release=250:makeup=1.5,"
    "silenceremove=stop_periods=-1:stop_duration=0.16:stop_threshold=-42dB:stop_silence=0.07,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)

# Каждый голос: базовый нейроголос + опц. сдвиг высоты/темпа + опц. обработка af.
VOICES = {
    "Анна — женский, русский":      {"voice": "ru-RU-SvetlanaNeural"},
    "Дмитрий — мужской, русский":   {"voice": "ru-RU-DmitryNeural"},
    "Вера — тёплый, обработанный":  {"voice": "ru-RU-SvetlanaNeural",
                                     "pitch": "+5Hz", "rate_bias": -2, "af": VERA_AF},
}
DEFAULT_NARR = "Анна — женский, русский"
DEFAULT_CHAR = "Дмитрий — мужской, русский"

# Сколько фрагментов озвучивать одновременно и максимальная длина фрагмента.
CONCURRENCY = int(os.environ.get("TTS_CONCURRENCY", "4"))
MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "500"))

QUOTE_RE = re.compile(r'[«„"][^»"“]*[»"“]')
SENT_RE = re.compile(r'[^.!?…]+[.!?…]+|\S[^.!?…]*$')

def split_segments(text, dialogue_mode):
    """Делит текст на (фрагмент, роль): 'narr' — рассказчик, 'dlg' — реплика в кавычках."""
    text = text.strip()
    if not dialogue_mode:
        return [(text, "narr")] if text else []
    parts, last = [], 0
    for m in QUOTE_RE.finditer(text):
        if m.start() > last:
            s = text[last:m.start()].strip()
            if s: parts.append((s, "narr"))
        d = m.group().strip()
        if d: parts.append((d, "dlg"))
        last = m.end()
    tail = text[last:].strip()
    if tail: parts.append((tail, "narr"))
    return parts or ([(text, "narr")] if text else [])

def _hard_split(s, n):
    """Режет слишком длинное предложение по словам, не длиннее n символов."""
    words, parts, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > n:
            parts.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur: parts.append(cur)
    return parts or [s]

def chunk(text, role, max_chars=MAX_CHARS):
    """Бьёт кусок на фрагменты по границам предложений, не длиннее max_chars (роль сохраняется)."""
    text = text.strip()
    if not text:
        return []
    sentences = SENT_RE.findall(text) or [text]
    pieces = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        pieces.extend(_hard_split(s, max_chars) if len(s) > max_chars else [s])
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur); cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        chunks.append(cur)
    return [(c, role) for c in chunks]

def clean(t):
    t = t.replace("«", "").replace("»", "").replace("„", "").replace("“", "").replace('"', "")
    return re.sub(r"\s+", " ", t).strip()

# -------- Синтез (асинхронный, параллельный) --------

async def _edge_save(text, voice, rate, pitch, out):
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)

def _vera_process(src, dst, af):
    """Обработка «Веры» через ffmpeg (блокирующая, запускается в отдельном потоке).
    Если обработка упала — возвращаем исходный звук, чтобы голос всё равно звучал."""
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src, "-af", af,
                        "-c:a", "libmp3lame", "-b:a", "96k", "-ar", "24000", dst],
                       check=True, capture_output=True)
        return dst
    except subprocess.CalledProcessError as e:
        print("[Вера] обработка не прошла, выдаю без неё:",
              (e.stderr or b"").decode("utf-8", "ignore")[-400:], flush=True)
        return src

async def _make_segment(sem, idx, text, vcfg, speed, tmpdir, tries=5):
    """Озвучивает один фрагмент с повторами. Возвращает (idx, путь-к-файлу | None)."""
    voice = vcfg["voice"]
    pitch = vcfg.get("pitch", "+0Hz")
    rate = "%+d%%" % (int(speed) + vcfg.get("rate_bias", 0))
    af = vcfg.get("af")
    raw = os.path.join(tmpdir, "seg%04d.mp3" % idx)
    for attempt in range(tries):
        try:
            async with sem:                       # ограничиваем одновременные запросы
                await _edge_save(clean(text), voice, rate, pitch, raw)
            if os.path.exists(raw) and os.path.getsize(raw) > 0:
                if af:                            # «Вера» — обработка в отдельном потоке
                    dst = os.path.join(tmpdir, "seg%04d_v.mp3" % idx)
                    path = await asyncio.to_thread(_vera_process, raw, dst, af)
                    return idx, path
                return idx, raw                   # обычные голоса — без ffmpeg (быстро)
        except Exception:
            pass
        await asyncio.sleep(min(1.0 + attempt, 5))
    print("[фрагмент не удался]", idx, flush=True)
    return idx, None

async def _synthesize_all(jobs, speed, tmpdir, progress):
    """Параллельная озвучка всех фрагментов + добор неудавшихся. Порядок сохраняется."""
    sem = asyncio.Semaphore(CONCURRENCY)
    results = [None] * len(jobs)
    tasks = [asyncio.create_task(_make_segment(sem, i, t, vcfg, speed, tmpdir))
             for i, (t, vcfg) in enumerate(jobs)]
    done = 0
    for coro in asyncio.as_completed(tasks):
        idx, path = await coro
        results[idx] = path
        done += 1
        try:
            progress(done / len(jobs), desc="Озвучено %d из %d" % (done, len(jobs)))
        except Exception:
            pass
    # Спасательный проход по неудавшимся — по одному, без гонки (троттлинг мог спасть).
    failed = [i for i, p in enumerate(results) if p is None]
    if failed:
        sem1 = asyncio.Semaphore(1)
        for i in failed:
            idx, path = await _make_segment(sem1, i, jobs[i][0], jobs[i][1], speed, tmpdir, tries=4)
            results[idx] = path
    return results

def concat(files, out):
    """Склейка фрагментов в один mp3 одним потоковым проходом (экономно по памяти)."""
    if len(files) == 1:
        return files[0]
    listfile = out + ".txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for fp in files:
            f.write("file '%s'\n" % os.path.abspath(fp))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                    "-c:a", "libmp3lame", "-b:a", "96k", "-ar", "24000", out],
                   check=True, capture_output=True)
    return out

def generate(text, uploaded_file, dialogue_mode, narrator_label, character_label, speed,
             progress=gr.Progress()):
    if uploaded_file:
        with open(uploaded_file, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    text = (text or "").strip()
    if not text:
        raise gr.Error("Вставьте текст или загрузите .txt файл.")

    narr_cfg = VOICES[narrator_label]
    char_cfg = VOICES[character_label]

    # Текст → фрагменты (рассказчик/реплики) → дробление по предложениям → задания с голосом.
    jobs = []
    for seg_text, role in split_segments(text, dialogue_mode):
        for piece, r in chunk(seg_text, role):
            jobs.append((piece, char_cfg if r == "dlg" else narr_cfg))
    if not jobs:
        raise gr.Error("Не из чего делать озвучку.")

    tmpdir = tempfile.mkdtemp(prefix="book_")
    results = asyncio.run(_synthesize_all(jobs, speed, tmpdir, progress))

    files = [p for p in results if p]             # уже в исходном порядке
    failed = len(results) - len(files)
    if not files:
        raise gr.Error("Сервис голосов сейчас перегружен — попробуйте ещё раз через минуту.")

    final = concat(files, os.path.join(tmpdir, "audiobook.mp3"))
    note = "✅ Готово: озвучено фрагментов — %d." % len(files)
    if failed:
        note += " ⚠️ Не удалось — %d (нажмите «Озвучить» ещё раз)." % failed
    return final, note

with gr.Blocks(title="Озвучка книги") as demo:
    gr.Markdown(
        "# 🎧 Озвучка книги\n"
        "Живые нейросетевые голоса (бесплатно). Вставь текст или загрузи **.txt** — "
        "получишь аудио, которое можно слушать и скачать. Реплики в кавычках «…» читаются "
        "отдельным голосом.\n\n"
        "Голоса: **Анна** и **Дмитрий** — чистые нейроголоса; **Вера** — тёплый обработанный голос."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="Текст книги", lines=12,
                placeholder="Жил-был на свете человек. «Кто здесь?» — тихо спросил он. И никто не ответил…",
            )
            file = gr.File(label="…или загрузи .txt", file_types=[".txt"], type="filepath")
            dialogue = gr.Checkbox(label="Реплики в кавычках — отдельным голосом", value=True)
            with gr.Row():
                narrator = gr.Dropdown(list(VOICES), value=DEFAULT_NARR, label="📖 Голос рассказчика")
                character = gr.Dropdown(list(VOICES), value=DEFAULT_CHAR, label="💬 Голос персонажа")
            speed = gr.Slider(-50, 100, value=0, step=5,
                              label="Скорость речи, %  (− медленнее · + быстрее)")
            btn = gr.Button("▶ Озвучить", variant="primary", size="lg")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Результат — слушай и скачивай", type="filepath")
            status = gr.Markdown()
    btn.click(generate, [text, file, dialogue, narrator, character, speed], [audio, status])
    gr.Markdown(
        "<sub>Голоса Microsoft Neural через edge-tts. Фрагменты озвучиваются параллельно. "
        "Если фрагмент не удался — просто нажми «Озвучить» ещё раз.</sub>"
    )

if __name__ == "__main__":
    # На хостинге (Render и т.п.) порт задаётся переменной PORT; локально — 7860.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
