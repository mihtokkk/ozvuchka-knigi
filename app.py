# -*- coding: utf-8 -*-
"""
Озвучка книги — живые нейросетевые голоса (бесплатно, через edge-tts).
Работает локально, в Google Colab и на Hugging Face Spaces.
Голоса: Анна, Дмитрий (чистые нейроголоса) и Вера (обработанный тёплый голос).
"""
import os, re, time, tempfile, asyncio, subprocess
import edge_tts
import gradio as gr

# --- Обработка голоса «Вера» (ffmpeg): тепло, де-эссер, лёгкая хрипотца-придыхание,
#     короткие паузы. Применяется к каждому её фрагменту после синтеза. ---
VERA_FX = (
    "[0:a]highpass=f=80,equalizer=f=280:t=q:w=1.2:g=-1.5,bass=g=2:f=180,"
    "acompressor=threshold=0.06:ratio=2.5:attack=15:release=220:makeup=2,"
    "aexciter=amount=1.2:drive=4:blend=1:freq=6000:ceil=14000,"
    "equalizer=f=6800:t=q:w=1.6:g=-3.5,treble=g=1.5:f=10000,"
    "equalizer=f=4000:t=q:w=2:g=1,dynaudnorm=f=250:g=7[vpre];"
    "[vpre]asplit=2[v][vkey];"
    "anoisesrc=color=pink:amplitude=0.03:seed=7[n0];"
    "[n0]highpass=f=3000,lowpass=f=7000[nf];"
    "[nf][vkey]sidechaingate=threshold=0.03:ratio=5:attack=3:release=50[ng];"
    "[ng]volume=0.3[ngl];"
    "[v][ngl]amix=inputs=2:duration=first:normalize=0,"
    "silenceremove=stop_periods=-1:stop_duration=0.14:stop_threshold=-42dB:stop_silence=0.06,"
    "alimiter=limit=0.95[out]"
)

# Каждый голос: базовый нейроголос + опц. сдвиг высоты/темпа + опц. обработка fx.
VOICES = {
    "Анна — женский, русский":      {"voice": "ru-RU-SvetlanaNeural"},
    "Дмитрий — мужской, русский":   {"voice": "ru-RU-DmitryNeural"},
    "Вера — тёплый, обработанный":  {"voice": "ru-RU-SvetlanaNeural",
                                     "pitch": "+5Hz", "rate_bias": -2, "fx": VERA_FX},
}
DEFAULT_NARR = "Анна — женский, русский"
DEFAULT_CHAR = "Дмитрий — мужской, русский"

QUOTE_RE = re.compile(r'[«„"][^»"“]*[»"“]')

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

def clean(t):
    t = t.replace("«", "").replace("»", "").replace("„", "").replace("“", "").replace('"', "")
    return re.sub(r"\s+", " ", t).strip()

async def _save(text, voice, rate, pitch, out):
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)

def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-y"] + args, check=True, capture_output=True)

def _finalize(src, dst, fx):
    """Приводит фрагмент к единому формату 44.1кГц/192к; для «Веры» прогоняет обработку."""
    if fx:
        _ffmpeg(["-i", src, "-filter_complex", fx, "-map", "[out]",
                 "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", dst])
    else:
        _ffmpeg(["-i", src, "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", dst])

def synth_segment(text, vcfg, speed, out, tries=6):
    """Синтез фрагмента нужным голосом (+обработка). Повторы — сервис иногда троттлит."""
    voice = vcfg["voice"]
    pitch = vcfg.get("pitch", "+0Hz")
    rate = "%+d%%" % (int(speed) + vcfg.get("rate_bias", 0))
    fx = vcfg.get("fx")
    raw = out + ".raw.mp3"
    for attempt in range(tries):
        try:
            asyncio.run(_save(text, voice, rate, pitch, raw))
            if os.path.exists(raw) and os.path.getsize(raw) > 0:
                _finalize(raw, out, fx)
                return True
        except Exception:
            pass
        time.sleep(min(1.5 + attempt, 6))  # нарастающая пауза
    return False

def concat(files, out):
    """Склейка фрагментов в один файл. Все фрагменты уже в едином формате 44.1кГц."""
    listfile = out + ".txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for fp in files:
            f.write("file '%s'\n" % os.path.abspath(fp))
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", listfile,
             "-c:a", "libmp3lame", "-b:a", "192k", out])

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
    segs = split_segments(text, dialogue_mode)
    if not segs:
        raise gr.Error("Не из чего делать озвучку.")

    tmpdir = tempfile.mkdtemp(prefix="book_")
    # jobs: [индекс, текст, конфиг-голоса, файл, готово?]
    jobs = []
    for i, (seg_text, role) in enumerate(segs):
        vcfg = char_cfg if role == "dlg" else narr_cfg
        jobs.append([i, clean(seg_text), vcfg,
                     os.path.join(tmpdir, "seg%04d.mp3" % i), False])
    total = len(jobs)

    # Первый проход — с лёгкой паузой между фрагментами против троттлинга.
    for n, job in enumerate(jobs):
        progress(n / total, desc="Озвучиваю фрагмент %d из %d" % (n + 1, total))
        if synth_segment(job[1], job[2], speed, job[3]):
            job[4] = True
        time.sleep(0.3)

    # Второй проход — добиваем неудавшиеся фрагменты, чтобы не было пропусков.
    pending = [j for j in jobs if not j[4]]
    if pending:
        time.sleep(3)
        for k, job in enumerate(pending):
            progress((k + 1) / len(pending), desc="Повторяю сложные фрагменты (%d)" % len(pending))
            if synth_segment(job[1], job[2], speed, job[3], tries=8):
                job[4] = True
            time.sleep(1)

    files = [j[3] for j in jobs if j[4]]          # в исходном порядке
    failed = total - len(files)

    if not files:
        raise gr.Error("Сервис голосов сейчас перегружен — попробуйте ещё раз через минуту.")

    if len(files) == 1:
        final = files[0]
    else:
        final = os.path.join(tmpdir, "audiobook.mp3")
        concat(files, final)

    note = "✅ Готово: озвучено фрагментов — %d." % len(files)
    if failed:
        note += " ⚠️ Не удалось — %d (попробуйте переозвучить)." % failed
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
        "<sub>Голоса Microsoft Neural через edge-tts. Голос «Вера» обрабатывается и считается "
        "чуть дольше. Длинные книги озвучиваются по фрагментам. Если фрагмент не удался — "
        "просто нажми «Озвучить» ещё раз.</sub>"
    )

if __name__ == "__main__":
    # На хостинге (Render/Fly/Koyeb и т.п.) порт задаётся переменной PORT; локально — 7860.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
