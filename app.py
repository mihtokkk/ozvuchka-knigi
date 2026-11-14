# -*- coding: utf-8 -*-
"""
Аудиотекст — озвучка текста голосом «Вера» (бесплатно, через edge-tts).
Генерация параллельная + прогрессивная выдача: начало можно слушать, пока
остальное догружается. Один голос — Вера (тёплый обработанный).
"""
import os, re, asyncio, tempfile, subprocess
import edge_tts
import gradio as gr

# --- Обработка голоса «Вера» (ffmpeg, -af): тепло снизу, немного «тела», мягкая
#     компрессия (слышнее дыхание), де-эссер против шипения, короткие паузы, ровная
#     громкость. Только естественные фильтры — без эксайтера/сатурации («металл»). ---
VERA_AF = (
    "highpass=f=70,"
    "bass=g=2.5:f=170,"
    "equalizer=f=450:t=q:w=1.2:g=1.5,"
    "equalizer=f=6800:t=q:w=1.5:g=-3.5,"
    "acompressor=threshold=0.08:ratio=2:attack=20:release=250:makeup=1.5,"
    "silenceremove=stop_periods=-1:stop_duration=0.16:stop_threshold=-42dB:stop_silence=0.07,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)
# Голос Вера = Светлана с небольшим сдвигом высоты/темпа + обработка.
VERA = {"voice": "ru-RU-SvetlanaNeural", "pitch": "+5Hz", "rate_bias": -2, "af": VERA_AF}

# Параллельные запросы; размер крупного фрагмента; размер маленького первого (быстрый старт).
CONCURRENCY = int(os.environ.get("TTS_CONCURRENCY", "8"))
MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "1500"))
FIRST_CHARS = int(os.environ.get("TTS_FIRST_CHARS", "160"))

SENT_RE = re.compile(r'[^.!?…]+[.!?…]+|\S[^.!?…]*$')

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

def split_blocks(text):
    """Весь текст → крупные блоки по границам абзацев и предложений (реплики читаются
    Верой как есть). Первый блок — маленький, чтобы быстрее зазвучало начало."""
    text = text.strip()
    if not text:
        return []
    units = []
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if not para:
            continue
        for s in (SENT_RE.findall(para) or [para]):
            s = s.strip()
            if not s:
                continue
            units.extend(_hard_split(s, MAX_CHARS) if len(s) > MAX_CHARS else [s])
    # Гарантируем маленький ПЕРВЫЙ блок (даже если первое предложение длинное) —
    # чтобы начало зазвучало как можно быстрее.
    if units and len(units[0]) > FIRST_CHARS:
        units = _hard_split(units[0], FIRST_CHARS) + units[1:]
    blocks, cur, limit = [], "", FIRST_CHARS
    for u in units:
        if cur and len(cur) + len(u) + 1 > limit:
            blocks.append(cur); cur = u; limit = MAX_CHARS   # после первого — полный лимит
        else:
            cur = (cur + " " + u).strip()
    if cur:
        blocks.append(cur)
    return blocks

def clean(t):
    return re.sub(r"\s+", " ", t).strip()

# -------- Синтез (асинхронный, параллельный) --------

async def _edge_save(text, voice, rate, pitch, out):
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)

def _vera_process(src, dst, af):
    """Обработка «Веры» через ffmpeg (в отдельном потоке). Если упала — отдаём исходный звук."""
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src, "-af", af,
                        "-c:a", "libmp3lame", "-b:a", "96k", "-ar", "24000", dst],
                       check=True, capture_output=True)
        return dst
    except subprocess.CalledProcessError as e:
        print("[Вера] обработка не прошла, выдаю без неё:",
              (e.stderr or b"").decode("utf-8", "ignore")[-400:], flush=True)
        # Привести к единому формату (96к/24кГц) без фильтра — чтобы финальная склейка
        # прошла без перекодирования (-c copy).
        try:
            subprocess.run(["ffmpeg", "-y", "-i", src,
                            "-c:a", "libmp3lame", "-b:a", "96k", "-ar", "24000", dst],
                           check=True, capture_output=True)
            return dst
        except Exception:
            return src

async def _make_segment(sem, idx, text, speed, tmpdir, tries=5):
    """Озвучивает один блок Верой (синтез + обработка) с повторами. → (idx, путь | None)."""
    rate = "%+d%%" % (int(speed) + VERA["rate_bias"])
    raw = os.path.join(tmpdir, "seg%04d.mp3" % idx)
    for attempt in range(tries):
        try:
            async with sem:
                await _edge_save(clean(text), VERA["voice"], rate, VERA["pitch"], raw)
            if os.path.exists(raw) and os.path.getsize(raw) > 0:
                dst = os.path.join(tmpdir, "seg%04d_v.mp3" % idx)
                return idx, await asyncio.to_thread(_vera_process, raw, dst, VERA["af"])
        except Exception:
            pass
        await asyncio.sleep(min(1.0 + attempt, 5))
    print("[фрагмент не удался]", idx, flush=True)
    return idx, None

def concat(files, out, fast=False):
    """Склейка фрагментов в один mp3 одним потоковым проходом (экономно по памяти).
    fast=True — без перекодирования (-c copy), мгновенно, для промежуточных версий."""
    if len(files) == 1:
        return files[0]
    listfile = out + ".txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for fp in files:
            f.write("file '%s'\n" % os.path.abspath(fp))
    if fast:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", out]
    else:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
               "-c:a", "libmp3lame", "-b:a", "96k", "-ar", "24000", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out

async def generate(text, uploaded_file, speed):
    """Async-генератор: отдаёт растущий файл по мере готовности (прогрессивное
    прослушивание), в конце — полную озвучку. Порядок фрагментов сохраняется."""
    if uploaded_file:
        with open(uploaded_file, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    text = (text or "").strip()
    if not text:
        raise gr.Error("Вставьте текст или загрузите .txt файл.")

    blocks = split_blocks(text)
    if not blocks:
        raise gr.Error("Не из чего делать озвучку.")

    tmpdir = tempfile.mkdtemp(prefix="book_")
    n = len(blocks)
    results = [None] * n
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [asyncio.create_task(_make_segment(sem, i, blocks[i], speed, tmpdir))
             for i in range(n)]

    out = os.path.join(tmpdir, "audiobook.mp3")
    emitted, done = 0, 0
    for coro in asyncio.as_completed(tasks):
        idx, path = await coro
        results[idx] = path
        done += 1
        ready = emitted
        while ready < n and results[ready] is not None:
            ready += 1
        # Отдаём начало сразу (первый готовый блок), дальше — по мере роста префикса.
        if ready < n and ready > emitted and (emitted == 0 or ready - emitted >= 3):
            emitted = ready
            part_files = [results[i] for i in range(ready) if results[i]]
            if part_files:
                try:
                    part = await asyncio.to_thread(concat, part_files, out + ".part.mp3", True)
                    yield part, "▶ Можно слушать начало. Озвучено %d из %d, остальное догружается…" % (done, n)
                except Exception:
                    pass

    # Добор упавших фрагментов (по одному — троттлинг мог спасть).
    sem1 = asyncio.Semaphore(1)
    for i in [k for k, p in enumerate(results) if p is None]:
        _, path = await _make_segment(sem1, i, blocks[i], speed, tmpdir, tries=4)
        results[i] = path

    files = [p for p in results if p]
    if not files:
        raise gr.Error("Сервис голосов сейчас перегружен — попробуйте ещё раз через минуту.")
    try:                                     # быстрая склейка без перекодирования (все фрагменты 96к/24кГц)
        final = await asyncio.to_thread(concat, files, out, True)
    except Exception:                        # страховка, если копия всё же не прошла
        final = await asyncio.to_thread(concat, files, out, False)
    failed = n - len(files)
    note = "✅ Готово полностью: фрагментов — %d." % len(files)
    if failed:
        note += " ⚠️ Не удалось — %d (нажмите «Озвучить» ещё раз)." % failed
    yield final, note

# Оформление «Парижский салон»: тёплый айвори, пыльная роза, антикварное золото,
# петербургская синь нотой; заголовок Playfair Display курсивом, золотая рамка-гравюра.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,500;1,600;1,700&family=Lora:wght@400;500&display=swap');
.gradio-container { background:#F1EADB !important; font-family:'Lora',Georgia,serif !important; color:#322D28; }
.gradio-container .main { max-width:980px; margin:0 auto; }
/* рамка-гравюра вокруг всего */
#salon {
  background:#FBF8F2; border:1px solid #B4925A; border-radius:4px;
  padding:26px 32px 30px; margin:16px 0 26px;
  box-shadow:0 0 0 4px #FBF8F2, 0 0 0 5px rgba(180,146,90,.75), 0 18px 46px rgba(50,45,40,.12);
}
/* баннер-заголовок */
#app-banner { text-align:center; margin-bottom:12px; }
#app-banner .title {
  font-family:'Playfair Display',serif; font-style:italic; font-weight:600;
  font-size:48px; line-height:1.05; color:#322D28; margin:0; letter-spacing:.5px;
}
#app-banner .rule { width:120px; height:1px; background:#B4925A; margin:12px auto 8px; }
#app-banner .tag {
  font-family:'Playfair Display',serif; font-style:italic; font-size:15px;
  color:#8A6D4B; letter-spacing:5px; margin:0;
}
#app-banner .fleuron { color:#B4925A; font-size:20px; margin-top:2px; }
/* поля */
label span { font-family:'Lora',serif !important; color:#4A4038 !important; }
textarea, input[type=text] {
  background:#FBF9F4 !important; border:1px solid #D9C9A8 !important; border-radius:3px !important;
  color:#322D28 !important; font-family:'Lora',serif !important;
}
textarea:focus, input:focus { border-color:#28364B !important; box-shadow:0 0 0 2px rgba(40,54,75,.18) !important; }
/* кнопка — нежная пыльная роза, курсивная подпись, золотая обводка */
#go-btn button {
  background:#E9CAD0 !important; background-image:none !important; border:1px solid #B4925A !important;
  color:#6E2E3C !important; font-family:'Playfair Display',serif !important; font-style:italic !important;
  font-weight:600 !important; font-size:20px !important; letter-spacing:.5px !important;
  border-radius:4px !important; box-shadow:0 4px 14px rgba(110,46,60,.14) !important;
}
#go-btn button:hover { background:#EDD3D8 !important; filter:none; }
footer { display:none !important; }
"""
with gr.Blocks(title="Аудиотекст", css=CSS) as demo:
    with gr.Column(elem_id="salon"):
        gr.HTML(
            '<div id="app-banner">'
            '<p class="title">Аудиотекст</p>'
            '<div class="rule"></div>'
            '<p class="tag">TEXTE · VOIX</p>'
            '<div class="fleuron">&#10087;</div>'
            '</div>'
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Текст", lines=12, placeholder="Вставь сюда текст…")
                file = gr.File(label="…или загрузи .txt", file_types=[".txt"], type="filepath")
                speed = gr.Slider(-50, 100, value=0, step=5,
                                  label="Скорость речи, %  (− медленнее · + быстрее)")
                btn = gr.Button("Озвучить", variant="primary", size="lg", elem_id="go-btn")
            with gr.Column(scale=2):
                audio = gr.Audio(label="Аудио — слушать и скачать", type="filepath")
                status = gr.Markdown()
        btn.click(generate, [text, file, speed], [audio, status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
