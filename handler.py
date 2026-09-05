import base64
import os
import re
import sys
import tempfile
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch

# Speed control (time-stretch, pitch မထိခိုက်) အတွက် librosa ကို optional dependency
# အနေနဲ့ သုံးထားပါသည် — install မထားရင်တောင် crash မဖြစ်ဘဲ speed adjustment ကိုသာ
# skip လုပ်ပါမည် (pip install librosa ထည့်ပေးမှ 'speed' input အလုပ်လုပ်ပါမည်)
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    _LIBROSA_AVAILABLE = False

MODEL_ID = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "/runpod-volume/models/VoxCPM2")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "2000"))
WORKER_SECRET = os.environ.get("WORKER_SECRET")

# VoxCPM2's model.generate() has its own internal "max_len" cap (maximum token
# length during generation) that is SEPARATE from MAX_CHARS above. If we don't
# pass it explicitly, the model falls back to a low internal default — which is
# why long text was getting cut off to a fixed ~8 seconds of audio no matter how
# long the input was. 4096 tokens is the value VoxCPM2's own docs use as the
# default for generate_streaming(), so we request the same ceiling here for the
# regular (non-streaming) generate() call. Raise via env var if you still see
# long text getting cut off.
VOXCPM_MAX_LEN = int(os.environ.get("VOXCPM_MAX_LEN", "4096"))

# Speech ကို "AI ပီပီ တစ်ဆက်တည်းမြန်မြန်" မဟုတ်ဘဲ သဘာဝကျအောင် အနည်းငယ် နှေးအောင်
# default ထား — input.speed နဲ့ request တစ်ခုချင်းစီက override လုပ်နိုင်ပါသည်
# (rate < 1.0 = ပိုနှေး, rate > 1.0 = ပိုမြန်, 1.0 = model ရဲ့ မူလ speed)
DEFAULT_SPEED = float(os.environ.get("DEFAULT_SPEED", "1.0"))
MIN_SPEED = float(os.environ.get("MIN_SPEED", "0.5"))
MAX_SPEED = float(os.environ.get("MAX_SPEED", "1.6"))

# [pause] marker မှာ duration မတိတိကျကျ ထည့်မထားရင် သုံးမည့် default silence (စက္ကန့်)
DEFAULT_PAUSE_SECONDS = float(os.environ.get("DEFAULT_PAUSE_SECONDS", "0.6"))

# torch.compile (optimize=True) ကို Worker Cold Start တိုင်း အသစ်ပြန် Compile ရလို့
# ~80-90 စက္ကန့် Warmup Cost ပါဝင်လာပါသည်။ Cold Start မကြာခဏ ဖြစ်နေရင် ဒီ Env Var ကို
# "false" ထားပြီး Off ချနိုင်ပါသည် — Warmup Cost ကျသွားမည်၊ Steady-state Generation
# အနည်းငယ် နှေးနိုင်ပါသည်။ Default က "true" (Compile ဖွင့်ထား - မူလအတိုင်း)။
VOXCPM_OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "true").strip().lower() not in ("false", "0", "no")

# VoxCPM2 has no separate "voice_type"/gender argument in model.generate() —
# gender/voice is steered by a natural-language description prepended to the
# text in parentheses ("Voice Design" mode, only when no reference audio).
VOICE_DESCRIPTIONS = {
    "female": "a young woman, clear and pleasant voice",
    "male": "a young man, clear voice",
    "child": "a young child, playful high-pitched voice",
}

# Multi Voice (dialogue): each line can be tagged with a speaker prefix like
# "M:", "F:", "C:" (or Male:/Female:/Child:) to switch voice per line, e.g.:
#   F: မင်္ဂလာပါ ခင်ဗျာ
#   M: ဟုတ်ကဲ့ ကျွန်တော် ကူညီပေးပါ့မယ်
#   C: ကျွန်တော်လည်း ပါဝင်မယ်
SPEAKER_ALIASES = {
    "m": "male", "male": "male", "man": "male", "boy": "male",
    "f": "female", "female": "female", "woman": "female", "girl": "female",
    "c": "child", "child": "child", "kid": "child",
}
_SPEAKER_TAG_RE = re.compile(r"^\s*([A-Za-z]{1,6})\s*[:：]\s*(.+)$")


def parse_multi_speaker_lines(text):
    """text ကို line အလိုက် ပိုင်းပြီး (voice_key_or_None, line_text) tuple list ပြန်ပေးသည်။
    Tag မပါတဲ့ line က ယခင် speaker အသံကို ဆက်သုံးမည်။"""
    segments = []
    last_voice = None
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        voice = last_voice
        match = _SPEAKER_TAG_RE.match(line)
        if match:
            mapped = SPEAKER_ALIASES.get(match.group(1).strip().lower())
            if mapped:
                voice = mapped
                line = match.group(2).strip()
        segments.append((voice, line))
        last_voice = voice
    return segments


# [pause] (default duration) သို့မဟုတ် [pause:1.2] (တိတိကျကျ 1.2 စက္ကန့်) ပုံစံနှစ်မျိုးလုံး
# ကို လက်ခံသည် — case-insensitive
_PAUSE_RE = re.compile(r"\[\s*pause(?:\s*:\s*(\d+(?:\.\d+)?))?\s*\]", re.IGNORECASE)


def split_on_pauses(text, default_pause_s):
    """text ထဲက [pause]/[pause:N] marker များအလိုက် (segment_text, pause_after_seconds)
    tuple list အဖြစ် ခွဲပေးသည်။ marker မပါရင် pause=0 ပါတဲ့ segment တစ်ခုတည်း ပြန်ပေးမည်
    (ဆိုလိုသည်မှာ ဒီ function ကို ခေါ်ခြင်းက ရှိပြီးသား output ကို ဘာမှ မပြောင်းပါ)။"""
    segments = []
    last_end = 0
    for m in _PAUSE_RE.finditer(text):
        seg_text = text[last_end:m.start()].strip()
        dur = float(m.group(1)) if m.group(1) else default_pause_s
        if seg_text:
            segments.append([seg_text, dur])
        elif segments:
            # ယခင် segment ပြီးတာနဲ့ text အသစ်မလာခင် marker ထပ်တွေ့ရင် (ဥပမာ "[pause][pause:1]")
            # ယခင် segment ရဲ့ pause ကို stack လုပ် (ပေါင်း) ပေးမည်
            segments[-1][1] += dur
        last_end = m.end()
    tail = text[last_end:].strip()
    if tail:
        segments.append([tail, 0.0])
    return [(t, d) for t, d in segments] if segments else [(text.strip(), 0.0)]


def generate_with_pauses(model, text, desc_prefix, base_gen_kwargs, sample_rate, default_pause_s, seed=None):
    """text ထဲမှာ [pause]/[pause:N] marker ပါရင် segment အလိုက် သီးခြား generate လုပ်ပြီး
    marker နေရာမှာ တကယ့် silence gap ထည့်ကာ ပေါင်းစည်းပေးသည်။ marker မပါရင် တစ်ကြိမ်တည်း
    (ရှိပြီးသား behaviour အတိုင်း) generate လုပ်ပါမည် — base_gen_kwargs ထဲ 'text' မပါရပါ။
    seed ပါလာရင် model.generate() ခေါ်တိုင်း (segment တစ်ခုချင်းစီအပါအဝင်) reseed
    လုပ်ပေးသည် — voice ရောစပ်မသွားစေရန်။"""
    if "[pause" not in text.lower():
        gen_text = f"{desc_prefix}{text}" if desc_prefix else text
        maybe_reseed(seed)
        return np.asarray(model.generate(text=gen_text, **base_gen_kwargs), dtype=np.float32)

    segments = split_on_pauses(text, default_pause_s)
    chunks = []
    for seg_text, pause_after in segments:
        if not seg_text:
            continue
        gen_text = f"{desc_prefix}{seg_text}" if desc_prefix else seg_text
        maybe_reseed(seed)
        chunk = np.asarray(model.generate(text=gen_text, **base_gen_kwargs), dtype=np.float32)
        chunks.append(chunk)
        if pause_after > 0:
            chunks.append(np.zeros(int(sample_rate * pause_after), dtype=np.float32))

    if not chunks:
        gen_text = f"{desc_prefix}{text}" if desc_prefix else text
        maybe_reseed(seed)
        return np.asarray(model.generate(text=gen_text, **base_gen_kwargs), dtype=np.float32)

    return np.concatenate(chunks)


def apply_speed(wav, rate):
    """rate=1.0 => မပြောင်းလဲပါ။ rate<1.0 => ပိုနှေး, rate>1.0 => ပိုမြန် — pitch (အသံအနုတ်/အမြင့်)
    ကို မထိခိုက်စေဘဲ speed ကိုသာ ချိန်ညှိပေးသည် (librosa phase-vocoder time-stretch)။
    librosa မရှိရင် (install မလုပ်ရသေးရင်) ဘာမှ မပြောင်းလဲဘဲ audio ကို အတိုင်းပြန်ပေးမည်။"""
    if not rate or abs(rate - 1.0) < 1e-3:
        return wav
    if not _LIBROSA_AVAILABLE:
        log("[voxcpm] 'speed' requested but librosa is not installed — skipping speed adjustment "
            "(pip install librosa သို့မဟုတ် requirements.txt ထဲ ထည့်ပေးပါ).")
        return wav
    try:
        rate = max(MIN_SPEED, min(MAX_SPEED, float(rate)))
        return librosa.effects.time_stretch(np.asarray(wav, dtype=np.float32), rate=rate)
    except Exception:
        log("[voxcpm] speed time-stretch failed, returning original-speed audio. Traceback below:")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return wav


def generate_multi_speaker(model, segments, cfg_value, inference_timesteps, sample_rate, max_len=None,
                            default_pause_s=DEFAULT_PAUSE_SECONDS, seed=None):
    """Segment တစ်ခုချင်းစီကို voice description နှင့်အတူ သီးခြား generate လုပ်ပြီး
    တစ်ခုတည်းအဖြစ် ပေါင်းစည်းပေးသည် (segment တစ်ခုချင်းကြား ခဏနားချိန် ခြားသတ်ပြီး)။
    line တစ်ကြောင်းချင်းစီထဲမှာ [pause] marker ပါရင်လည်း အလုပ်လုပ်ပါသည်။ seed ပါလာရင်
    speaker (M/F/C) တစ်ခုချင်းစီအတွက် အသံ တသမတ်တည်း ဖြစ်စေရန် ထပ်ဆင့် reseed လုပ်ပေးမည်။"""
    silence_gap = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
    base_gen_kwargs = dict(cfg_value=cfg_value, inference_timesteps=inference_timesteps)
    if max_len:
        base_gen_kwargs["max_len"] = max_len
    chunks = []
    for voice_key, line_text in segments:
        if not line_text:
            continue
        desc = VOICE_DESCRIPTIONS.get(voice_key)
        desc_prefix = f"({desc}) " if desc else ""
        chunk = generate_with_pauses(model, line_text, desc_prefix, base_gen_kwargs, sample_rate, default_pause_s,
                                      seed=seed)
        chunks.append(chunk)
        chunks.append(silence_gap)

    if not chunks:
        raise ValueError("No valid dialogue lines found.")

    return np.concatenate(chunks[:-1]) if len(chunks) > 1 else chunks[0]


def maybe_reseed(seed):
    """input.seed ပေးထားရင် torch (နှင့် cuda ရှိရင် cuda) ကို reseed လုပ်ပေးသည်။
    VoxCPM2 ဟာ reference audio မပါတဲ့ "Voice Design" mode (voice_type ချည်းသုံးတဲ့
    Text-To-Speech) မှာ speaker embedding ကို torch ရဲ့ global RNG ကနေ stochastic
    အနေနဲ့ sample ယူတတ်ပါသည် — reseed မလုပ်ရင် model.generate() ကို independent
    ခေါ်တိုင်း (RunPod job အသီးသီးဖြစ်ဖြစ်၊ [pause]/multi-voice segment တစ်ခုချင်းစီ
    ဖြစ်ဖြစ်) voice အသစ်ပြန်ရွေးတတ်လို့ text ရှည်ပြီး chunk ခွဲထားရင် voice ရောနေတတ်ပါသည်
    (caller (worker-9.js) က chunk အားလုံးကို seed တစ်ခုတည်းမျှဝေပေးမှသာ ဒီ function
    က အလုပ်ဖြစ်မည် — seed မပို့ရင် ယခင်အတိုင်း random behaviour အတိုင်း ဆက်ကျန်ပါမည်)။"""
    if seed is None:
        return
    try:
        seed = int(seed) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg):
    # -u (unbuffered) is set in CMD, but flush explicitly anyway so nothing
    # gets lost if the process exits abruptly right after printing.
    print(msg, flush=True)


log("[voxcpm] Starting worker, importing voxcpm ...")
try:
    from voxcpm import VoxCPM
except Exception:
    log("[voxcpm] FATAL: failed to import voxcpm. Full traceback below:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    # Re-raise so the crash (and this traceback) is unambiguous in logs
    # instead of silently limping on with model=None.
    raise

# Model Loading
load_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID
log(f"[voxcpm] Loading model from: {load_path}")

cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
log(f"[voxcpm] torch.cuda.is_available()={cuda_available} -> using device='{device}'")
if not cuda_available:
    log(
        "[voxcpm] WARNING: CUDA GPU not detected — running on CPU. "
        "Generation will be dramatically slower (often 10x+) than on GPU. "
        "Check that this RunPod endpoint actually has a GPU attached and that "
        "the container's CUDA/driver versions are compatible."
    )
else:
    log(f"[voxcpm] GPU detected: {torch.cuda.get_device_name(0)}")

log(f"[voxcpm] torch.compile optimize={VOXCPM_OPTIMIZE} "
    f"({'expect ~80-90s warmup on this cold start, faster steady-state' if VOXCPM_OPTIMIZE else 'no compile warmup, slightly slower steady-state'})")

model = None
try:
    model = VoxCPM.from_pretrained(load_path, load_denoiser=False, device=device, optimize=VOXCPM_OPTIMIZE)
    log("[voxcpm] Model loaded successfully, worker is ready.")
except Exception:
    log("[voxcpm] Critical error loading model. Full traceback below:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    # Keep model=None so the handler can report a clean error per-request
    # instead of crashing the whole worker on every request.


def handler(event):
    if model is None:
        return {"error": "Model failed to load on worker startup. Check container logs."}

    inp = event.get("input", {}) or {}

    if WORKER_SECRET and inp.get("secret") != WORKER_SECRET:
        return {"error": "unauthorized: missing or wrong 'secret'."}

    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "input.text is required."}
    if len(text) > MAX_CHARS:
        return {"error": f"input.text exceeds {MAX_CHARS} characters limit."}

    cfg_value = float(inp.get("cfg_value", 2.0))
    inference_timesteps = int(inp.get("inference_timesteps", 10))

    kwargs = dict(text=text, cfg_value=cfg_value, inference_timesteps=inference_timesteps, max_len=VOXCPM_MAX_LEN)

    # Base64 Reference Audio Handling
    ref_b64 = inp.get("reference_audio_base64")
    if ref_b64:
        ref_b64 = ref_b64.strip()
        # Browser FileReader.readAsDataURL() စသည်တို့က "data:audio/wav;base64,...."
        # ပုံစံ Full Data URI ပို့တတ်ပါသည် — Prefix ပါလာရင် Auto ဖယ်ပေးမည်
        if ref_b64.lower().startswith("data:") and "," in ref_b64:
            ref_b64 = ref_b64.split(",", 1)[1]

        try:
            ref_bytes = base64.b64decode(ref_b64, validate=False)
        except Exception as e:
            return {"error": f"reference_audio_base64 ကို base64 decode မလုပ်နိုင်ပါ: {e}"}

        # Audio File အစစ်တစ်ခုက ပုံမှန် ထောင်ချီ bytes ရှိတတ်ပါသည် — သေးငယ်လွန်းရင်
        # Audio File အစစ် မဟုတ်နိုင်ကြောင်း ချက်ချင်း ရှင်းလင်းသော Error ပြန်ပေးမည်
        # (Model ထဲ ဝင်ပြီးမှ Cryptic Error နဲ့ Fail မဖြစ်စေရန်)
        if len(ref_bytes) < 1000:
            return {
                "error": (
                    "reference_audio_base64 ဟာ audio file အစစ်လို့ မထင်ရပါ "
                    f"(decode ပြီးရင် {len(ref_bytes)} bytes သာ ရှိပါသည်). "
                    "data:...;base64, Prefix မပါဘဲ raw base64 audio content တစ်ခုတည်း "
                    "ပို့ထားကြောင်း ပြန်စစ်ဆေးပါ။"
                )
            }

        try:
            tmp_dir = tempfile.mkdtemp()
            tmp_ref_path = os.path.join(tmp_dir, "reference.wav")
            with open(tmp_ref_path, "wb") as f:
                f.write(ref_bytes)
            kwargs["reference_wav_path"] = tmp_ref_path

            prompt_text = inp.get("prompt_text")
            if prompt_text:
                kwargs["prompt_wav_path"] = tmp_ref_path
                kwargs["prompt_text"] = prompt_text.strip()
        except Exception as e:
            return {"error": f"Failed to process reference audio: {e}"}

    voice_type = inp.get("voice_type")

    # Caller (worker-9.js) က request တစ်ခုလုံးအတွက် seed တစ်ခုတည်းကို parallel
    # chunk များအားလုံးကို မျှဝေပေးထားပါက ဒီနေရာမှာ လက်ခံသည် — reference audio
    # မပါတဲ့ Text-To-Speech (voice_type ချည်း) mode မှာ chunk တစ်ခုချင်းစီအတွက်
    # voice တစ်မျိုးစီ ရောစပ်မသွားစေရန်။ မပါလာရင် ယခင်အတိုင်း random ဆက်ဖြစ်နေမည်။
    seed = inp.get("seed")

    # Voice Type (female/male) — only meaningful without reference audio to clone from.
    # desc_prefix ကို သီးခြား သိမ်းထားသည် — [pause] marker ပါလာရင် segment တစ်ခုချင်းစီအတွက်
    # ပြန်သုံးရန် လိုအပ်လို့ဖြစ်ပြီး (voice-design mode မှာ segment တစ်ခုစီက generate() ကို
    # သီးခြား ခေါ်ရမှာမို့ desc ကို ထပ်ခါထပ်ခါ ထည့်ပေးဖို့ လိုအပ်ပါသည်)၊ marker မပါရင်တော့
    # ယခင်အတိုင်းပဲ kwargs['text'] ထဲ တစ်ခါတည်း ထည့်ပြီး behaviour မပြောင်းအောင် ထားထားသည်
    desc_prefix = ""
    if "reference_wav_path" not in kwargs:
        desc = VOICE_DESCRIPTIONS.get(voice_type) if voice_type in ("female", "male") else None
        if desc:
            desc_prefix = f"({desc}) "
            kwargs["text"] = f"{desc_prefix}{kwargs['text']}"

    # AI ရဲ့ output speed ကို ထိန်းချုပ်ချင်တဲ့ တောင်းဆိုချက်အတွက် — input.speed မပါလာရင်
    # DEFAULT_SPEED (default အားဖြင့် အနည်းငယ်နှေးအောင် 0.93) ကို အလိုအလျောက် သုံးမည်
    speed = float(inp.get("speed", DEFAULT_SPEED))
    pause_seconds = float(inp.get("pause_seconds", DEFAULT_PAUSE_SECONDS))

    try:
        sample_rate = model.tts_model.sample_rate

        # Multi Voice (Dialogue): reference audio မပါ၊ "M:"/"F:"/"C:" tag များ ပါဝင်ရင်
        # line အလိုက် အသံအမျိုးမျိုး (male/female/child) ခွဲပြီး တစ်ခါတည်း ပေါင်းစပ်ပေးမည်
        multi_segments = None
        if voice_type == "multi" and "reference_wav_path" not in kwargs:
            parsed = parse_multi_speaker_lines(text)
            if any(voice for voice, _ in parsed):
                multi_segments = parsed

        if multi_segments:
            wav = generate_multi_speaker(model, multi_segments, cfg_value, inference_timesteps, sample_rate,
                                          max_len=VOXCPM_MAX_LEN, default_pause_s=pause_seconds, seed=seed)
        elif "[pause" in text.lower():
            # Audio book စတဲ့ script ရှည်ရှည်တွေမှာ "[pause]" (သို့) "[pause:1.2]" ဆိုတဲ့ marker
            # ထည့်ပြီး ရပ်ချင်တဲ့နေရာမှာ ရပ်နားနိုင်အောင် ကူညီပေးသည် — segment အလိုက် သီးခြား
            # generate လုပ်ပြီး marker နေရာတွေမှာ တကယ့် silence gap ထည့်ကာ ပေါင်းစည်းပေးမည်
            base_gen_kwargs = {k: v for k, v in kwargs.items() if k != "text"}
            wav = generate_with_pauses(model, text, desc_prefix, base_gen_kwargs, sample_rate, pause_seconds,
                                        seed=seed)
        else:
            maybe_reseed(seed)
            wav = model.generate(**kwargs)

        # Speech ကို "AI ပီပီ တစ်ဆက်တည်းမြန်မြန်" မဟုတ်ဘဲ ထိန်းချုပ်ထားနိုင်အောင် — pitch ကို
        # မထိခိုက်စေဘဲ speed ကိုသာ ချိန်ညှိပေးသည် (rate=1.0 ဆိုရင် ဘာမှ မပြောင်းလဲပါ)
        wav = apply_speed(wav, speed)

        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "output.wav")
        sf.write(out_path, wav, sample_rate)

        with open(out_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "audio_base64": out_b64,
            "sample_rate": sample_rate,
            "format": "wav",
        }
    except Exception:
        log("[voxcpm] Generation failed. Full traceback below:")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return {"error": "Voice generation failed. Check container logs for the full traceback."}


runpod.serverless.start({"handler": handler})
