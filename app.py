import os, re, math, json, warnings, tempfile, subprocess, stat

# Install static ffmpeg binary
def _install_ffmpeg():
    ffmpeg_path = "/tmp/ffmpeg"
    if not os.path.exists(ffmpeg_path):
        subprocess.run([
            "curl", "-L",
            "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
            "-o", "/tmp/ffmpeg.tar.xz"
        ], check=True)
        subprocess.run(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"], check=True)
        import glob
        bins = glob.glob("/tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg")
        if bins:
            import shutil
            shutil.copy(bins[0], ffmpeg_path)
            os.chmod(ffmpeg_path, os.stat(ffmpeg_path).st_mode | stat.S_IEXEC)
            # Also copy ffprobe
            probes = glob.glob("/tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe")
            if probes:
                shutil.copy(probes[0], "/tmp/ffprobe")
                os.chmod("/tmp/ffprobe", os.stat("/tmp/ffprobe").st_mode | stat.S_IEXEC)
        os.environ["PATH"] = "/tmp:" + os.environ.get("PATH", "")

_install_ffmpeg()

subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)

from datetime import datetime
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import librosa
import cv2
import mediapipe as mp
import textstat
from pydub import AudioSegment
from pydub.silence import detect_silence
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)

# ================================================================
# SCRIPT COACH ROUTE
# ================================================================
SCRIPT_SYSTEM = """You are VOYCE, an expert public speaking coach and speechwriter.
Respond with ONLY a valid JSON object. No preamble, no markdown.
{
  "mode": "full",
  "feedback_summary": "2-4 sentence overall assessment",
  "inline_suggestions": [{"original": "phrase", "suggestion": "improved", "reason": "why"}],
  "rewritten_script": "full rewrite",
  "delivery_tips": ["tip 1"],
  "generic_tips": [],
  "estimated_duration": "X-Y minutes or null"
}
Rules: inline_suggestions 3-7 edits, delivery_tips specific to content, generic_tips only if nothing provided."""


@app.route("/coach/analyze", methods=["POST"])
def coach_analyze():
    data = request.json
    topic = data.get("topic", "")
    setting = data.get("setting", "")
    audience = data.get("audience", "")
    time_limit = data.get("timeLimit", "")
    script = data.get("script", "")

    has_script = bool(script and script.strip())
    has_context = any([topic, setting, audience, time_limit])

    if not has_script and not has_context:
        user_prompt = 'Return mode="generic" with 6-8 universal tips. All other fields empty.'
    else:
        mode = "full" if has_script else "no_script"
        ctx = ""
        if topic: ctx += "\nTopic: " + topic
        if setting: ctx += "\nSetting: " + setting
        if audience: ctx += "\nAudience: " + audience
        if time_limit: ctx += "\nDuration: " + time_limit + " min"
        script_block = "\n\nScript:\n" + script.strip() if has_script else "\n\nNo script provided."
        instruction = "Give inline suggestions, full rewrite, delivery tips." if has_script else "Give delivery tips only. Leave rewritten_script and inline_suggestions empty."
        user_prompt = "Analyze this speech.\n" + ctx + script_block + "\n\n" + instruction + "\nSet mode=\"" + mode + "\"."

    try:
        from together import Together
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            max_tokens=4096,
            messages=[{
                "role": "system",
                "content": SCRIPT_SYSTEM
            }, {
                "role": "user",
                "content": user_prompt
            }])
        raw = response.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result = json.loads(raw[start:end])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# AUDIO EXTRACTION
# ================================================================
def extract_audio(src):
    out = src + "_audio.wav"
    try:
        y, sr = librosa.load(src, sr=16000, mono=True)
        import soundfile as sf
        sf.write(out, y, 16000)
        return out
    except Exception as e:
        raise ValueError("Audio extraction error: " + str(e))

# ================================================================
# VOICE ANALYZER
# ================================================================
class VoiceAnalyzer:
    _FILLER = [
        r'\bum+\b', r'\buh+\b', r'\ber+\b', r'\bah+\b', r'\blike\b',
        r'\byou know\b', r'\bbasically\b', r'\bliterally\b', r'\bactually\b',
        r'\bso\b', r'\bwell\b', r'\bkind of\b', r'\bsort of\b'
    ]
    _STOP = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "as", "by", "from", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "not",
        "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
        "we", "they", "my", "your", "his", "her", "our", "their"
    }

    def __init__(self, wav):
        self.wav = wav
        self.y, self.sr = librosa.load(wav, sr=None)
        self.dur = librosa.get_duration(y=self.y, sr=self.sr)
        self.seg = AudioSegment.from_file(wav)
        self.transcript = ""
        self.results = {}

    def transcribe(self):
        from faster_whisper import WhisperModel
        mdl = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = mdl.transcribe(self.wav)
        self.transcript = " ".join(s.text for s in segments).strip()
        return self

    def _speaking_rate(self):
        words = self.transcript.split()
        wpm = len(words) / (self.dur / 60) if self.dur else 0
        sc, st = (100, "excellent") if 120 <= wpm <= 150 else (
            80, "good") if 110 <= wpm <= 160 else (
                65, "fair") if 100 <= wpm <= 170 else (45, "needs-work")
        self.results["speaking_rate"] = dict(
            score=sc, status=st, value=str(round(wpm, 1)) + " WPM",
            raw_wpm=round(wpm, 1), word_count=len(words),
            feedback="Optimal pace." if sc == 100 else "Slightly outside optimal range." if sc >= 65 else "Pace too fast or slow.")

    def _pause_control(self):
        sils = detect_silence(self.seg, min_silence_len=400, silence_thresh=-35)
        tot = sum((e - s) / 1000 for s, e in sils)
        ratio = tot / self.dur if self.dur else 0
        sc, st = (100, "excellent") if 0.10 <= ratio <= 0.25 else (
            80, "good") if 0.05 <= ratio <= 0.35 else (55, "needs-work")
        self.results["pause_control"] = dict(
            score=sc, status=st, value=str(round(ratio * 100)) + "% silence",
            pause_count=len(sils), silence_ratio=round(ratio, 3),
            effective_speaking_s=round(self.dur - tot, 1),
            feedback="Well-balanced pausing." if sc == 100 else "Acceptable pausing." if sc >= 80 else "Too many or too few pauses.")

    def _filler_words(self):
        txt = self.transcript.lower()
        bd = {}
        total = 0
        for p in self._FILLER:
            c = len(re.findall(p, txt))
            if c:
                bd[p.strip(r'\b')] = c
                total += c
        rate = total / (self.dur / 60) if self.dur else 0
        sc, st = (100, "excellent") if rate <= 1 else (
            75, "good") if rate <= 3 else (55, "fair") if rate <= 5 else (35, "needs-work")
        self.results["filler_words"] = dict(
            score=sc, status=st, value=str(round(rate, 1)) + "/min",
            total=total, rate=round(rate, 2), breakdown=bd,
            feedback="Very few fillers." if sc == 100 else str(round(rate, 1)) + "/min. Practice silent pausing." if sc <= 55 else str(round(rate, 1)) + "/min. Keep working on it.")

    def _volume_energy(self):
        rms = librosa.feature.rms(y=self.y, frame_length=1024)[0]
        db = 20 * np.log10(rms + 1e-10)
        mdb = float(np.mean(db))
        sdb = float(np.std(db))
        sc, st = (100, "excellent") if 5 <= sdb <= 12 else (
            80, "good") if 3 <= sdb <= 15 else (
                55, "needs-work") if sdb < 3 else (65, "fair")
        self.results["volume_energy"] = dict(
            score=sc, status=st,
            value=str(min(100, round((max(0, ((mdb + 40) / 30) * 100) + min(100, sdb * 15)) / 2))) + "% energy",
            mean_db=round(mdb, 1), std_db=round(sdb, 1),
            feedback="Strong vocal energy." if sc == 100 else "Good energy." if sc >= 80 else "Low energy. Project more.")

    def _pitch_intonation(self):
        pitches, mags = librosa.piptrack(y=self.y, sr=self.sr, fmin=50, fmax=450)
        vals = [pitches[mags[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[mags[:, t].argmax(), t] > 0]
        if not vals:
            self.results["pitch_intonation"] = dict(score=50, status="fair", value="N/A", feedback="Could not extract pitch.")
            return
        arr = np.array(vals)
        mean = float(np.mean(arr))
        cv = float(np.std(arr)) / mean if mean > 0 else 0
        sc, st = (100, "excellent") if 0.05 <= cv <= 0.15 else (
            80, "good") if 0.02 <= cv <= 0.20 else (55, "needs-work")
        self.results["pitch_intonation"] = dict(
            score=sc, status=st, value=str(round(mean)) + " Hz avg",
            mean_hz=round(mean, 1), cv=round(cv, 3),
            feedback="Excellent pitch variation." if sc == 100 else "Good intonation." if sc >= 80 else "Pitch too flat or erratic.")

    def _speech_fluency(self):
        words = self.transcript.lower().split()
        reps = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1] and len(words[i]) > 2)
        hes = sum(len(re.findall(p, self.transcript.lower())) for p in [r'\buh+\b', r'\bum+\b', r'\ber+\b', r'\bah+\b'])
        rate = (reps + hes) / (self.dur / 60) if self.dur else 0
        sc, st = (100, "excellent") if rate <= 1 else (75, "good") if rate <= 3 else (50, "needs-work")
        self.results["speech_fluency"] = dict(
            score=sc, status=st, value=str(round(rate, 1)) + " breaks/min",
            repetitions=reps, hesitations=hes,
            feedback="Minimal hesitations." if sc == 100 else "Minor hesitations." if sc >= 75 else "Frequent hesitations.")

    def _voice_stability(self):
        pitches, mags = librosa.piptrack(y=self.y, sr=self.sr, fmin=50, fmax=450)
        vals = [pitches[mags[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[mags[:, t].argmax(), t] > 0]
        if len(vals) < 2:
            self.results["voice_stability"] = dict(score=50, status="fair", value="N/A", feedback="Insufficient data.")
            return
        arr = np.array(vals)
        jitter = float(np.std(np.diff(arr)) / np.mean(arr)) if np.mean(arr) > 0 else 0
        shimmer = float(np.std(np.diff(20 * np.log10(librosa.feature.rms(y=self.y, frame_length=1024)[0] + 1e-10))))
        stab = (jitter * 100) + (shimmer / 10)
        sc, st = (100, "excellent") if stab <= 2 else (75, "good") if stab <= 5 else (50, "needs-work")
        self.results["voice_stability"] = dict(
            score=sc, status=st, value=str(round(stab, 2)) + " index",
            feedback="Excellent stability." if sc == 100 else "Moderate stability." if sc >= 75 else "Instability detected.")

    def _vocabulary(self):
        txt = re.sub(r'[^\w\s]', '', self.transcript.lower()).split()
        cont = [w for w in txt if w not in self._STOP and w.isalpha()]
        if not cont:
            self.results["vocabulary"] = dict(score=50, status="fair", value="N/A", feedback="No transcript.")
            return
        uniq = len(set(cont))
        total = len(cont)
        div = uniq / total if total else 0
        avgL = float(np.mean([len(w) for w in cont]))
        lv, sc, st = ("advanced", 100, "excellent") if div > 0.7 and avgL > 5 else (
            "intermediate", 80, "good") if div > 0.5 and avgL > 4 else ("basic", 55, "needs-work")
        self.results["vocabulary"] = dict(
            score=sc, status=st, value=str(uniq) + " unique words",
            unique_words=uniq, total_words=total, diversity=round(div, 3), level=lv,
            feedback="Excellent vocabulary." if sc == 100 else "Good vocabulary." if sc >= 80 else "Basic vocabulary.")

    def _voice_energy(self):
        ve = self.results.get("volume_energy", {})
        pi = self.results.get("pitch_intonation", {})
        sr = self.results.get("speaking_rate", {})
        ov = round((max(0, min(100, ((ve.get("mean_db", -30) + 40) / 30) * 100)) +
                    min(100, ve.get("std_db", 5) * 15) +
                    min(100, pi.get("cv", 0.1) * 300) +
                    min(100, max(0, (sr.get("raw_wpm", 130) - 80) / 100 * 100))) / 4, 1)
        sc, st = (100, "excellent") if ov >= 70 else (80, "good") if ov >= 50 else (55, "needs-work")
        self.results["voice_energy"] = dict(
            score=sc, status=st, value=str(round(ov)) + "%", overall=ov,
            feedback="Great voice energy." if sc == 100 else "Good energy." if sc >= 80 else "Low energy.")

    def analyze(self):
        self.transcribe()
        for fn in [self._speaking_rate, self._pause_control, self._filler_words,
                   self._volume_energy, self._pitch_intonation, self._speech_fluency,
                   self._voice_stability, self._vocabulary, self._voice_energy]:
            fn()
        return self

    def overall_score(self):
        W = dict(speaking_rate=.15, pause_control=.10, filler_words=.15,
                 volume_energy=.10, pitch_intonation=.10, speech_fluency=.10,
                 voice_stability=.10, vocabulary=.10, voice_energy=.10)
        return round(sum(self.results[k]["score"] * w for k, w in W.items() if k in self.results), 1)


# ================================================================
# VIDEO ANALYZER
# ================================================================
class VideoAnalyzer:

    def __init__(self, path):
        self.path = path
        self.results = {}
        mf = mp.solutions.face_mesh
        mp_ = mp.solutions.pose
        mh = mp.solutions.hands
        self.face = mf.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=.5,
                                min_tracking_confidence=.5)
        self.pose = mp_.Pose(static_image_mode=False, model_complexity=1,
                             smooth_landmarks=True, min_detection_confidence=.5,
                             min_tracking_confidence=.5)
        self.hands = mh.Hands(static_image_mode=False, max_num_hands=2,
                              min_detection_confidence=.5, min_tracking_confidence=.5)

    @staticmethod
    def _d(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def analyze(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError("Cannot open: " + self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        samp = max(1, int(fps / 5))
        eye_f, smile_s, post_s, head_pos, hand_pos = [], [], [], [], []
        emo_cnt = Counter()
        proc = 0
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            fi += 1
            if fi % samp != 0: continue
            proc += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            fr = self.face.process(rgb)
            pr = self.pose.process(rgb)
            if fr.multi_face_landmarks:
                lm = fr.multi_face_landmarks[0].landmark
                li = [33, 7, 163, 144, 145, 153, 154, 155, 133]
                ri = [362, 382, 381, 380, 374, 373, 390, 249, 263]
                cx = (np.mean([lm[i].x * w for i in li]) + np.mean([lm[i].x * w for i in ri])) / 2
                cy = (np.mean([lm[i].y * h for i in li]) + np.mean([lm[i].y * h for i in ri])) / 2
                eye_f.append((abs(cx - w / 2) / (w / 2) < .30) and (abs(cy - h / 2) / (h / 2) < .40))
                ml, mr, mt, mb = lm[61], lm[291], lm[13], lm[14]
                mw = abs(mr.x - ml.x) * w
                mht = abs(mt.y - mb.y) * h + 1e-6
                cyy = ((mt.y + mb.y) / 2) * h
                up = cyy - ((ml.y + mr.y) / 2) * h
                sm = float(np.clip(up / 10, 0, 1) * np.clip((mw / mht) / 15, 0, 1))
                smile_s.append(sm)
                emo_cnt["happy" if sm > .4 else "neutral-positive" if sm > .2 else "neutral"] += 1
                head_pos.append((lm[1].x * w, lm[1].y * h))
            else:
                eye_f.append(False)
                smile_s.append(0.0)
            if pr.pose_landmarks:
                lm2 = pr.pose_landmarks.landmark
                ls, rs, nose = lm2[11], lm2[12], lm2[0]
                post_s.append(max(0, 100 - (abs(ls.y - rs.y) * h * 4) - (abs(nose.x - (ls.x + rs.x) / 2) * w * .5)))
                hand_pos.append([(lm2[15].x * w, lm2[15].y * h), (lm2[16].x * w, lm2[16].y * h)])
        cap.release()
        n = len(eye_f) or 1
        ec = sum(eye_f) / n * 100
        ecs, est = (100, "excellent") if ec >= 60 else (75, "good") if ec >= 40 else (55, "fair") if ec >= 25 else (35, "needs-work")
        self.results["eye_contact"] = dict(score=ecs, status=est, value=str(round(ec)) + "% of time", pct=round(ec, 1),
            feedback="Excellent eye contact." if ecs == 100 else "Good eye contact." if ecs >= 75 else "Low eye contact.")
        sp = (float(np.mean(smile_s)) if smile_s else 0) * 100
        dom = emo_cnt.most_common(1)[0][0] if emo_cnt else "neutral"
        fes, fst = (100, "excellent") if sp >= 30 else (80, "good") if sp >= 15 else (60, "fair") if sp >= 5 else (40, "needs-work")
        self.results["facial_expression"] = dict(score=fes, status=fst, value=dom.replace("-", " ").title(), smile_pct=round(sp, 1),
            feedback="Warm, expressive engagement." if fes == 100 else "Good expression." if fes >= 80 else "Limited expressiveness.")
        ap = float(np.mean(post_s)) if post_s else 50
        pos_s, pos_st = (100, "excellent") if ap >= 80 else (80, "good") if ap >= 65 else (60, "fair") if ap >= 50 else (40, "needs-work")
        self.results["posture"] = dict(score=pos_s, status=pos_st, value=str(round(ap)) + "/100",
            feedback="Great posture." if pos_s == 100 else "Good posture." if pos_s >= 80 else "Posture needs work.")
        gf = sum(1 for hp in hand_pos if hp)
        gp = gf / proc * 100 if proc else 0
        hgs, hgst = (100, "excellent") if 30 <= gp <= 70 else (75, "good") if 15 <= gp < 30 else (65, "fair") if gp > 70 else (45, "needs-work")
        self.results["hand_gestures"] = dict(score=hgs, status=hgst, value=str(round(gp)) + "% visible",
            feedback="Purposeful gesturing." if hgs == 100 else "Good gestures." if hgs >= 75 else "Use more open gestures.")
        hm = [self._d(head_pos[i], head_pos[i - 1]) for i in range(1, len(head_pos)) if head_pos[i] and head_pos[i - 1]]
        am = float(np.mean(hm)) if hm else 0
        mvs, mvst = (100, "excellent") if 2 <= am <= 15 else (65, "fair") if am < 2 else (75, "good") if 15 < am <= 30 else (45, "needs-work")
        self.results["movement"] = dict(score=mvs, status=mvst, value=str(round(am, 1)) + "px/frame",
            feedback="Natural movement." if mvs == 100 else "Slightly static." if mvs == 65 else "Good movement." if mvs >= 75 else "Excessive movement.")
        return self

    def overall_score(self):
        W = dict(eye_contact=.25, facial_expression=.20, posture=.20, hand_gestures=.15, movement=.20)
        return round(sum(self.results[k]["score"] * w for k, w in W.items() if k in self.results), 1)


# ================================================================
# AI SCORING
# ================================================================
def run_ai_scoring(voice_res, video_res, transcript, api_key):
    if not api_key: return None
    from together import Together
    client = Together(api_key=api_key)
    vr = voice_res
    vid = video_res
    prompt = (
        "Score this speaker 1-5 on each criterion. Return ONLY JSON.\n"
        "Voice: WPM=" + str(vr.get('speaking_rate', {}).get('raw_wpm', 'N/A')) +
        ", Fillers=" + str(vr.get('filler_words', {}).get('rate', 'N/A')) + "/min"
        ", Energy=" + str(vr.get('volume_energy', {}).get('score', 'N/A')) + "/100\n"
        "Video: EyeContact=" + str(vid.get('eye_contact', {}).get('pct', 'N/A')) + "%"
        ", Posture=" + str(vid.get('posture', {}).get('score', 'N/A')) + "/100\n"
        "Transcript: " + transcript[:800] + "\n\n"
        '{"scores":{"content_organization":3,"delivery_vocal_quality":3,"body_language_eye_contact":3,"audience_engagement":3,"language_clarity":3,"total_score":15,"interpretation":"Competent speaker","feedback_summary":"summary"},"strengths":["s1"],"weaknesses":["w1"],"suggestions":["s1"]}'
    )
    try:
        res = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}])
        raw = res.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return None


# ================================================================
# REPORT GENERATOR
# ================================================================
def generate_report(voice_res, video_res, voice_score, video_score, ai_res,
                    transcript, duration, scenario, is_video):
    date = datetime.now().strftime("%b %d, %Y")

    if is_video and ai_res:
        combined = round(voice_score * .40 + video_score * .35 + (ai_res["scores"]["total_score"] * 4) * .25, 1)
    elif is_video:
        combined = round(voice_score * .55 + video_score * .45, 1)
    elif ai_res:
        combined = round(voice_score * .60 + (ai_res["scores"]["total_score"] * 4) * .40, 1)
    else:
        combined = round(voice_score, 1)

    def grade(s):
        for t, g in [(93, "A"), (90, "A-"), (87, "B+"), (83, "B"), (80, "B-"), (77, "C+"), (73, "C"), (70, "C-")]:
            if s >= t: return g
        return "D"

    def perf(s):
        if s >= 90: return ("Excellent Performance", "#4ade80")
        if s >= 80: return ("Strong Delivery", "#86efac")
        if s >= 70: return ("Good — Keep Practicing", "#fbbf24")
        if s >= 60: return ("Developing Skills", "#fb923c")
        return ("Needs Improvement", "#f87171")

    COL = {"excellent": "#4ade80", "good": "#fbbf24", "fair": "#fb923c", "needs-work": "#f87171"}

    def col(st):
        return COL.get(st, "#fb923c")

    def card(key, d, icon):
        c = col(d["status"])
        return (
            '<div class="metric-card">'
            '<div class="metric-top">'
            '<div class="metric-name"><span>' + icon + '</span>' + key.replace("_", " ").title() + '</div>'
            '<div class="metric-score" style="color:' + c + '">' + str(d["score"]) + '%</div>'
            '</div>'
            '<div class="bar-bg"><div class="bar-fill" style="width:' + str(d["score"]) + '%;background:linear-gradient(90deg,' + c + '88,' + c + ')"></div></div>'
            '<div style="font-size:11px;color:' + c + ';margin-bottom:6px">' + d["value"] + '</div>'
            '<div class="metric-feedback">' + d["feedback"] + '</div>'
            '</div>'
        )

    v_icons = dict(speaking_rate="🗣️", pause_control="⏸️", filler_words="🚫",
                   volume_energy="📢", pitch_intonation="🎵", speech_fluency="😊",
                   voice_stability="💎", vocabulary="📚", voice_energy="⚡")
    vid_icons = dict(eye_contact="👁️", facial_expression="😄", posture="🧍",
                     hand_gestures="🙌", movement="🔄")

    v_cards = "".join(card(k, voice_res[k], v_icons[k]) for k in v_icons if k in voice_res)

    if is_video:
        vid_cards = "".join(card(k, video_res[k], vid_icons[k]) for k in vid_icons if k in video_res)
    else:
        vid_cards = '<div class="empty-state">Audio only — video metrics not available.</div>'

    ai_section = ""
    if ai_res:
        sc = ai_res["scores"]
        ai100 = sc["total_score"] * 4
        ac = "#4ade80" if ai100 >= 80 else "#fbbf24" if ai100 >= 60 else "#f87171"

        def rcolor(v):
            return "#4ade80" if v >= 4 else "#fbbf24" if v >= 3 else "#f87171"

        rubric_items = [
            ("content_organization", "Content & Organisation"),
            ("delivery_vocal_quality", "Delivery & Vocal Quality"),
            ("body_language_eye_contact", "Body Language & Eye Contact"),
            ("audience_engagement", "Audience Engagement"),
            ("language_clarity", "Language & Clarity"),
        ]
        rubric = "".join(
            '<div class="comp-row">'
            '<div class="comp-label">' + lbl + '</div>'
            '<div class="comp-bar-bg"><div class="comp-bar-fill" style="width:' + str(sc[k] / 5 * 100) + '%;background:' + rcolor(sc[k]) + '"></div></div>'
            '<div class="comp-score" style="color:' + rcolor(sc[k]) + '">' + str(sc[k]) + '/5</div>'
            '</div>'
            for k, lbl in rubric_items
        )

        strengths = "".join("<li>" + s + "</li>" for s in ai_res.get("strengths", []))
        weaknesses = "".join("<li>" + s + "</li>" for s in ai_res.get("weaknesses", []))
        suggestions = "".join("<li>" + s + "</li>" for s in ai_res.get("suggestions", []))

        ai_section = (
            '<div class="section-head"><h3>AI Coach</h3><div class="badge">LLaMA 3.3</div></div>'
            '<div class="ai-hero">'
            '<div style="text-align:center;flex-shrink:0">'
            '<div style="font-size:42px;font-weight:900;color:' + ac + '">' + str(sc["total_score"]) + '<span style="font-size:18px;color:var(--muted)">/25</span></div>'
            '<div style="font-size:11px;color:var(--muted);margin-top:4px;text-transform:uppercase">' + sc["interpretation"] + '</div>'
            '</div>'
            '<div><p style="font-size:13px;color:var(--muted);line-height:1.7">' + sc["feedback_summary"] + '</p></div>'
            '</div>'
            '<div class="comp-grid">' + rubric + '</div>'
            '<div class="swl-grid">'
            '<div class="swl-box"><div class="swl-title" style="color:#4ade80">&#10003; Strengths</div><ul class="swl-list">' + strengths + '</ul></div>'
            '<div class="swl-box"><div class="swl-title" style="color:#f87171">&#10007; Weaknesses</div><ul class="swl-list">' + weaknesses + '</ul></div>'
            '<div class="swl-box"><div class="swl-title" style="color:#fbbf24">&#8594; Suggestions</div><ul class="swl-list">' + suggestions + '</ul></div>'
            '</div>'
        )

    all_m = dict(**voice_res)
    if is_video:
        all_m.update(video_res)

    def fb_class(status):
        if status in ("excellent", "good"): return "good"
        if status == "needs-work": return "warn"
        return ""

    def fb_icon(status):
        if status in ("excellent", "good"): return "&#10003;"
        if status == "needs-work": return "&#9888;"
        return "&#8594;"

    feedback = "".join(
        '<div class="feedback-item ' + fb_class(d["status"]) + '">'
        '<div class="feedback-title">' + fb_icon(d["status"]) + ' ' + k.replace("_", " ").title() + ' — ' + d["status"].replace("-", " ").title() + '</div>'
        '<div class="feedback-text">' + d["feedback"] + '</div>'
        '</div>'
        for k, d in sorted(all_m.items(), key=lambda x: x[1]["score"])
    )

    sr = voice_res.get("speaking_rate", {})
    fw = voice_res.get("filler_words", {})
    vc = voice_res.get("vocabulary", {})
    pc = voice_res.get("pause_control", {})
    ec = video_res.get("eye_contact", {}) if is_video else {}
    g = grade(combined)
    pf, pc2 = perf(combined)
    dur_str = str(int(duration // 60)) + ":" + str(int(duration % 60)).zfill(2)

    video_sub_score = (
        '<div class="sub-score"><div class="sv">' + str(round(video_score)) + '</div><div class="sl">Video</div></div>'
        if is_video else ""
    )
    ai_sub_score = (
        '<div class="sub-score"><div class="sv">' + str(ai_res["scores"]["total_score"]) + '<span style="font-size:11px;color:var(--muted)">/25</span></div><div class="sl">AI</div></div>'
        if ai_res else ""
    )
    footer_scores = "Voice " + str(round(voice_score))
    if is_video:
        footer_scores += " &middot; Video " + str(round(video_score))
    if ai_res:
        footer_scores += " &middot; AI " + str(round(ai_res["scores"]["total_score"] * 4))

    return '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>VOYCE Report</title>
<style>*{margin:0;padding:0;box-sizing:border-box}:root{--teal:#4a90b8;--teal-light:#7bb3d3;--teal-bright:#5bc4e0;--bg:#0d1117;--bg2:#111820;--bg3:#162030;--text:#f0f4f8;--muted:#8ba5b8;--border:rgba(123,179,211,0.15)}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);padding:40px 24px}.wrap{max-width:1200px;margin:0 auto;padding:0 40px}.report-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:48px;padding-bottom:24px;border-bottom:1px solid var(--border)}.logo{font-size:22px;font-weight:900;letter-spacing:3px;color:var(--teal-light)}.report-meta .label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}.report-meta .val{font-size:13px;font-weight:600;color:var(--text);margin-top:2px;text-align:right}.score-hero{display:grid;grid-template-columns:auto 1fr;gap:40px;align-items:center;background:linear-gradient(135deg,rgba(74,144,184,0.1),rgba(91,196,224,0.04));border:1px solid rgba(123,179,211,0.3);border-radius:20px;padding:36px;margin-bottom:32px}.grade-ring{width:110px;height:110px;border-radius:50%;background:linear-gradient(135deg,var(--teal),var(--teal-light));display:flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:0 0 40px rgba(74,144,184,0.25);flex-shrink:0}.grade-letter{font-size:38px;font-weight:900;color:#fff;line-height:1}.grade-sub{font-size:11px;color:rgba(255,255,255,0.8);margin-top:2px}.score-details h2{font-size:26px;font-weight:800;margin-bottom:6px;background:linear-gradient(135deg,var(--teal-light),var(--teal-bright));-webkit-background-clip:text;-webkit-text-fill-color:transparent}.score-details p{color:var(--muted);font-size:14px;line-height:1.6;margin-bottom:16px}.perf-badge{display:inline-block;border-radius:20px;padding:5px 14px;font-size:12px;font-weight:700;border:1px solid}.sub-scores{display:flex;gap:28px;margin-top:18px;flex-wrap:wrap}.sub-score .sv{font-size:22px;font-weight:800;color:var(--teal-light)}.sub-score .sl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px}.stats-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:32px}.stat-box{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:18px 12px;text-align:center}.stat-val{font-size:20px;font-weight:800;color:var(--teal-light);margin-bottom:4px}.stat-lbl{font-size:10px;color:var(--muted);line-height:1.3}.section-head{display:flex;align-items:center;gap:10px;margin-bottom:20px;margin-top:36px}.section-head h3{font-size:18px;font-weight:800;background:linear-gradient(135deg,var(--teal-light),var(--teal-bright));-webkit-background-clip:text;-webkit-text-fill-color:transparent}.section-head .badge{background:rgba(74,144,184,0.1);border:1px solid rgba(74,144,184,0.25);border-radius:20px;padding:4px 12px;font-size:11px;color:var(--teal-bright);font-weight:700}.metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:32px}.metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:20px;position:relative;overflow:hidden}.metric-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--teal),var(--teal-bright))}.metric-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}.metric-name{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:700}.metric-score{font-size:20px;font-weight:900}.bar-bg{height:5px;background:rgba(255,255,255,0.06);border-radius:3px;margin-bottom:8px;overflow:hidden}.bar-fill{height:100%;border-radius:3px}.metric-feedback{font-size:12px;color:var(--muted);line-height:1.5}.comp-grid{display:flex;flex-direction:column;gap:8px;margin-bottom:32px}.comp-row{display:flex;align-items:center;gap:12px}.comp-label{font-size:12px;color:var(--muted);width:180px;flex-shrink:0}.comp-bar-bg{flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden}.comp-bar-fill{height:100%;border-radius:3px}.comp-score{font-size:12px;font-weight:700;width:40px;text-align:right}.ai-hero{display:flex;gap:28px;align-items:flex-start;background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:24px}.swl-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:32px}.swl-box{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:18px}.swl-title{font-size:13px;font-weight:800;margin-bottom:10px}.swl-list{list-style:none;display:flex;flex-direction:column;gap:7px}.swl-list li{font-size:12px;color:var(--muted);line-height:1.5;padding-left:8px;border-left:2px solid rgba(255,255,255,0.08)}.feedback-item{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--teal);border-radius:0 12px 12px 0;padding:16px 18px;margin-bottom:10px}.feedback-title{font-size:13px;font-weight:700;color:var(--teal-light);margin-bottom:5px}.feedback-text{font-size:13px;color:var(--muted);line-height:1.6}.feedback-item.warn{border-left-color:#f87171}.feedback-item.warn .feedback-title{color:#f87171}.feedback-item.good{border-left-color:#4ade80}.feedback-item.good .feedback-title{color:#4ade80}.transcript-box{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:20px;font-size:13px;color:var(--muted);line-height:1.8;max-height:200px;overflow-y:auto}.empty-state{background:var(--bg2);border:1px dashed var(--border);border-radius:12px;padding:24px;text-align:center;font-size:13px;color:var(--muted)}.report-footer{margin-top:48px;padding-top:24px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}.report-footer p{font-size:11px;color:var(--muted)}@media(max-width:600px){.metrics-grid,.swl-grid{grid-template-columns:1fr}.stats-row{grid-template-columns:repeat(3,1fr)}.score-hero{grid-template-columns:1fr}}</style>
</head><body><div class="wrap">
<div class="report-header"><div class="logo">VOYCE</div><div class="report-meta"><div class="label">Performance Report</div><div class="val">''' + scenario + " &middot; " + date + '''</div></div></div>
<div class="score-hero">
  <div class="grade-ring"><div class="grade-letter">''' + g + '''</div><div class="grade-sub">''' + str(round(combined)) + '''/100</div></div>
  <div class="score-details">
    <h2>''' + pf + '''</h2>
    <p>Combined: 9 voice metrics''' + (", 5 video metrics" if is_video else "") + (", AI scoring" if ai_res else "") + '''.</p>
    <span class="perf-badge" style="color:''' + pc2 + ";border-color:" + pc2 + "44;background:" + pc2 + '''11">''' + pf + '''</span>
    <div class="sub-scores">
      <div class="sub-score"><div class="sv">''' + str(round(voice_score)) + '''</div><div class="sl">Voice</div></div>
      ''' + video_sub_score + ai_sub_score + '''
      <div class="sub-score"><div class="sv">''' + dur_str + '''</div><div class="sl">Duration</div></div>
    </div>
  </div>
</div>
<div class="stats-row">
  <div class="stat-box"><div class="stat-val">''' + str(sr.get("raw_wpm", "—")) + '''</div><div class="stat-lbl">Words/Min</div></div>
  <div class="stat-box"><div class="stat-val">''' + str(sr.get("word_count", "—")) + '''</div><div class="stat-lbl">Total Words</div></div>
  <div class="stat-box"><div class="stat-val">''' + str(fw.get("rate", "—")) + '''</div><div class="stat-lbl">Fillers/Min</div></div>
  <div class="stat-box"><div class="stat-val">''' + str(vc.get("unique_words", "—")) + '''</div><div class="stat-lbl">Unique Words</div></div>
  <div class="stat-box"><div class="stat-val">''' + str(pc.get("effective_speaking_s", "—")) + '''s</div><div class="stat-lbl">Speaking Time</div></div>
  <div class="stat-box"><div class="stat-val">''' + str(ec.get("pct", "—")) + '''%</div><div class="stat-lbl">Eye Contact</div></div>
</div>
<div class="section-head"><h3>Voice Analysis</h3><div class="badge">9 metrics</div></div>
<div class="metrics-grid">''' + v_cards + '''</div>
<div class="section-head"><h3>Video Analysis</h3><div class="badge">5 metrics</div></div>
<div class="metrics-grid">''' + vid_cards + '''</div>
''' + ai_section + '''
<div class="section-head"><h3>Feedback</h3><div class="badge">Ranked by priority</div></div>
<div class="feedback-grid">''' + feedback + '''</div>
<div class="section-head"><h3>Transcript</h3></div>
<div class="transcript-box">''' + (transcript or "No transcript available.") + '''</div>
<div class="report-footer">
  <div class="logo" style="font-size:14px">VOYCE</div>
  <p>''' + footer_scores + '''</p>
  <p style="color:var(--teal-light);font-weight:600">voyce.app</p>
</div>
</div></body></html>'''


# ================================================================
# ANALYZE ROUTE
# ================================================================
@app.route("/analyzer/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    scenario = request.form.get("scenario", "Practice Session")
    ext = os.path.splitext(f.filename)[1].lower()

    tmp_path = None
    wav = None
    mp4_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        wav = extract_audio(tmp_path)
        va = VoiceAnalyzer(wav)
        va.analyze()
        vs = va.overall_score()

        is_video = False
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of",
                 "default=noprint_wrappers=1:nokey=1", tmp_path],
                capture_output=True)
            is_video = bool(probe.stdout.decode().strip())

        vid_res = {}
        vids = 0
        if is_video:
            if ext == ".webm":
                mp4_path = tmp_path + "_converted.mp4"
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264",
                     "-preset", "fast", "-c:a", "aac", mp4_path],
                    capture_output=True)
                if r.returncode != 0:
                    raise ValueError("ffmpeg transcode error: " + r.stderr.decode())
                video_input = mp4_path
            else:
                video_input = tmp_path
            vid = VideoAnalyzer(video_input)
            vid.analyze()
            vid_res = vid.results
            vids = vid.overall_score()
        else:
            for k in ["eye_contact", "facial_expression", "posture", "hand_gestures", "movement"]:
                vid_res[k] = dict(score=0, status="fair", value="N/A (audio only)",
                                  feedback="Video analysis requires a video file.")

        ai_res = run_ai_scoring(va.results, vid_res, va.transcript, os.getenv("TOGETHER_API_KEY", ""))
        html = generate_report(va.results, vid_res, vs, vids, ai_res, va.transcript, va.dur, scenario, is_video)

        if is_video and ai_res:
            combined = round(vs * .40 + vids * .35 + (ai_res["scores"]["total_score"] * 4) * .25, 1)
        elif is_video:
            combined = round(vs * .55 + vids * .45, 1)
        elif ai_res:
            combined = round(vs * .60 + (ai_res["scores"]["total_score"] * 4) * .40, 1)
        else:
            combined = round(vs, 1)

        return jsonify({"html": html, "voice_score": vs, "video_score": vids, "combined": combined})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [tmp_path, wav, mp4_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
