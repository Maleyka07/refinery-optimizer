"""
Neft Emalı Optimallaşdırma Sistemi
Modul  : chatbot.py
Məqsəd : MeloSense — proses analitik mühərriki
TTS    : OpenAI tts-1-hd (nova)
"""

import os
import re
import time
import tempfile
from datetime import datetime
from typing import Generator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# TTS konfiqurasiyası — yalnız OpenAI
TTS_MODEL = "tts-1-hd"
TTS_VOICE = "nova"
TTS_SPEED = 1.0

# Uyğunluq üçün köhnə ad
TTS_ELEVENLABS_VOICE = TTS_VOICE


# ── Tələffüz normallaşdırıcısı ────────────────────────────────────────────

_BIRLER = ["", "bir", "iki", "üç", "dörd", "beş", "altı", "yeddi", "səkkiz", "doqquz"]
_ONLAR  = ["", "on", "iyirmi", "otuz", "qırx", "əlli", "altmış", "yetmiş", "səksən", "doxsan"]
_YUZLER = ["", "yüz", "iki yüz", "üç yüz", "dörd yüz", "beş yüz",
           "altı yüz", "yeddi yüz", "səkkiz yüz", "doqquz yüz"]

def _tam_sozle(n: int) -> str:
    if n == 0: return "sıfır"
    if n < 0:  return "mənfi " + _tam_sozle(-n)
    hisseler = []
    if n >= 1_000_000:
        hisseler.append(_tam_sozle(n // 1_000_000) + " milyon")
        n %= 1_000_000
    if n >= 1000:
        prefix = "" if (n // 1000) == 1 else _tam_sozle(n // 1000) + " "
        hisseler.append(prefix + "min")
        n %= 1000
    if n >= 100:
        hisseler.append(_YUZLER[n // 100])
        n %= 100
    if n >= 10:
        hisseler.append(_ONLAR[n // 10])
        n %= 10
    if n > 0:
        hisseler.append(_BIRLER[n])
    return " ".join(h for h in hisseler if h)

def _reqem_sozle(s: str) -> str:
    s = s.strip().replace(",", ".")
    menfi = s.startswith("-")
    if menfi:
        s = s[1:]
    if "." in s:
        tam_h, kesir_h = s.split(".", 1)
        tam_soz = _tam_sozle(int(tam_h)) if tam_h and tam_h != "0" else "sıfır"
        kesir_sozler = " ".join(_BIRLER[int(c)] for c in kesir_h if c.isdigit() and _BIRLER[int(c)])
        if not kesir_sozler:
            kesir_sozler = "sıfır"
        netice = f"{tam_soz} nöqtə {kesir_sozler}"
    else:
        netice = _tam_sozle(int(s))
    return ("mənfi " if menfi else "") + netice

_VAHİD = {
    "g/cm³": "qram santimetr kub", "g/sm³": "qram santimetr kub",
    "m³/h":  "kubmetr saat",       "m³/s":  "kubmetr saniyə",
    "GJ/h":  "giqacoul saat",      "GS/s":  "giqacoul saniyə",
    "MPa":   "meqapaskal",         "kPa":   "kilopaskal",
    "m/s":   "metr saniyə",        "atm":   "atmosfer",
    "bar":   "bar",                "°C":    "dərəcə Selsi",
    "°c":    "dərəcə Selsi",       "%":     "faiz",
    "°":     "dərəcə",
}

_QISALTMA = {
    "NSGA-II": "en es ci a ikinci", "NSGA": "en es ci a",
    "KKT":     "ka ka te",          "IQR":  "i kyu ar",
    "ARO":     "a ar o",            "R²":   "ar kvadrat",
    "F*":      "ef ulduz",          "H₂":   "hidrogen",
    "CO₂":     "karbon dioksid",    "α":    "alfa",
    "β":       "beta",              "γ":    "qamma",
    "δ":       "delta",
}

_HERF_TELEFFUZ = {
    "T": "temperatur", "P": "təzyiq", "F": "axın",
    "R": "refluks",    "E": "enerji", "S": "kükürd",
}

def _tts_metni_hazirla(metn: str, max_simvol: int = 4000) -> str:
    xetler = []
    for xet in metn.split("\n"):
        xet = xet.strip()
        if not xet or xet.startswith("|") or set(xet) <= set("|-:= "):
            continue
        xet = re.sub(r'^#{1,6}\s+', '', xet)
        xet = re.sub(r'^[-*•]\s+', '', xet)
        xetler.append(xet)
    metn = " ".join(xetler)

    metn = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', metn)
    metn = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', metn)
    metn = re.sub(r'`{1,3}[^`]*`{1,3}', '', metn)
    metn = re.sub(r'https?://\S+', '', metn)

    for qisa, aciq in sorted(_QISALTMA.items(), key=lambda x: -len(x[0])):
        metn = metn.replace(qisa, aciq)

    for herf, sozl in _HERF_TELEFFUZ.items():
        metn = re.sub(rf'\b{re.escape(herf)}\b(?=\s*[=\d])', sozl, metn)

    vahidler  = sorted(_VAHİD.keys(), key=len, reverse=True)
    v_pattern = "|".join(re.escape(v) for v in vahidler)

    def _reqem_vahid(m):
        isare = m.group(1) or ""
        eded  = m.group(2)
        vahid = m.group(3) or ""
        try:
            soz = _reqem_sozle(eded)
        except Exception:
            soz = eded
        if isare == "+":
            soz = "artı " + soz
        elif isare == "-":
            soz = "mənfi " + soz
        vahid_soz = _VAHİD.get(vahid, "")
        return (soz + (" " + vahid_soz if vahid_soz else "")).strip()

    metn = re.sub(rf'([+\-]?)(\d+(?:[.,]\d+)?)\s*({v_pattern})', _reqem_vahid, metn)

    def _tek_reqem(m):
        try:
            return _reqem_sozle(m.group(0))
        except Exception:
            return m.group(0)

    metn = re.sub(
        r'(?<![a-zA-ZəçşğüöıƏÇŞĞÜÖI])\d+(?:[.,]\d+)?(?![a-zA-ZəçşğüöıƏÇŞĞÜÖI])',
        _tek_reqem, metn,
    )

    metn = re.sub(r'\s*—\s*', ', ', metn)
    metn = re.sub(r'\s*–\s*', ', ', metn)
    metn = re.sub(r'\(([^)]+)\)', r'\1', metn)
    metn = metn.replace(';', ',')
    metn = re.sub(r':\s*', ', ', metn)
    metn = re.sub(r',\s*,+', ',', metn)

    AZ = 'çşğüöıəÇŞĞÜÖIƏ'
    metn = re.sub(rf'[^a-zA-Z{AZ}\u0400-\u04FF\s\.,!?]', ' ', metn, flags=re.UNICODE)

    metn = re.sub(r'\s{2,}', ' ', metn).strip()
    if len(metn) > max_simvol:
        metn = metn[:max_simvol].rsplit('.', 1)[0] + "."

    return metn


def _metni_temizle(metn: str, max_simvol: int = 4000) -> str:
    return _tts_metni_hazirla(metn, max_simvol)


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sən MeloSense adlı AI köməkçisisin — Neft Emalı Optimallaşdırma Sisteminin proses analitik mühərriki. ADNSU SABAH fakültəsinin buraxılış işi çərçivəsində hazırlanmışdır.

Sistem modulları:
- data_generator.py — 9 sensor (T, P, F, H₂...), 24 saatlıq sintetik time-series, sinusoidal drift + anomaliya injeksiyası
- preprocessor.py — filtrasiya → Min-Maks normallaşdırma → IQR + Isolation Forest → xətti reqressiya (Y=a₁T+a₂P+a₃F+ε, E=0.042T+0.018F+0.008Tf−11.5)
- optimizer.py — NSGA-II, KKT şərtləri, Pareto frontu, 100 fərd × 150 nəsil
- app.py — 5 səhifəli Streamlit dashboard

Sənəd tənlikləri:
- (1) F* = α·Y_benz + β·Y_diz + γ·Y_ker − δ·(E/E_max) + ε·S_removal — məqsəd funksiyası
- (2) x_norm = (x−x_min)/(x_max−x_min) — normallaşdırma
- (3) ΣF_giriş = ΣF_çıxış — kütlə balansı
- (4) Q_verilən = Q_istifadə + Q_itki — enerji balansı
- (5) T(z) = T_alt − k·z — temperatur profili
- (6-8) g(x)≤0, x_min≤x≤x_max — KKT məhdudiyyətləri
- Y = a₁T + a₂P + a₃F + a₄ + ε — xətti reqressiya modeli
- E = 0.042·T + 0.018·F + 0.008·Tf − 11.5 — enerji balansı (feed_temp daxil)
- S_removal = 0.55·((H2−30)/30) + 0.30·((Tc−280)/80) + 0.40 — kükürd çıxarılması

Fiziki parametr həddləri:
Soba: 340–400°C | Kolon: 1.2–1.5 atm | Axın: 80–160 m³/s
H₂: 30–60 bar | Katalizator: 280–360°C | Sıxlıq: 0.820–0.920 q/sm³

Dil və üslub qaydaları:
- Konkret rəqəm və parametr aralıqları ver — ümumi ifadələrdən çəkin
- Hər cavabda səbəb → nəticə məntiqi olsun
- Paragraflar qısa olsun — 2-3 cümlə, sonra format dəyiş
- Azərbaycan dilinin orfoqrafik qaydalarına ciddi riayət et
- Cavabda dırnaq işarəsi işlətmə
- Emoji istifadə etmə — yalnız analitik məlumat ver
"""


# ── Kontekst builder ─────────────────────────────────────────────────────

def kontekst_yarat(
    xulase: dict,
    opt_netice=None,
    req_goster=None,
    aktiv_sehife: str = "İcmal",
    çəkilər: dict | None = None,
) -> str:
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")

    # İqtisadi çəkilər
    w = çəkilər or {}
    alpha   = w.get("alpha",   0.35)
    beta    = w.get("beta",    0.30)
    gamma   = w.get("gamma",   0.20)
    delta   = w.get("delta",   0.15)
    epsilon = w.get("epsilon", 0.10)
    E_MAX   = 22.0

    # Cari F* hesabla
    yb  = xulase.get("avg_yield_gasoline", 0) / 100
    yd  = xulase.get("avg_yield_diesel",   0) / 100
    yk  = xulase.get("avg_yield_kerosene", 0) / 100
    e   = xulase.get("avg_energy",         0)
    sr  = xulase.get("avg_sulfur_removal", 0) / 100
    f_cari = alpha*yb + beta*yd + gamma*yk - delta*(e/E_MAX) + epsilon*sr

    ctx = f"""Canlı Sistem Vəziyyəti ({zaman}) | Aktiv: {aktiv_sehife}

Məqsəd funksiyası:
F* = {alpha}·Yb + {beta}·Yd + {gamma}·Yk − {delta}·(E/{E_MAX}) + {epsilon}·S_removal
E_max = {E_MAX} GJ/s

Cari Sensor KPI-ları:
Soba temperaturu: {xulase.get('avg_furnace_temp',0):.1f} °C
Benzin verimi (Yb): {xulase.get('avg_yield_gasoline',0):.2f}%
Dizel verimi (Yd): {xulase.get('avg_yield_diesel',0):.2f}%
Kerosin verimi (Yk): {xulase.get('avg_yield_kerosene',0):.2f}%
Ümumi verim: {xulase.get('avg_total_yield',0):.2f}%
Enerji sərfi (E): {xulase.get('avg_energy',0):.3f} GJ/s
Kükürd çıxarılması (S_removal): {xulase.get('avg_sulfur_removal',0):.2f}%
Nümunə sayı: {xulase.get('n_samples',1440)}
Anomaliya: {xulase.get('n_anomalies',0)} ədəd ({xulase.get('anomaly_rate_pct',0):.1f}%)
Cari F* balı: {f_cari:.4f}

KKT məhdudiyyətləri (cari):
g1 — Ümumi verim ≤ 85%: {xulase.get('avg_total_yield',0):.2f}% → {'Ödənilir' if xulase.get('avg_total_yield',0)<=85 else 'POZULUR'}
g4 — Kükürd çıxarılması ≥ 50%: {xulase.get('avg_sulfur_removal',0):.2f}% → {'Ödənilir' if xulase.get('avg_sulfur_removal',0)>=50 else 'POZULUR'}
"""
    if opt_netice is not None:
        ey  = opt_netice.best_compromise
        dv  = ey.get("total_yield_pct", 0) - xulase.get("avg_total_yield", 0)
        de  = ey.get("energy_gj_h", 0) - xulase.get("avg_energy", 0)
        il  = abs(de) * 8760
        e_izah = "artım (yüksək verim üçün tradeoff)" if de > 0 else "azalma (qənaət)"
        ctx += f"""
NSGA-II Optimal Həll (Pareto kompromis nöqtəsi):
Optimal soba temp: {ey.get('furnace_temp',0):.1f} °C
Optimal axın sürəti: {ey.get('flow_rate',0):.1f} m³/s
Optimal refluks nisbəti: {ey.get('reflux_ratio',0):.2f}
Optimal qidalanma temp: {ey.get('feed_temp',0):.1f} °C
Optimal H₂ təzyiqi: {ey.get('h2_pressure',0):.1f} bar
Optimal katalizator temp: {ey.get('catalyst_temp',0):.1f} °C
Optimal benzin verimi: {ey.get('yield_gasoline',0):.2f}%
Optimal dizel verimi: {ey.get('yield_diesel',0):.2f}%
Optimal kerosin verimi: {ey.get('yield_kerosene',0):.2f}%
Optimal ümumi verim: {ey.get('total_yield_pct',0):.2f}%
Optimal enerji sərfi: {ey.get('energy_gj_h',0):.3f} GJ/s
Optimal kükürd çıxarılması: {ey.get('sulfur_removal_pct',0):.2f}%
Optimal F* balı: {ey.get('f_star',0):.4f}
Pareto həll sayı: {opt_netice.n_solutions}
Verim fərqi: Cari {xulase.get('avg_total_yield',0):.2f}% → Optimal {ey.get('total_yield_pct',0):.2f}% (Δ {dv:+.2f}%)
Enerji dəyişimi: Cari {xulase.get('avg_energy',0):.3f} → Optimal {ey.get('energy_gj_h',0):.3f} GJ/s (Δ {de:+.3f}, {e_izah})
İllik enerji fərqi: {il:,.0f} GJ/il
"""
    return ctx.strip()


# ── Chatbot sinifi ────────────────────────────────────────────────────────

class NeftEmalChatbot:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-your"):
            raise ValueError("OPENAI_API_KEY tapılmadı")
        self.client = OpenAI(api_key=api_key)
        self.model  = "gpt-4o"
        self.son_latency: float = 0.0

    def _danisiq_metni_yarat(self, texniki_cavab: str) -> str:
        """
        Texniki cavabı TTS üçün danışıq dilinə çevirir.
        Qısa, axıcı, rəqəmlər sözlə yazılmış mətn qaytarır.
        """
        prompt = (
            "Sən neft emalı sahəsindəki mütəxəssissən. "
            "Aşağıdakı texniki analizi oxu və bir həmkarına telefonda "
            "izah edirsən kimi sadə, axıcı Azərbaycan dilində danış.\n\n"
            "Qaydalar:\n"
            "- Maksimum 4-5 cümlə. Artıq heç nə əlavə etmə.\n"
            "- Heç bir cədvəl, siyahı, işarə — yalnız düz cümlələr.\n"
            "- Rəqəmləri sözlə de: üç yüz altmış iki dərəcə, yeddi nöqtə altı faiz.\n"
            "- Cümlələrin başlanğıcı müxtəlif olsun.\n"
            "- Sanki bir insanın öz fikrini danışdığı kimi səslənsin.\n\n"
            f"Texniki məlumat:\n{texniki_cavab[:2000]}\n\n"
            "İndi bunu canlı danışıq kimi yaz — yalnız düz mətn, sonda nöqtə:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return _metni_temizle(texniki_cavab, max_simvol=600)

    def ses_yarat(
        self,
        metn: str,
        ses: str = TTS_VOICE,
        surət: float = TTS_SPEED,
    ) -> tuple[bytes | None, str]:
        """
        OpenAI TTS ilə audio yaradır.
        Returns: (audio_bytes, "openai") və ya (None, "error")
        """
        danisiq     = self._danisiq_metni_yarat(metn)
        if not danisiq or len(danisiq) < 5:
            return None, "error"
        danisiq_tts = _tts_metni_hazirla(danisiq)

        # ses parametri OpenAI voices ilə uyğun gəlməsə default istifadə et
        _openai_voices = {"nova", "shimmer", "alloy", "onyx", "echo", "fable", "ash", "coral", "sage"}
        tts_ses = ses if ses in _openai_voices else TTS_VOICE

        try:
            cavab = self.client.audio.speech.create(
                model=TTS_MODEL,
                voice=tts_ses,
                input=danisiq_tts,
                speed=surət,
                response_format="mp3",
            )
            return cavab.content, "openai"
        except Exception as ex:
            print(f"[OpenAI TTS xəta]: {ex}")
            return None, "error"

    def ses_yarat_dosya(
        self,
        metn: str,
        ses: str = TTS_VOICE,
        surət: float = TTS_SPEED,
    ) -> tuple[str | None, str]:
        audio_bytes, motor = self.ses_yarat(metn, ses=ses, surət=surət)
        if audio_bytes is None:
            return None, motor
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix="neft_tts_") as tmp:
                tmp.write(audio_bytes)
                return tmp.name, motor
        except Exception as ex:
            print(f"[TTS fayl xəta]: {ex}")
            return None, "error"

    def el_limit_goster(self) -> dict:
        """Uyğunluq üçün saxlanılıb — ElevenLabs artıq istifadə edilmir."""
        return {"aktiv": False}

    def cavab_ver_stream(
        self,
        mesajlar: list[dict],
        kontekst: str,
    ) -> Generator[str, None, None]:
        tam_system   = SYSTEM_PROMPT + f"\n\n{kontekst}"
        api_mesajlar = [{"role": "system", "content": tam_system}] + mesajlar

        bashlama = time.time()
        try:
            axin = self.client.chat.completions.create(
                model=self.model,
                messages=api_mesajlar,
                max_tokens=1200,
                temperature=0.4,
                stream=True,
            )
            tam = ""
            for chunk in axin:
                if (chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content):
                    tam += chunk.choices[0].delta.content

            tam = re.sub(r"```[\w]*\n?", "", tam)
            tam = re.sub(r"```", "", tam)
            tam = re.sub(r"`", "", tam)
            tam = tam.replace('"', '')
            tam = re.sub(r"\s{2,}", " ", tam).strip()
            yield tam

        except Exception as ex:
            err_msg = str(ex)
            if "api_key" in err_msg.lower() or "authentication" in err_msg.lower():
                yield "API açarı yanlışdır. .env faylındakı OPENAI_API_KEY-i yoxlayın."
            elif "rate_limit" in err_msg.lower():
                yield "Sorğu limiti aşıldı. Bir neçə saniyə gözləyib yenidən cəhd edin."
            elif "network" in err_msg.lower() or "connection" in err_msg.lower():
                yield "Şəbəkə problemi. İnternet bağlantınızı yoxlayın."
            else:
                yield "Cavab alınarkən problem yarandı. Yenidən cəhd edin."

        self.son_latency = round(time.time() - bashlama, 2)

    def tez_analiz(self, kontekst: str, nov: str) -> Generator[str, None, None]:
        sorğular = {
            "pareto": (
                "Yuxarıdakı sistem vəziyyətinə bax. "
                "NSGA-II Pareto frontunu şərh et: optimal F* balı, enerji-verim kompromisi, "
                "ən yaxşı həllin soba temperaturu və kükürd çıxarılması dəyərləri nədir? "
                "Cari vəziyyətlə fərqi rəqəmlərlə göstər."
            ),
            "anomaliya": (
                "Yuxarıdakı sistem vəziyyətinə bax. "
                "Aşkar edilmiş anomaliyaların sayı və faizini götür. "
                "Bu faiz sənaye norması olan 1.5–3.0% ilə müqayisədə necədir? "
                "Hansı sensor kanallarında anomaliya ehtimalı yüksəkdir və bunun texniki səbəbi nədir?"
            ),
            "tövsiyə": (
                "Yuxarıdakı sistem vəziyyətinə bax. "
                "Cari soba temperaturu, enerji sərfi, kükürd çıxarılması və verim dəyərlərini əsas götür. "
                "Bu dəyərlərə əsasən 3 konkret operator tövsiyəsi ver: "
                "hər biri üçün hansı parametri, hansı istiqamətdə dəyişdirmək lazımdır və "
                "gözlənilən nəticəni rəqəmlə göstər."
            ),
            "tənlik": (
                "Yuxarıdakı sistem vəziyyətinə bax. "
                "F* = α·Yb + β·Yd + γ·Yk − δ·(E/E_max) + ε·S_removal tənliyini "
                "cari benzin, dizel, kerosin verimi, enerji və kükürd çıxarılması dəyərləri ilə hesabla. "
                "KKT şərtlərindən g1 (verim ≤ 0.85) və g4 (kükürd ≥ 50%) yerinə yetirilirmi?"
            ),
            "müqayisə": (
                "Yuxarıdakı sistem vəziyyətinə bax. "
                "Cari ümumi verim, enerji sərfi və F* balını optimal həllin dəyərləri ilə müqayisə et. "
                "Hər göstərici üçün fərqi faiz və mütləq dəyər kimi göstər. "
                "İllik enerji qənaətini GJ/il kimi hesabla."
            ),
        }
        sual = sorğular.get(nov, "Yuxarıdakı sistem vəziyyətini qiymətləndir.")
        yield from self.cavab_ver_stream([{"role": "user", "content": sual}], kontekst)


def chatbot_yarat() -> "NeftEmalChatbot | None":
    try:
        return NeftEmalChatbot()
    except ValueError:
        return None
