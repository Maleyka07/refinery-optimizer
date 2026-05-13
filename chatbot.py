"""
Neft Emalı Optimallaşdırma Sistemi
=====================================
Modul  : chatbot.py
Məqsəd : MeloSense — Proses İntelligensiyası analitik mühərriki
Version: OpenAI SDK 2.x uyğun
"""

import os
import re
import time
import tempfile
from datetime import datetime
from typing import Generator
from openai import OpenAI
from dotenv import load_dotenv

# ElevenLabs SDK — pip install elevenlabs
try:
    from elevenlabs.client import ElevenLabs as ElevenLabsClient
    from elevenlabs import VoiceSettings
    _ELEVENLABS_AVAILABLE = True
except ImportError:
    _ELEVENLABS_AVAILABLE = False

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════
# TTS KONFİQURASİYASI — ElevenLabs (2026 üçün ən yüksək keyfiyyət)
# ══════════════════════════════════════════════════════════════════════════
# ElevenLabs eleven_multilingual_v2:
#   - Azərbaycan dilinə native dəstək (aze)
#   - Ən yüksək insan səsinə yaxın keyfiyyət (2026 liderboard #2)
#   - Robotik deyil, natural intonasiya, düzgün rəqəm tələffüzü
#
# Tövsiyə olunan səslər (ElevenLabs Voice Library-dən):
#   "Charlotte"  — Nadia Rashidova  — professional, sakit, aydın qadın səsi
#   "George"     — dərin, mü权威li kişi səsi
#   "Callum"     — neytral, analitik
#
# Fallback: OpenAI tts-1-hd (ELEVENLABS_API_KEY tapılmasa)

TTS_ELEVENLABS_MODEL  = "eleven_multilingual_v2"   # ← ən keyfiyyətli multilingual model
TTS_ELEVENLABS_VOICE  = "Charlotte"                # ← professional, insan kimi qadın səsi
TTS_FORMAT            = "mp3"

# OpenAI TTS fallback (köhnə konfiqurasiya)
TTS_MODEL_FALLBACK = "tts-1-hd"
TTS_VOICE_FALLBACK = "nova"
TTS_SPEED          = 1.0

# ══════════════════════════════════════════════════════════════════════════
# TƏLƏFFÜZ NORMALLAŞDIRICISİ — TTS üçün tam hazırlıq pipeline-ı
# ══════════════════════════════════════════════════════════════════════════
# Prinsip: TTS-ə gedən mətn elə yazılmalıdır ki, model onu oxuduqda
# dinləyici EYNƏN eşitsin. Heç bir simvol, qısaltma, rəqəm qalmamalıdır.

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
    # Ondalıqlı: hər rəqəm ayrıca oxunur — "sıfır nöqtə səkkiz beş"
    # Bu TTS üçün ən sabit yoldur, tam ədəd kimi oxuma qarışıqlıq yaradır
    s = s.strip().replace(",", ".")
    menfi = s.startswith("-")
    if menfi:
        s = s[1:]
    if "." in s:
        tam_h, kesir_h = s.split(".", 1)
        tam_soz = _tam_sozle(int(tam_h)) if tam_h and tam_h != "0" else "sıfır"
        # Kəsir hissəsi: hər rəqəm ayrıca — "0.85" → "sıfır nöqtə səkkiz beş"
        kesir_sozler = " ".join(_BIRLER[int(c)] for c in kesir_h if c.isdigit() and _BIRLER[int(c)])
        if not kesir_sozler:
            kesir_sozler = "sıfır"
        netice = f"{tam_soz} nöqtə {kesir_sozler}"
    else:
        netice = _tam_sozle(int(s))
    return ("mənfi " if menfi else "") + netice

# Vahid → tələffüz
_VAHİD = {
    "g/cm³":  "qram santimetr kub",
    "g/sm³":  "qram santimetr kub",
    "m³/h":   "kubmetr saat",
    "m³/s":   "kubmetr saniyə",
    "GJ/h":   "giqacoul saat",
    "GS/s":   "giqacoul saniyə",
    "km/h":   "kilometr saat",
    "MPa":    "meqapaskal",
    "kPa":    "kilopaskal",
    "m/s":    "metr saniyə",
    "atm":    "atmosfer",
    "bar":    "bar",
    "°C":     "dərəcə Selsi",
    "°c":     "dərəcə Selsi",
    "%":      "faiz",
    "°":      "dərəcə",
}

# Qısaltma → tam söz
# Defisli hərflər ("en-es-ci-a") TTS-i qarışdırır — tam söz kimi yazılır
_QISALTMA = {
    "NSGA-II":  "en es ci a ikinci",
    "NSGA":     "en es ci a",
    "KKT":      "ka ka te",
    "IQR":      "i kyu ar",
    "ARO":      "a ar o",
    "R²":       "ar kvadrat",
    "F*":       "ef ulduz",
    "Y_benz":   "benzin verimi",
    "Y_diz":    "dizel verimi",
    "Y_ker":    "kerosin verimi",
    "α":        "alfa",
    "β":        "beta",
    "γ":        "qamma",
    "δ":        "delta",
    "Δ":        "dəyişim",
    "ΣF":       "cəm axın",
    "H₂":       "hidrogen",
    "CO₂":      "karbon dioksid",
    "°":        "dərəcə",
}

# İzolə edilmiş hərflərin tələffüzü (məsələn "T = 370" ifadəsindəki T)
_HERF_TELEFFUZ = {
    "T":  "temperatur",
    "P":  "təzyiq",
    "F":  "axın",
    "R":  "refluks",
    "E":  "enerji",
    "S":  "kükürd",
    "k":  "k əmsalı",
    "z":  "z koordinatı",
}

def _tts_metni_hazirla(metn: str, max_simvol: int = 4000) -> str:
    # 1. Sətir təmizliyi — cədvəl, markdown başlıq, siyahı nişanları
    xetler = []
    for xet in metn.split("\n"):
        xet = xet.strip()
        if not xet:
            continue
        if xet.startswith("|") or set(xet) <= set("|-:= "):
            continue
        xet = re.sub(r'^#{1,6}\s+', '', xet)
        xet = re.sub(r'^[-*•]\s+', '', xet)
        xetler.append(xet)
    metn = " ".join(xetler)

    # inline markdown
    metn = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', metn)
    metn = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', metn)
    metn = re.sub(r'`{1,3}[^`]*`{1,3}', '', metn)

    # URL-ləri sil
    metn = re.sub(r'https?://\S+', '', metn)

    # 2. Qısaltmalar — uzundan qısaya sırala ki, "NSGA-II" "NSGA"-dan əvvəl işlənsin
    for qisa, aciq in sorted(_QISALTMA.items(), key=lambda x: -len(x[0])):
        metn = metn.replace(qisa, aciq)

    # 3. İzolə hərfləri genişləndir: "T = 370" → "temperatur = 370"
    for herf, sozl in _HERF_TELEFFUZ.items():
        metn = re.sub(rf'\b{re.escape(herf)}\b(?=\s*[=\d])', sozl, metn)

    # 4. Rəqəm + vahid birləşməsi
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

    metn = re.sub(
        rf'([+\-]?)(\d+(?:[.,]\d+)?)\s*({v_pattern})',
        _reqem_vahid,
        metn,
    )

    # 5. Qalan tək rəqəmlər
    def _tek_reqem(m):
        try:
            return _reqem_sozle(m.group(0))
        except Exception:
            return m.group(0)

    metn = re.sub(
        r'(?<![a-zA-ZəçşğüöıƏÇŞĞÜÖI])\d+(?:[.,]\d+)?(?![a-zA-ZəçşğüöıƏÇŞĞÜÖI])',
        _tek_reqem,
        metn,
    )

    # 6. Durğu işarələrini TTS üçün normallaşdır
    # Tire cümlə əlaqəsini kəsir — vergüllə əvəz et
    metn = re.sub(r'\s*—\s*', ', ', metn)
    metn = re.sub(r'\s*–\s*', ', ', metn)
    # Mötərizələri sil — içərisini saxla
    metn = re.sub(r'\(([^)]+)\)', r'\1', metn)
    # Nöqtəli vergül → vergül
    metn = metn.replace(';', ',')
    # İki nöqtə → vergül
    metn = re.sub(r':\s*', ', ', metn)
    # Ardıcıl vergüllər → bir vergül
    metn = re.sub(r',\s*,+', ',', metn)

    # 7. Yalnız Azərbaycan + latın hərfləri, rəqəm, əsas durğu saxla
    AZ = 'çşğüöıəÇŞĞÜÖIƏ'
    metn = re.sub(
        rf'[^a-zA-Z{AZ}\u0400-\u04FF\s\.,!?]',
        ' ',
        metn,
        flags=re.UNICODE,
    )

    # 8. Boşluq normallaşdırma və uzunluq kəsmə
    metn = re.sub(r'\s{2,}', ' ', metn).strip()
    # Sonda yarımçıq cümlə qalmasın — ən yaxın nöqtədə kəs
    if len(metn) > max_simvol:
        metn = metn[:max_simvol].rsplit('.', 1)[0] + "."

    return metn


# Köhnə adı saxla — geriyə uyğunluq üçün
def _metni_temizle(metn: str, max_simvol: int = 4000) -> str:
    return _tts_metni_hazirla(metn, max_simvol)


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Sən **MeloSense** adlı AI köməkçisisin — Neft Emalı Optimallaşdırma Sisteminin proses analitik mühərriki. Adın MeloSense-dir, sənə bu adla müraciət oluna bilər.
ADNSU SABAH fakültəsinin buraxılış işi çərçivəsində hazırlanmış bu sistemin hər modulunu, tənliyini və fiziki məhdudiyyətlərini dərindən bilirsən.

## Sistem modulları
- **data_generator.py** — 9 sensor (T, P, F, H₂...), 24 saatlıq sintetik time-series, sinusoidal drift + anomaliya injeksiyası
- **preprocessor.py** — Filtrasiya → Min-Maks normallaşdırma → IQR + Isolation Forest → ARO reqressiya (R²≥0.95)
- **optimizer.py** — NSGA-II, KKT şərtləri, Pareto frontu, 100 fərd × 150 nəsil, Evklid məsafəsi ilə kompromis seçim
- **app.py** — 5 səhifəli Streamlit dashboard, premium dark UI

## Sənəd tənlikləri
- **(1)** F* = α·Y_benz + β·Y_diz + γ·Y_ker − δ·E  ← məqsəd funksiyası
- **(2)** x_norm = (x−x_min)/(x_max−x_min)  ← normallaşdırma
- **(3)** ΣF_giriş = ΣF_çıxış  ← kütlə balansı
- **(4)** Q_verilən = Q_istifadə + Q_itki  ← enerji balansı
- **(5)** T(z) = T_alt − k·z  ← temperatur profili
- **(6-8)** g(x)=0, g(x)≤0, x_min≤x≤x_max  ← KKT məhdudiyyətləri
- **(10)** L(x,λ,μ) = f(x)+Σλᵢgᵢ+Σμⱼhⱼ  ← Laqranj funksiyası
- **Y** = a₁T + a₂P + a₃F + a₄ + ε  ← reqressiya modeli

## Fiziki parametr həddləri
Soba: 340–400°C | Kolon: 1.2–1.5 atm | Axın: 80–160 m³/s
H₂: 30–60 bar | Katalizator: 280–360°C | Sıxlıq: 0.820–0.920 q/sm³

## DİL VƏ ÜSLUB QAYDLARI — BUNLARA CIDDI RIAYƏT ET

### Qadağan olunan ifadələr (heç vaxt istifadə etmə):
- "əhəmiyyətli artım/azalma" → əvəzinə: "nəzərəçarpacaq artım", "ölçülə bilən azalma", "qabarıq fərq"
- "tövsiyə olunur" (generic) → əvəzinə: konkret rəqəm və parametr aralığı ver
- "daha ekoloji cəhətdən dost" → əvəzinə: "enerji sərfini X GJ/s azaldır", "proses effektivliyini artırır"
- "sistemin optimallaşdırılması tövsiyə olunur" → əvəzinə: konkret dəyişən dəyəri ver
- "hesablanmamışdır" → əvəzinə: "Cari vəziyyət üçün inteqrasiya olunmuş performans skoru mövcud deyil"
- "normaldırmı" → HƏMIŞƏ "normaldır" (düzgün Azərbaycan dili)

### Cavab strukturu — MƏCBURI FORMAT:

Verim/enerji müqayisəsi üçün:
📈 [Göstərici adı]
Cari: XX.XX%
Optimal: XX.XX%
Δ: +X.XX%
Şərh: [Səbəb-nəticə cümləsi]

Tövsiyə üçün:
✅ [Parametr adı]: XXX–XXX°C aralığı
Səbəb: [texniki izah]
Gözlənilən effekt: [rəqəmli nəticə]

Anomaliya üçün:
⚠️ Anomaliya səviyyəsi: X.X%
Sənaye norması: 1.5–3.0%
Qiymət: normaldır / yüksəkdir / kritikdir

### Üslub prinsipləri:
- Hər cavabda səbəb → nəticə məntiqi olsun. Nümunə: Soba temperaturunun 362°C-yə endirilməsi enerji sərfini 0.8 GJ/s azaldarkən ümumi verim balansını qoruyur.
- Rəqəm olmayan tövsiyə verme — canlı kontekstdəki dəyərlərdən istifadə et
- Paragraflar qısa olsun — 2-3 cümlə maksimum, sonra format dəyiş
- F* skoru mövcud deyilsə: Cari vəziyyət üçün inteqrasiya olunmuş performans skoru mövcud deyil — yaz
- Azərbaycan dilinin orfoqrafik qaydalarına ciddi riayət et: ı, ə, ö, ü, ğ, ş, ç hərfləri düzgün işlənsin
- Cavabda heç vaxt dırnaq işarəsi (") işlətmə — hamısını düz cümlə kimi yaz
"""

# ══════════════════════════════════════════════════════════════════════════
# KONTEKST BUILDER
# ══════════════════════════════════════════════════════════════════════════
def kontekst_yarat(
    xulase: dict,
    opt_netice=None,
    req_goster=None,
    aktiv_sehife: str = "İcmal",
) -> str:
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    ctx = f"""## Canlı Sistem Vəziyyəti ({zaman}) | Aktiv: {aktiv_sehife}

### Sensor KPI-ları
| Göstərici | Dəyər |
|-----------|-------|
| Soba temp | {xulase.get('avg_furnace_temp',0):.1f} °C |
| Benzin verimi | {xulase.get('avg_yield_gasoline',0):.2f}% |
| Dizel verimi | {xulase.get('avg_yield_diesel',0):.2f}% |
| Kerosin verimi | {xulase.get('avg_yield_kerosene',0):.2f}% |
| Ümumi verim | {xulase.get('avg_total_yield',0):.2f}% |
| Enerji sərfi | {xulase.get('avg_energy',0):.3f} GS/s |
| Kükürd çıxarılması | {xulase.get('avg_sulfur_removal',0):.2f}% |
| Sensor nümunəsi | {xulase.get('n_samples',1440)} |
| Anomaliya | {xulase.get('n_anomalies',0)} ({xulase.get('anomaly_rate_pct',0):.1f}%) |
"""
    if opt_netice is not None:
        ey = opt_netice.best_compromise
        dv = ey.get('total_yield_pct',0) - xulase.get('avg_total_yield',0)
        de = xulase.get('avg_energy',0) - ey.get('energy_gj_h',0)
        ctx += f"""
### NSGA-II Optimal Həll
| | Cari | Optimal | Δ |
|-|------|---------|---|
| Verim | {xulase.get('avg_total_yield',0):.2f}% | {ey.get('total_yield_pct',0):.2f}% | +{dv:.2f}% |
| Enerji | {xulase.get('avg_energy',0):.3f} | {ey.get('energy_gj_h',0):.3f} | -{de:.3f} GS/s |
| F* balı | — | {ey.get('f_star',0):.4f} | — |
| Pareto həll | — | {opt_netice.n_solutions} | — |

Optimal soba temp: {ey.get('furnace_temp',0):.1f}°C | İllik qənaət: ~{de*8760:,.0f} GS/il
"""
    if req_goster is not None:
        ctx += "\n### Reqressiya R² Skorları\n"
        for idx_val, row in req_goster.iterrows():
            ctx += f"- {idx_val}: R²={row.get('R²',0):.4f}\n"

    return ctx.strip()


# ══════════════════════════════════════════════════════════════════════════
# CHATBOT — OpenAI SDK 2.x uyğun
# ══════════════════════════════════════════════════════════════════════════
class NeftEmalChatbot:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-your"):
            raise ValueError("OPENAI_API_KEY tapılmadı")
        self.client = OpenAI(api_key=api_key)
        self.model  = "gpt-4o"

    def _danisiq_metni_yarat(self, texniki_cavab: str) -> str:
        """
        Texniki AI cavabını TTS üçün təbii danışıq dilinə çevirir.
        Qısa, insan kimi, cədvəlsiz, formatsız mətn qaytarır.
        Məqsəd: Bir mühəndisin həmkarına şifahi izah etdiyi kimi səslənmək.
        """
        prompt = f"""Sən neft emalı sahəsindəki bir mütəxəssissən. Aşağıdakı texniki analizi oxu \
və bir həmkarına telefonda izah edirsən kimi sadə, axıcı Azərbaycan dilində danış.

Ciddi qaydalar:
- Maksimum 4–5 cümlə. Lazımsız heç nə əlavə etmə.
- Heç bir cədvəl, siyahı, işarə, markdown yoxdur — yalnız düz cümlələr.
- Rəqəmləri mütləq sözlə de: "üç yüz altmış iki dərəcə", "yeddi nöqtə altı faiz".
- Faiz işarəsi, °C, % kimi simvollar yoxdur — hamısını sözlə ifadə et.
- Cümlələrin başlanğıcı müxtəlif olsun. Hər cümlə "Bu..." ilə başlamasın.
- Doğal danışıq axışı: "Baxanda görürük ki...", "Qısaca desəm...", "Maraqlı olan budur ki...",
  "Nəticə etibarilə...", "Praktiki olaraq..." kimi giriş ifadələri işlət.
- Heç vaxt "Bu analiz göstərir ki" və ya "Hesablamalar nəticəsində" kimi texniki-robotik başlanğıc işlətmə.
- Sanki bir insanın öz fikrini danışdığı kimi səslənsin.

Texniki məlumat:
{texniki_cavab[:2000]}

İndi bunu canlı danışıq kimi yaz — yalnız düz mətn, sonda nöqtə:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.75,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return _metni_temizle(texniki_cavab, max_simvol=600)

    # ── ElevenLabs limit vəziyyəti (session cache) ────────────────────────
    _el_quota_exhausted: bool = False   # True olduqda ElevenLabs-ı devre dışı say

    def _el_kalan_limit(self) -> int | None:
        """
        ElevenLabs hesabındakı qalan simvol limitini qaytarır.
        API /v1/user endpoint-ini oxuyur.
        Xəta baş versə None qaytarır.
        """
        el_key = os.getenv("ELEVENLABS_API_KEY")
        if not el_key or el_key.startswith("your"):
            return None
        try:
            import urllib.request, json as _json
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": el_key, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = _json.loads(r.read())
            sub = data.get("subscription", {})
            used      = sub.get("character_count", 0)
            toplam    = sub.get("character_limit", 10_000)
            return max(0, toplam - used)
        except Exception:
            return None

    def ses_yarat(
        self,
        metn: str,
        ses: str = TTS_ELEVENLABS_VOICE,
        surət: float = TTS_SPEED,
    ) -> tuple[bytes | None, str]:
        """
        Cavabı TTS-ə çevirir. Avtomatik fallback mexanizmi:

          1. ElevenLabs limiti yetərincə varsa  → ElevenLabs eleven_multilingual_v2
          2. Limit bitibsə / API xətası varsa   → OpenAI tts-1-hd (fallback)

        Returns
        -------
        (audio_bytes, istifadə_olunan_motor)
          motor: "elevenlabs" | "openai" | "error"
        """
        # ── Mərhələ 1: danışıq mətni yarat ───────────────────────────────
        danisiq = self._danisiq_metni_yarat(metn)
        if not danisiq or len(danisiq) < 5:
            return None, "error"

        # ── Mərhələ 2: tələffüz normallaşdır ─────────────────────────────
        danisiq_tts = _tts_metni_hazirla(danisiq)
        simvol_sayi = len(danisiq_tts)

        # ── Mərhələ 3: ElevenLabs (limit yetərlidirsə) ────────────────────
        el_key = os.getenv("ELEVENLABS_API_KEY")
        el_aktiv = (
            _ELEVENLABS_AVAILABLE
            and el_key
            and not el_key.startswith("your")
            and not NeftEmalChatbot._el_quota_exhausted
        )

        if el_aktiv:
            # Limitin kifayət edib etmədiyini yoxla
            qalan = self._el_kalan_limit()
            if qalan is not None and qalan < simvol_sayi + 200:
                # +200 — kiçik ehtiyat bufer
                NeftEmalChatbot._el_quota_exhausted = True
                print(f"[ElevenLabs] Limit bitdi (qalan: {qalan}, lazım: {simvol_sayi}). "
                      f"OpenAI fallback-a keçilir.")
                el_aktiv = False

        if el_aktiv:
            try:
                el_client = ElevenLabsClient(api_key=el_key)
                # ElevenLabs üçün voice ID: əgər OpenAI səs adı gəlibsə default-a qayıt
                _openai_voices = {"nova", "shimmer", "alloy", "onyx", "echo", "fable", "ash", "coral", "sage"}
                el_voice = ses if ses not in _openai_voices else TTS_ELEVENLABS_VOICE
                audio_gen = el_client.text_to_speech.convert(
                    text=danisiq_tts,
                    voice_id=el_voice,
                    model_id=TTS_ELEVENLABS_MODEL,
                    voice_settings=VoiceSettings(
                        stability=0.45,
                        similarity_boost=0.82,
                        style=0.35,
                        use_speaker_boost=True,
                    ),
                    output_format="mp3_44100_128",
                    language_code="az",
                )
                audio = b"".join(audio_gen)
                return audio, "elevenlabs"
            except Exception as ex:
                err = str(ex).lower()
                if any(k in err for k in ("quota", "limit", "credit", "402", "429")):
                    NeftEmalChatbot._el_quota_exhausted = True
                    print(f"[ElevenLabs] Limit xətası — OpenAI fallback: {ex}")
                else:
                    print(f"[ElevenLabs] Xəta — OpenAI fallback: {ex}")

        # ── Mərhələ 4: OpenAI TTS fallback ───────────────────────────────
        try:
            cavab = self.client.audio.speech.create(
                model=TTS_MODEL_FALLBACK,
                voice=TTS_VOICE_FALLBACK,
                input=danisiq_tts,
                speed=surət,
                response_format=TTS_FORMAT,
            )
            return cavab.content, "openai"
        except Exception as ex:
            print(f"[OpenAI TTS xəta]: {ex}")
            return None, "error"

    def ses_yarat_dosya(
        self,
        metn: str,
        ses: str = TTS_ELEVENLABS_VOICE,
        surət: float = TTS_SPEED,
    ) -> tuple[str | None, str]:
        """
        Audio bytes-ı müvəqqəti .mp3 faylına yazır.
        Streamlit st.audio() üçün istifadə olunur.

        Returns
        -------
        (fayl_yolu, motor)
          fayl_yolu : str yolu və ya None
          motor     : "elevenlabs" | "openai" | "error"
        """
        audio_bytes, motor = self.ses_yarat(metn, ses=ses, surət=surət)
        if audio_bytes is None:
            return None, motor
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False, prefix="neft_tts_"
            ) as tmp:
                tmp.write(audio_bytes)
                return tmp.name, motor
        except Exception as ex:
            print(f"[TTS fayl xəta]: {ex}")
            return None, "error"

    def el_limit_goster(self) -> dict:
        """
        ElevenLabs cari limit vəziyyətini qaytarır.
        Streamlit dashboard-da göstərmək üçün.

        Returns: {qalan, toplam, istifade, faiz, exhausted}
        """
        el_key = os.getenv("ELEVENLABS_API_KEY")
        if not _ELEVENLABS_AVAILABLE or not el_key or el_key.startswith("your"):
            return {"aktiv": False}
        try:
            import urllib.request, json as _json
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": el_key, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = _json.loads(r.read())
            sub     = data.get("subscription", {})
            used    = sub.get("character_count", 0)
            toplam  = sub.get("character_limit", 10_000)
            qalan   = max(0, toplam - used)
            return {
                "aktiv":     True,
                "qalan":     qalan,
                "toplam":    toplam,
                "istifade":  used,
                "faiz":      round(used / toplam * 100, 1) if toplam else 0,
                "exhausted": NeftEmalChatbot._el_quota_exhausted,
                "plan":      sub.get("tier", "free"),
            }
        except Exception:
            return {"aktiv": True, "qalan": None, "exhausted": NeftEmalChatbot._el_quota_exhausted}

    def cavab_ver_stream(        self,
        mesajlar: list[dict],
        kontekst: str,
    ) -> Generator[str, None, None]:
        """
        OpenAI SDK 2.x — stream=True ilə create() istifadə et.
        chunk.choices[0].delta.content pattern-i düzgün işləyir.
        """
        tam_system = SYSTEM_PROMPT + f"\n\n---\n\n{kontekst}"
        api_mesajlar = [{"role": "system", "content": tam_system}] + mesajlar

        bashlama = time.time()
        try:
            axin = self.client.chat.completions.create(
                model=self.model,
                messages=api_mesajlar,
                max_tokens=1200,
                temperature=0.45,
                stream=True,            # ← SDK 2.x üçün düzgün yol
            )
            # Tam cavabı yığ — backtick/dırnaq filtrini axın bitdikdən
            # sonra bir dəfə tətbiq etmək daha etibarlıdır (chunk sərhəddini aşan
            # ``` ardıcıllığını parça-parça tutmaq mümkün deyil)
            tam = ""
            for chunk in axin:
                if (chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content):
                    tam += chunk.choices[0].delta.content
            # ── Artefakt təmizliyi ───────────────────────────────────────
            tam = re.sub(r"```[\w]*\n?", "", tam)   # ``` blok açılışı
            tam = re.sub(r"```", "", tam)              # ``` blok bağlanışı
            tam = re.sub(r"`", "", tam)                # tək backtick
            tam = tam.replace('"', '')              # dırnaq işarəsi
            tam = re.sub(r"\s{2,}", " ", tam).strip()
            yield tam

        except Exception as ex:
            # Texniki xəta istifadəçiyə sadə dildə göstərilir
            err_msg = str(ex)
            if "api_key" in err_msg.lower() or "authentication" in err_msg.lower():
                yield "\n\n⚠️ **API açarı yanlışdır.** `.env` faylındakı `OPENAI_API_KEY`-i yoxlayın."
            elif "rate_limit" in err_msg.lower():
                yield "\n\n⚠️ **Sorğu limiti aşıldı.** Bir neçə saniyə gözləyib yenidən cəhd edin."
            elif "network" in err_msg.lower() or "connection" in err_msg.lower():
                yield "\n\n⚠️ **Şəbəkə problemi.** İnternet bağlantınızı yoxlayın."
            else:
                yield "\n\n⚠️ **AI cavabı alınarkən problem yarandı.** Yenidən cəhd edin."

        self.son_latency = round(time.time() - bashlama, 2)

    def tez_analiz(self, kontekst: str, nov: str) -> Generator[str, None, None]:
        """Hazır analiz sorğuları."""
        sorğular = {
            "pareto":    "Cari Pareto frontunu şərh et. Kompromis həllin seçilmə meyarını (Evklid məsafəsi) izah et. Enerji-verim mübadiləsini rəqəmlərlə göstər. Niyə bu nöqtə utopik nöqtəyə ən yaxındır?",
            "anomaliya": "Aşkar edilmiş anomaliyaları analiz et. Bu faiz sənaye normallaşdırılmış həddə (1.5–3.0%) uyğundur ya yüksəkdir? Tövsiyəni texniki dillə ifadə et.",
            "tövsiyə":   "Cari sensor məlumatlarına əsasən proses effektivliyini artırmaq üçün 3-5 konkret parametr dəyişikliyi təklif et. Hər tövsiyə üçün: mövcud dəyər → tövsiyə olunan aralıq → gözlənilən effekt (rəqəmlə).",
            "tənlik":    "Məqsəd funksiyasını F* = α·Y_benz + β·Y_diz + γ·Y_ker − δ·E (tənlik 1) canlı dəyərlərlə hesabla. α, β, γ, δ çəkilərinin hər birinin F*-a miqdar töhfəsini göstər. KKT şərtlərinin mövcud rejimdə yerinə yetirilməsini yoxla.",
            "müqayisə":  "Cari vəziyyəti NSGA-II optimal həlli ilə müqayisəli cədvəl şəklində göstər. Hər göstərici üçün: Cari → Optimal → Δ (mütləq və faizlə). Ümumi enerji qənaətini illik bazada hesabla.",
        }
        sual = sorğular.get(nov, "Sistemin ümumi vəziyyətini qiymətləndir.")
        yield from self.cavab_ver_stream([{"role": "user", "content": sual}], kontekst)
        self.son_latency = getattr(self, 'son_latency', 0)


def chatbot_yarat() -> "NeftEmalChatbot | None":
    try:
        return NeftEmalChatbot()
    except ValueError:
        return None