"""Timbre y ritmo en hasta seis ventanas no solapadas de diez segundos."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf

FEATURE_VERSION = "numpy_timbre_rhythm_six_windows_v2"
SAMPLE_RATE = 22050
MAX_BYTES = 100 * 1024 * 1024
FEATURE_NAMES = tuple(
    f"{name}_{stat}" for name in
    [*[f"mfcc_{i + 1}" for i in range(20)], "centroid", "bandwidth", "rolloff", "zcr",
     *[f"chroma_{i}" for i in range(12)]] for stat in ("mean", "std")
)
RHYTHM_NAMES = ("tempo_bpm", "periodicity", "onsets_per_second", "flux_variation",
                "pulse_60", "pulse_80", "pulse_100", "pulse_120", "pulse_150", "pulse_180")
FEATURE_NAMES += tuple(f"{name}_{stat}" for name in RHYTHM_NAMES for stat in ("mean", "std"))


@dataclass
class Analysis:
    duration: float
    vector: np.ndarray
    intensity: float
    sample_rate: int
    analyzed_seconds: float


def file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def analyze(path: str | Path) -> Analysis:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("No se encuentra el archivo o supera el límite de 100 MB.")
    try:
        with sf.SoundFile(path) as audio:
            duration = len(audio) / audio.samplerate
            if not 3 <= duration <= 1800:
                raise ValueError("Usa audio de entre 3 segundos y 30 minutos.")
            sr = audio.samplerate
            if not 8000 <= sr <= 192000 or audio.channels > 8:
                raise ValueError("Frecuencia de muestreo o canales no compatibles.")
            window = min(10.0, duration)
            count = min(6, max(1, int(duration / window)))
            starts = np.linspace(0, duration - window, count)
            chunks = []
            for start in starts:
                audio.seek(round(start * sr))
                y = audio.read(round(window * sr), dtype="float32", always_2d=True).mean(axis=1)
                if not np.isfinite(y).all():
                    raise ValueError("El audio contiene valores inválidos.")
                if sr != SAMPLE_RATE:
                    from scipy.signal import resample_poly
                    common = gcd(sr, SAMPLE_RATE)
                    y = resample_poly(y, SAMPLE_RATE // common, sr // common)
                chunks.append(y)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("No se pudo decodificar el audio. Prueba una copia WAV o MP3 válida.") from exc
    if max(float(np.max(np.abs(y))) for y in chunks) < 1e-5:
        raise ValueError("Los fragmentos analizados están en silencio; no se puede clasificarlos.")
    matrices, rms, rhythms = [], [], []
    for y in chunks:
        matrices.append(frame_features(y))
        rhythms.append(rhythm_features(y))
        rms.append(float(np.sqrt(np.mean(y**2))))
    matrix = np.concatenate(matrices, axis=1)
    vector = np.column_stack([matrix.mean(axis=1), matrix.std(axis=1)]).ravel()
    rhythm = np.asarray(rhythms)
    vector = np.concatenate([vector, np.column_stack([rhythm.mean(axis=0), rhythm.std(axis=0)]).ravel()])
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("No se pudieron extraer características válidas.")
    # Proxy explícito de amplitud, no emoción ni intensidad perceptual universal.
    db = 20 * np.log10(max(float(np.mean(rms)), 1e-8))
    intensity = float(np.clip((db + 60) / 60, 0, 1))
    return Analysis(duration, vector, intensity, sr, sum(len(y) / SAMPLE_RATE for y in chunks))


def rhythm_features(y):
    """Flujo espectral positivo y autocorrelación; tempo aproximado, no detector de género.

    Se calcula por ventana para no crear golpes falsos entre fragmentos.
    La normalización reduce la dependencia del volumen. El tempo puede presentar
    ambigüedad de mitad/doble; se conservan además seis valores de periodicidad.
    """
    from scipy.signal import correlate, find_peaks
    hop, n_fft = 256, 1024
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    scale = float(magnitude.max())
    if scale < 1e-8:
        return np.zeros(len(RHYTHM_NAMES))
    spectrum = np.log1p(100 * magnitude / scale)
    flux = np.maximum(np.diff(spectrum, axis=0), 0).mean(axis=1)
    mean, std = float(flux.mean()), float(flux.std())
    if std < 1e-8:
        return np.zeros(len(RHYTHM_NAMES))
    envelope = flux - mean
    ac = correlate(envelope, envelope, mode="full", method="fft")[len(envelope)-1:]
    ac /= max(float(ac[0]), 1e-12)
    fps = SAMPLE_RATE / hop
    low, high = int(np.ceil(fps * 60 / 200)), int(np.floor(fps * 60 / 50))
    lag = low + int(np.argmax(ac[low:high+1]))
    strength = max(0.0, float(ac[lag]))
    peaks, _ = find_peaks(flux, height=mean + .5 * std, prominence=.3 * std,
                          distance=max(1, int(.12 * fps)))
    pulses = [max(0.0, float(ac[round(fps * 60 / bpm)])) for bpm in (60, 80, 100, 120, 150, 180)]
    return np.array([60 * fps / lag if strength > .05 else 0, strength,
                     len(peaks) / (len(y) / SAMPLE_RATE), std / max(mean, 1e-8), *pulses])


def frame_features(y):
    """STFT Hann 2048/salto512; mel HTK triangular y DCT-II ortonormal.

    Implementación propia documentada, no numéricamente intercambiable con librosa.
    El chroma agrega potencia en el semitono temperado más cercano (A4=440 Hz).
    """
    from scipy.fft import dct
    n_fft, hop = 2048, 512
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (n_fft // 2, n_fft // 2)), n_fft)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)).T
    power = spectrum**2
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    energy = np.maximum(spectrum.sum(axis=0), 1e-12)
    centroid = (freqs[:, None] * spectrum).sum(axis=0) / energy
    bandwidth = np.sqrt(((freqs[:, None] - centroid)**2 * spectrum).sum(axis=0) / energy)
    rolloff = freqs[(np.cumsum(spectrum, axis=0) >= .85 * energy).argmax(axis=0)]
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    mel_points = np.linspace(0, 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700), 130)
    hz = 700 * (10**(mel_points / 2595) - 1)
    lower = (freqs[None, :] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
    upper = (hz[2:, None] - freqs[None, :]) / (hz[2:] - hz[1:-1])[:, None]
    filters = np.maximum(0, np.minimum(lower, upper))
    mel_db = 10 * np.log10(np.maximum(filters @ power, 1e-10))
    mfcc = dct(mel_db, type=2, axis=0, norm="ortho")[:20]
    chroma = np.zeros((12, power.shape[1]))
    pitched = freqs >= 27.5
    bins = np.rint(69 + 12 * np.log2(freqs[pitched] / 440)).astype(int) % 12
    for note in range(12):
        chroma[note] = power[pitched][bins == note].sum(axis=0)
    chroma /= np.maximum(np.linalg.norm(chroma, axis=0), 1e-12)
    return np.vstack([mfcc, centroid, bandwidth, rolloff, zcr, chroma])
