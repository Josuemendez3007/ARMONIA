"""Importación local por carpeta; extrae audio MP4 sin modificar originales."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from music_ai import GENRES
from music_ai.audio import analyze, file_hash
from music_ai.knowledge import open_catalog, now


def import_folder(folder, genre, data_dir="data"):
    folder, data = Path(folder).resolve(), Path(data_dir).resolve()
    if not folder.is_dir() or genre not in GENRES:
        raise ValueError("Carpeta o género inválido.")
    library = data / "library"
    library.mkdir(parents=True, exist_ok=True)
    catalog = open_catalog(data)
    results = []
    for source in sorted(folder.iterdir()):
        if not source.is_file() or source.suffix.lower() not in (".mp4", ".wav", ".mp3", ".flac"):
            continue
        record = {"original_path": str(source), "label": genre, "label_source": "carpeta etiquetada por el usuario",
                  "imported_at": now(), "split": None, "usage_permission": "pendiente de documentar"}
        try:
            record["original_sha256"] = file_hash(source)
            with tempfile.TemporaryDirectory(prefix="musical_import_") as temp:
                audio = source
                if source.suffix.lower() == ".mp4":
                    import imageio_ffmpeg
                    audio = Path(temp) / "audio.flac"
                    conversion = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner",
                        "-loglevel", "error", "-n", "-i", str(source), "-map", "0:a:0", "-vn",
                        "-map_metadata", "-1", "-c:a", "flac", str(audio)], capture_output=True, timeout=180)
                    if conversion.returncode:
                        raise ValueError("No se pudo extraer la pista de audio: " + conversion.stderr.decode(errors="replace")[-500:])
                analysis = analyze(audio)
                digest = file_hash(audio)
                target = library / (digest + audio.suffix.lower())
                if not target.exists():
                    shutil.copyfile(audio, target)
                added = catalog.add(digest, source.stem + audio.suffix.lower(), target, analysis)
                current = next(s for s in catalog.songs() if s["id"] == digest)
                if not current["confirmed_genre"]:
                    catalog.confirm(digest, genre)
                record.update(id=digest, audio_path=str(target), duration_seconds=analysis.duration,
                              feature_count=len(analysis.vector), confirmed_genre=current["confirmed_genre"] or genre,
                              status="importada" if added else "ya registrada")
                if current["confirmed_genre"] and current["confirmed_genre"] != genre:
                    record["warning"] = "Se conservó la etiqueta previa del catálogo; revisar discrepancia."
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            record.update(status="error", error=str(exc))
        results.append(record)
        print(json.dumps({"file": source.name, "status": record["status"],
                          "seconds": record.get("duration_seconds"), "error": record.get("error")}, ensure_ascii=False), flush=True)
    report = data / ("intake_" + genre + "_" + now().replace(":", "-") + ".json")
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--genre", required=True, choices=GENRES)
    args = parser.parse_args()
    rows, report = import_folder(args.folder, args.genre)
    print(f"Informe: {report}")
    raise SystemExit(1 if any(r["status"] == "error" for r in rows) else 0)
