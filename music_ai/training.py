"""Entrenamiento reproducible con particiones explícitas y validación independiente."""
import argparse
from functools import lru_cache
from threading import RLock
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np

from music_ai import GENRES
from music_ai.audio import analyze, file_hash, FEATURE_VERSION, FEATURE_NAMES

SPLITS = ("train", "validation", "test")


def read_manifest(path, allow_partial_test=False):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "genre", "group", "split", "source", "license"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requerido: path,genre,group,split,source,license.")
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]
    seen_groups, seen_hashes = {}, set()
    for i, row in enumerate(rows, start=2):
        if any(not row.get(k) for k in required):
            raise ValueError(f"Fila {i}: completa todos los campos, incluida procedencia y permiso.")
        if row["genre"] not in GENRES or row["split"] not in SPLITS:
            raise ValueError(f"Fila {i}: género o partición inválidos.")
        group = row["group"].casefold()
        if group in seen_groups and seen_groups[group] != row["split"]:
            raise ValueError(f"Fuga de información: el grupo {row['group']} aparece en distintas particiones.")
        seen_groups[group] = row["split"]
        audio_path = (path.parent / row["path"]).resolve()
        if not audio_path.is_file():
            raise ValueError(f"Fila {i}: no se encuentra {audio_path.name}.")
        digest = file_hash(audio_path)
        if digest in seen_hashes:
            raise ValueError(f"Fila {i}: archivo duplicado por contenido; elimina la repetición.")
        seen_hashes.add(digest)
        row["path"], row["sha256"] = str(audio_path), digest
    for split in SPLITS:
        counts = Counter(r["genre"] for r in rows if r["split"] == split)
        minimum = 2 if split == "train" else 1
        if split == "test" and allow_partial_test and sum(counts.values()) > 0:
            continue
        if any(counts[g] < minimum for g in GENRES):
            raise ValueError(f"Faltan datos en {split}: mínimo técnico {minimum} canciones por género. No basta para una evaluación concluyente.")
    return rows


def fit_models(vectors, rows):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X = np.asarray(vectors)
    y = np.array([r["genre"] for r in rows])
    split = np.array([r["split"] for r in rows])
    masks = {s: split == s for s in SPLITS}
    if X.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(X).all():
        raise ValueError("Matriz de características incompatible.")
    candidates = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "svm": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                class_weight="balanced", random_state=1907, n_jobs=-1),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(X[masks["train"]], y[masks["train"]])
        prediction = model.predict(X[masks["validation"]])
        validation[name] = float(f1_score(y[masks["validation"]], prediction, labels=list(GENRES), average="macro", zero_division=0))
    # La línea base se reporta; la elección entre modelos usa exclusivamente validación.
    selected = max(("svm", "random_forest"), key=lambda n: validation[n])
    model = candidates[selected]
    test_prediction = model.predict(X[masks["test"]])
    report = {
        "selected": selected, "validation_f1_macro": validation,
        "test": classification_report(y[masks["test"]], test_prediction, labels=list(GENRES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y[masks["test"]], test_prediction, labels=list(GENRES)).tolist(),
        "matrix_labels": list(GENRES), "matrix_axes": "filas reales, columnas predichas",
        "counts": {s: dict(Counter(y[masks[s]])) for s in SPLITS},
        "calibrated": False, "automatic_acceptance": False,
        "limitations": ["Puntuaciones no calibradas: siempre requiere revisión humana.",
                        "El test evalúa solo el modelo seleccionado, sin reentrenar con validación.",
                        "El hash detecta duplicados binarios, no versiones recodificadas. Agrupar manualmente obras/versiones.",
                        "Reutilizar el test para cambiar decisiones deja de ser una evaluación independiente."]}
    return model, report


def train(manifest, output="models/genre_model.joblib", allow_partial_test=False):
    rows = read_manifest(manifest, allow_partial_test=allow_partial_test)
    vectors = []
    for i, row in enumerate(rows, 1):
        print(f"Extrayendo {i}/{len(rows)}: {Path(row['path']).name}", flush=True)
        vectors.append(analyze(row["path"]).vector)
    model, report = fit_models(vectors, rows)
    missing_test = [g for g in GENRES if not report["counts"]["test"].get(g)]
    report["test_missing_genres"] = missing_test
    if missing_test:
        report["limitations"].append("Prueba incompleta; no evalúa estos géneros: " + ", ".join(missing_test))
    model_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    versions = {p: version(p) for p in ("scikit-learn", "scipy", "numpy", "soundfile")}
    report.update(model_id=model_id, feature_version=FEATURE_VERSION, versions=versions,
                  manifest=[{k: r[k] for k in ("path", "genre", "group", "split", "sha256", "source", "license")} for r in rows])
    artifact = {"model": model, "model_id": model_id, "feature_version": FEATURE_VERSION,
                "feature_names": FEATURE_NAMES, "versions": versions, "calibrated": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Versiones inmutables además del puntero local usado por la aplicación.
    versioned = output.with_name(f"{output.stem}_{model_id}.joblib")
    joblib.dump(artifact, versioned)
    report_path = versioned.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp = output.with_suffix(".tmp")
    joblib.dump(artifact, temp)
    temp.replace(output)
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


_model_lock = RLock()


def load_model(path):
    """Reuse an immutable inference artifact until its file changes.

    Each process owns at most two versions. Never fit or mutate returned models.
    The lock prevents concurrent imports from deserializing the same file twice.
    """
    path = Path(path).resolve()
    with _model_lock:
        stamp = path.stat()
        return _load_model_cached(str(path), stamp.st_mtime_ns, stamp.st_ctime_ns, stamp.st_size)


@lru_cache(maxsize=2)
def _load_model_cached(path, modified, changed, size):
    # Solo artefactos locales de confianza: joblib no es un formato seguro para archivos ajenos.
    artifact = joblib.load(path)
    if artifact.get("feature_version") != FEATURE_VERSION or tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Modelo incompatible con el extractor de esta versión; vuelve a entrenar.")
    if artifact.get("versions", {}).get("scikit-learn") != version("scikit-learn"):
        raise ValueError("La versión de scikit-learn cambió. Usa requirements-lock.txt o vuelve a entrenar.")
    return artifact


def predict(artifact, vector):
    X = np.asarray(vector).reshape(1, -1)
    model = artifact["model"]
    genre = str(model.predict(X)[0])
    scores = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else model.decision_function(X)[0]
    return {"genre": genre, "scores": dict(zip(map(str, model.classes_), map(float, scores))),
            "score_type": "proporción no calibrada" if hasattr(model, "predict_proba") else "puntuación de decisión SVM",
            "model_id": artifact["model_id"], "requires_review": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar el clasificador con un CSV documentado.")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="models/genre_model.joblib")
    parser.add_argument("--allow-partial-test", action="store_true", help="Permite una prueba provisional sin todos los géneros; queda documentado en el informe.")
    args = parser.parse_args()
    try:
        result = train(args.manifest, args.output, allow_partial_test=args.allow_partial_test)
        print(json.dumps({k: result[k] for k in ("selected", "validation_f1_macro", "test")}, indent=2, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"No se pudo entrenar: {exc}\n")
