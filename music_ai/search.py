"""Enumeración exacta y haz; mismo costo y restricciones, garantías diferentes."""
from dataclasses import dataclass, asdict
import math
from heapq import nsmallest
from statistics import median
from time import perf_counter


@dataclass(frozen=True)
class Track:
    id: str
    duration: float
    intensity: float


@dataclass(frozen=True)
class Request:
    target: float
    tolerance: float = 15
    progression: str = "creciente"
    duration_weight: float = 0.6

    def validate(self):
        if not math.isfinite(self.target) or self.target <= 0:
            raise ValueError("La duración objetivo debe ser positiva.")
        if not math.isfinite(self.tolerance) or not 0 <= self.tolerance < self.target:
            raise ValueError("La tolerancia debe ser no negativa y menor que el objetivo.")
        if self.progression not in ("creciente", "decreciente", "constante"):
            raise ValueError("Progresión desconocida.")
        if not 0 <= self.duration_weight <= 1:
            raise ValueError("Peso de duración fuera de rango.")


def reference(i, n, mode):
    value = i / (n - 1) if n > 1 else 0.5
    return 0.5 if mode == "constante" else (1 - value if mode == "decreciente" else value)


def cost(sequence, req):
    if not sequence:
        return math.inf
    error = sum(abs(s.intensity - reference(i, len(sequence), req.progression))
                for i, s in enumerate(sequence)) / len(sequence)
    return req.duration_weight * abs(sum(s.duration for s in sequence) - req.target) / req.target + (1 - req.duration_weight) * error


@dataclass
class Result:
    ids: list[str]
    duration: float
    cost: float | None
    status: str
    method: str
    generated: int
    expanded: int
    pruned: int
    elapsed: float
    reason: str

    def to_dict(self):
        return asdict(self)


def search(tracks, req, method="beam", width=25, seconds=5.0):
    req.validate()
    tracks = sorted(list(tracks), key=lambda t: t.id)
    if method not in ("beam", "exact") or width < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Configuración de búsqueda inválida.")
    if method == "exact" and len(tracks) > 10:
        raise ValueError("La enumeración exacta admite como máximo 10 canciones elegibles.")
    if len({t.id for t in tracks}) != len(tracks):
        raise ValueError("Hay identificadores repetidos.")
    if any(not math.isfinite(t.duration) or t.duration <= 0 or not math.isfinite(t.intensity) or not 0 <= t.intensity <= 1 for t in tracks):
        raise ValueError("Duración o intensidad inválida en el catálogo.")
    start = perf_counter()
    generated = expanded = pruned = 0
    best, best_cost = (), math.inf
    timed_out = lost = False
    low, high = req.target - req.tolerance, req.target + req.tolerance
    total = sum(t.duration for t in tracks)
    typical = median([t.duration for t in tracks]) if tracks else 1

    def inspect(seq, duration):
        nonlocal best, best_cost
        if seq and low <= duration <= high:
            value = cost(seq, req)
            if value < best_cost:
                best, best_cost = seq, value

    def extend(seq, duration):
        nonlocal generated, expanded, pruned, timed_out
        expanded += 1
        used = {t.id for t in seq}
        for t in tracks:
            if perf_counter() - start >= seconds:
                timed_out = True
                return
            if t.id in used:
                continue
            generated += 1
            if duration + t.duration > high:
                pruned += 1
                continue
            child = seq + (t,)
            inspect(child, duration + t.duration)
            yield child, duration + t.duration

    def visit(seq, duration):
        for child, d in extend(seq, duration):
            visit(child, d)
            if timed_out:
                break

    def priority(state):
        seq, d = state
        n = min(len(tracks), len(seq) + math.ceil(max(0, req.target - d) / typical))
        error = sum(abs(t.intensity - reference(i, n, req.progression)) for i, t in enumerate(seq)) / len(seq)
        return req.duration_weight * abs(req.target - d) / req.target + (1 - req.duration_weight) * error

    if total >= low:
        if method == "exact":
            visit((), 0.0)
        else:
            frontier = [((), 0.0)]
            while frontier and not timed_out:
                # Keep only the best width states instead of materializing and
                # sorting all successors. nsmallest preserves ties in input order.
                child_count = 0
                def successors():
                    nonlocal child_count
                    for seq, d in frontier:
                        for child in extend(seq, d):
                            child_count += 1
                            yield child
                        if timed_out:
                            break
                next_frontier = nsmallest(width, successors(), key=priority)
                if child_count > width:
                    lost = True
                    pruned += child_count - width
                frontier = next_frontier
    complete = not timed_out and not lost
    status = ("optimal" if complete else "found") if best else ("infeasible" if complete else "not_found")
    reason = "Tiempo agotado" if timed_out else ("Exploración con descarte por ancho" if lost else "Espacio agotado o inviabilidad comprobada")
    return Result([t.id for t in best], sum(t.duration for t in best), best_cost if best else None,
                  status, method, generated, expanded, pruned, perf_counter() - start, reason)
