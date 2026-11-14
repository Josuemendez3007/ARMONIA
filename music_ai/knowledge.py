"""Catálogo persistente y revisiones: una corrección nunca borra la predicción."""
from datetime import datetime, timezone
import json
from pathlib import Path



def now():
    return datetime.now(timezone.utc).isoformat()


def open_catalog(data):
    """Conecta con la biblioteca musical de MySQL de esta instalación."""
    from music_ai.mysql_server import ensure_server
    from music_ai.mysql_catalog import MySQLCatalog

    data = Path(data)
    settings = json.loads((data / 'database.json').read_text(encoding='utf-8'))
    ensure_server(data)
    return MySQLCatalog(data, data / settings['credentials_file'])


def eligibility(song, allowed, excluded=()):
    """Reglas prioritarias; todas las sugerencias requieren revisión en v1."""
    if song["id"] in excluded:
        return False, "R01: excluida por el usuario."
    if not Path(song["path"]).is_file():
        return False, "R02: el archivo del catálogo no está disponible."
    if not song["confirmed_genre"]:
        return False, "R03: necesita confirmación humana; la sugerencia no está calibrada."
    if song["confirmed_genre"] not in allowed:
        return False, "R04: género confirmado fuera de los permitidos."
    return True, "R05: archivo disponible y género confirmado permitido."
