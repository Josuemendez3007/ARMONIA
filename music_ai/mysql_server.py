"""Servicio MySQL de esta copia de Armonía, con rutas relativas al proyecto."""
import json
import socket
import subprocess
import time
from pathlib import Path


def server_args(data):
    data = Path(data).resolve()
    base = data.parent / 'runtime' / 'mysql'
    private = data / 'mysql_local'
    port = json.loads((private / 'app.json').read_text(encoding='utf-8'))['port']
    return [str(base / 'bin' / 'mysqld.exe'), '--no-defaults',
            f'--basedir={base}', f'--datadir={private / "db"}',
            '--bind-address=127.0.0.1', f'--port={port}', '--mysqlx=0',
            '--skip-log-bin', f'--log-error={private / "server.err"}']


def ensure_server(data):
    data = Path(data).resolve()
    private = data / 'mysql_local'
    port = json.loads((private / 'app.json').read_text(encoding='utf-8'))['port']
    def ready():
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return True
        except OSError:
            return False
    if ready():
        return
    proc = subprocess.Popen(server_args(data), creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(90):
        if ready():
            return
        if proc.poll() is not None:
            break
        time.sleep(.5)
    raise RuntimeError('MySQL no pudo iniciar. Consulta data/mysql_local/server.err.')
