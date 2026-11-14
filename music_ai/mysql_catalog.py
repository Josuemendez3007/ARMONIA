"""Shared MySQL catalog for desktop, Streamlit and import scripts."""
from contextlib import contextmanager
from pathlib import Path
import json
import mysql.connector
from music_ai import GENRES
from music_ai.audio import FEATURE_VERSION


class MySQLCatalog:
    def __init__(self, data, credentials_file):
        self.data = Path(data)  # Carpeta que contiene los audios de la biblioteca.
        self.settings = json.loads(Path(credentials_file).read_text(encoding='utf-8'))

    @contextmanager
    def connection(self):
        db = mysql.connector.connect(**self.settings, connection_timeout=5)
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def songs(self, include_features=True):
        with self.connection() as db:
            with db.cursor(dictionary=True) as cursor:
                columns = '*' if include_features else 'id,name,path,duration,intensity,feature_version,prediction,confirmed_genre,created_at'
                cursor.execute(f'SELECT {columns} FROM songs ORDER BY created_at,id')
                rows = cursor.fetchall()
        for row in rows:
            row["path"] = str((self.data / row["path"]).resolve())
            if include_features:
                row['features'] = json.loads(row['features'])
            row['prediction'] = json.loads(row['prediction']) if row['prediction'] else None
        return rows

    def add(self, song_id, name, path, analysis, prediction=None):
        from music_ai.knowledge import now
        try:
            with self.connection() as db:
                with db.cursor() as cursor:
                    cursor.execute('INSERT INTO songs (id,name,path,duration,intensity,features,feature_version,prediction,confirmed_genre,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s)',
                        (song_id,name,Path(path).resolve().relative_to(self.data.resolve()).as_posix(),analysis.duration,analysis.intensity,
                         json.dumps(analysis.vector.tolist()),FEATURE_VERSION,json.dumps(prediction) if prediction else None,now()))
            return True
        except mysql.connector.IntegrityError as exc:
            if exc.errno == 1062: return False
            raise

    def contains(self, song_id):
        # Primary-key lookup avoids transferring the full library for duplicates.
        with self.connection() as db:
            with db.cursor() as cursor:
                cursor.execute('SELECT 1 FROM songs WHERE id=%s', (song_id,))
                return cursor.fetchone() is not None

    def get(self, song_id):
        with self.connection() as db:
            with db.cursor(dictionary=True) as cursor:
                cursor.execute('SELECT * FROM songs WHERE id=%s', (song_id,))
                row = cursor.fetchone()
        if row is None:
            raise ValueError('La canción ya no está en el catálogo.')
        row['path'] = str((self.data / row['path']).resolve())
        row['features'] = json.loads(row['features'])
        row['prediction'] = json.loads(row['prediction']) if row['prediction'] else None
        return row

    def confirm(self, song_id, genre):
        from music_ai.knowledge import now
        if genre not in GENRES: raise ValueError('Género no admitido.')
        with self.connection() as db:
            with db.cursor() as cursor:
                cursor.execute('SELECT confirmed_genre FROM songs WHERE id=%s FOR UPDATE',(song_id,))
                row=cursor.fetchone()
                if row is None: raise ValueError('La canción ya no está en el catálogo.')
                cursor.execute('UPDATE songs SET confirmed_genre=%s WHERE id=%s',(genre,song_id))
                cursor.execute('INSERT INTO revisions(song_id,old_genre,new_genre,created_at) VALUES(%s,%s,%s,%s)',(song_id,row[0],genre,now()))

    def history(self, song_id):
        with self.connection() as db:
            with db.cursor(dictionary=True) as cursor:
                cursor.execute('SELECT * FROM revisions WHERE song_id=%s ORDER BY id',(song_id,))
                return cursor.fetchall()
