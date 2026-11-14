# Armonía
Aplicación web del equipo Circo digital para clasificar música en pop, reguetón, rock y rap, revisar etiquetas y generar listas por género, duración e intensidad.

## Ejecutar en Windows
1. Instala Python 3.12 y crea un entorno con `py -3.12 -m venv .venv`.
2. Instala las dependencias con `.venv\Scripts\python -m pip install imageio-ffmpeg==0.7.0 joblib==1.6.0 mysql-connector-python==26.10.0 numpy==2.5.4 pandas==3.1.0 scikit-learn==1.9.1 scipy==1.18.1 soundfile==0.14.0 streamlit==1.66.1`.
3. Inicia un servidor MySQL local. Ejecuta `schema.sql` desde MySQL Workbench y concede a un usuario local permisos SELECT, INSERT y UPDATE sobre armonia.*.
4. Crea la carpeta `data/mysql_local` y dentro el archivo privado `app.json`. Ejemplo para un servidor ya iniciado en el puerto 3306:

```json
{"host":"127.0.0.1","port":3306,"user":"TU_USUARIO","password":"TU_CLAVE","database":"armonia"}
```

5. Desde la carpeta del proyecto ejecuta `.venv\Scripts\python -m streamlit run app.py`. Abre la dirección local indicada en la terminal.
6. En «Añadir música» importa tus audios y confirma sus géneros en «Biblioteca» para generar listas.

Se incluye el bosque aleatorio entrenado y su informe de evaluación. La biblioteca comienza vacía.
