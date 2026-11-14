"""Interfaz local. Ejecutar con iniciar.bat o python -m streamlit run app.py."""
# Documenta cómo arrancar la interfaz desde una terminal.
# 1. HERRAMIENTAS GENERALES
# csv e io generan exportaciones en memoria; json serializa informes y evidencia.
# sha256 identifica contenido; os permite configurar rutas y Path maneja archivos.
import csv
# Importa csv para escribir catalogo.csv, la tabla descargable con una fila por canción de la Biblioteca de Armonía.
from hashlib import sha256
# Importa SHA-256 para identificar archivos y detectar cambios en las condiciones de una lista.
import io
# Importa io para preparar el texto de catalogo.csv en la memoria del servidor antes de ofrecerlo con el botón «Exportar etiquetas CSV».
import json
# Permite leer informes JSON y convertir datos de búsqueda a texto JSON.
import os
# Da acceso a las variables de entorno que pueden cambiar las rutas de trabajo.
from pathlib import Path
# Importa Path para construir rutas y trabajar con archivos y carpetas.

# Streamlit dibuja controles en el navegador y ejecuta este archivo en su servidor Python.
# Cada interacción puede volver a ejecutar el script de arriba abajo.
# El estado temporal entre ejecuciones se conserva explícitamente en st.session_state.
import streamlit as st
# Usa el nombre corto st para llamar a los controles de Streamlit.

# 2. LÓGICA COMPARTIDA CON ESCRITORIO
# GENRES contiene claves internas; LABELS sus nombres visibles. analyze extrae las
# características acústicas; MySQL guarda datos; eligibility aplica reglas;
# search construye listas; load_model y predict cargan y consultan IA ya entrenada.
# Este archivo no ejecuta fit ni reentrena al confirmar una etiqueta.
from music_ai import GENRES, LABELS
# Importa los géneros admitidos y sus nombres de presentación.
from music_ai.audio import analyze, MAX_BYTES
# Importa el analizador de audio y el límite de tamaño por archivo.
from music_ai.knowledge import open_catalog, eligibility
# Importa el acceso al catálogo y la función que decide qué canciones pueden participar en una lista.
from music_ai.search import Track, Request, search
# Importa las estructuras que describen una pista y una solicitud, y el algoritmo que busca listas.
from music_ai.training import load_model, predict
# Importa las funciones para abrir el modelo entrenado y obtener una predicción.

# 3. RUTAS Y CATÁLOGO
# La raíz se obtiene del archivo, no de la carpeta desde donde se inicia Python.
# Las variables de entorno permiten usar otros datos o modelos, por ejemplo en pruebas.
ROOT = Path(__file__).resolve().parent
# Localiza la carpeta que contiene este archivo aunque se abra desde otra ubicación.
DATA = Path(os.environ.get("MUSIC_AI_DATA_DIR", str(ROOT / "data")))
# Elige la carpeta configurada para datos o utiliza la subcarpeta data del proyecto.
MODEL_PATH = Path(os.environ.get("MUSIC_AI_MODEL_PATH", str(ROOT / "models" / "genre_model.joblib")))
# Localiza el modelo indicado por el entorno o el archivo joblib predeterminado.
catalog = open_catalog(DATA)
# Abre la biblioteca musical. En este proyecto data/database.json selecciona MySQL y las canciones se consultan en la tabla songs de la base armonia.
# 4. PRESENTACIÓN GENERAL
# El CSS solo controla la apariencia. Los botones y formularios siguen siendo
# controles nativos de Streamlit, con su estado y validaciones en Python.
st.set_page_config(page_title="Armonía", page_icon=":material/graphic_eq:", layout="wide", initial_sidebar_state="collapsed")
# Configura el título de la pestaña, el icono, el ancho de la página y la barra lateral inicial.
st.markdown("<style>" + (ROOT / "assets" / "web.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
# Lee la hoja de estilos y la inserta en la página para aplicar el diseño de Armonía.
# El menú se abre con un botón real, operable también con teclado.
nav, brand = st.columns([1, 7])
# Reserva una columna estrecha para el menú y otra siete veces mayor para la marca.
with nav:
# Coloca los controles de este bloque dentro de la columna del menú.
    if st.button("☰", help="Abrir o cerrar el menú", key="menu_toggle"):
    # Detecta el clic en el botón de tres líneas; la clave identifica ese botón entre ejecuciones.
        st.session_state["menu_open"] = not st.session_state.get("menu_open", False)
        # Invierte el estado del menú; al iniciar por primera vez se considera cerrado.
with brand:
# Coloca el contenido siguiente en la columna de la marca.
    st.markdown('<div class="brand">armonía<span>CIRCO DIGITAL</span></div>', unsafe_allow_html=True)
    # Dibuja el nombre de la aplicación y del equipo con las clases visuales del CSS.
if st.session_state.get("menu_open", False):
# Muestra las opciones de navegación cuando el menú está abierto.
    if "page" not in st.session_state:
    # Comprueba si hace falta inicializar el control de selección de página.
        st.session_state["page"] = st.session_state.get("selected_page", "Biblioteca")
        # Recupera la sección elegida anteriormente o empieza en Biblioteca.
    with st.container(border=True):
    # Agrupa las opciones de navegación dentro de un recuadro.
        st.radio("Navegación", ["Biblioteca", "Añadir música", "Crear una lista", "Entrenamiento y proyecto"], key="page", horizontal=True)
        # Presenta cuatro opciones horizontales y guarda la elegida bajo la clave page.
        st.session_state["selected_page"] = st.session_state["page"]
        # Recuerda la sección elegida incluso después de cerrar el menú.
page = st.session_state.get("selected_page", "Biblioteca")
# Obtiene la sección que se va a dibujar en esta ejecución.
# Carga el artefacto local de confianza y verifica compatibilidad del extractor y
# scikit-learn. Si no existe o falla, la interfaz informa y permite etiquetar sin IA.
model = None
# Inicializa la variable que más adelante podrá contener el modelo cargado.
if page in ("Añadir música", "Entrenamiento y proyecto") and MODEL_PATH.exists():
# Comprueba que la sección necesita el modelo y que el archivo existe en disco.
    try:
    # Inicia un bloque que permite informar de un fallo al abrir el modelo.
        model = load_model(MODEL_PATH)
        # Carga la versión local del modelo para poder consultarla.
    except Exception as exc:
    # Recoge la excepción y permite acceder al detalle del fallo mediante exc.
        st.error(f"No se pudo cargar el modelo local: {exc}")
        # Muestra al usuario el motivo por el que falló la carga.
# Una sola vista se dibuja en cada ejecución. El menú conserva la selección.
# 5. IMPORTACIÓN DE AUDIO
# Primero se seleccionan archivos; solo el botón Analizar y guardar inicia el trabajo.
if page == "Añadir música":
# Activa los controles de importación cuando se eligió Añadir música.
    st.caption("IMPORTAR")
    # Muestra la etiqueta pequeña que identifica la sección de importación.
    st.header("Añadir música")
    # Dibuja el encabezado principal del formulario de carga.
    st.write("Analiza WAV, MP3 o FLAC de entre 3 segundos y 30 minutos. Conservaremos una copia local; el archivo original no cambia.")
    # Explica los formatos, el intervalo de duración y la copia local de cada audio.
    if model is None:
    # Comprueba si la importación se hará sin clasificador disponible.
        st.info("Primera etapa: analiza y organiza tus canciones. La clasificación automática estará disponible cuando entrenemos con un conjunto etiquetado.")
        # Informa de la posibilidad de analizar archivos y revisar el género manualmente.
    # El cargador entrega objetos de archivo en memoria y admite selección múltiple.
    files = st.file_uploader("Archivos de audio", type=["wav", "mp3", "flac"], accept_multiple_files=True)
    # Recoge uno o varios archivos elegidos en el navegador con las extensiones admitidas.
    if st.button("Analizar y guardar", disabled=not files, type="primary"):
    # Inicia el procesamiento al pulsar el botón, que solo se habilita al seleccionar archivos.
        library = DATA / "library"
        # Construye la ruta de la carpeta donde se guarda la copia de cada canción.
        library.mkdir(parents=True, exist_ok=True)
        # Crea esa carpeta y sus directorios superiores si hace falta.
        # Consulta los identificadores existentes para evitar importar contenido repetido.
        existing = set()  # Track this batch; query existing IDs using the primary key.
        # Prepara un conjunto de identificadores para detectar repeticiones dentro de este lote.
        for uploaded in files:
        # Recorre los archivos seleccionados para procesarlos individualmente.
            # Lee bytes y obtiene un identificador SHA256. Detecta archivos idénticos,
            # no necesariamente dos codificaciones diferentes de la misma canción.
            payload = uploaded.getvalue()
            # Obtiene el contenido binario del archivo enviado por el navegador.
            song_id = sha256(payload).hexdigest()
            # Calcula una huella hexadecimal del contenido, independiente del nombre del archivo.
            # Valida tamaño, duplicados y extensión antes de procesar. continue salta
            # solo el archivo rechazado y deja continuar los demás.
            if len(payload) > MAX_BYTES:
            # Compara la cantidad de bytes recibidos con el límite configurado.
                st.error(f"{uploaded.name}: supera 100 MB.")
                # Indica qué archivo supera el límite de 100 MB.
                continue
                # Pasa al siguiente archivo del lote después de detectar el exceso de tamaño.
            if song_id in existing or catalog.contains(song_id):
            # Busca la huella tanto en el lote actual como en las canciones ya guardadas.
                st.info(f"{uploaded.name}: ya está en tu biblioteca.")
                # Informa de que la biblioteca ya contiene ese audio.
                continue
                # Omite el duplicado y continúa con el siguiente archivo.
            suffix = Path(uploaded.name).suffix.lower()
            # Extrae la extensión y la convierte a minúsculas para compararla.
            if suffix not in (".wav", ".mp3", ".flac"):
            # Verifica que la extensión corresponda a uno de los tres formatos permitidos.
                st.error("Formato no admitido.")
                # Avisa que el tipo de archivo recibido queda fuera de los formatos admitidos.
                continue
                # Salta el archivo de formato incompatible.
            # Guarda una copia con nombre basado en hash para evitar colisiones de nombres.
            # El nombre original se conserva como dato del catálogo.
            path = library / (song_id + suffix)
            # Forma el nombre de la copia local con la huella y la extensión.
            path.write_bytes(payload)
            # Escribe los bytes recibidos en esa ruta.
            try:
            # Inicia el procesamiento que puede fallar al analizar o leer el audio.
                with st.spinner(f"Analizando {Path(uploaded.name).name}…"):
                # Muestra un indicador de actividad con el nombre del archivo que se analiza.
                    # analyze devuelve vector, duración e índice acústico; predict usa el vector
                    # y el modelo actual. catalog.add persiste la canción y su predicción original,
                    # pero deja pendiente confirmed_genre hasta la revisión humana.
                    result = analyze(path)
                    # Extrae la duración, la intensidad y el vector de características del audio.
                    prediction = predict(model, result.vector) if model else None
                    # Usa el vector para sugerir un género si hay un modelo cargado.
                    catalog.add(song_id, Path(uploaded.name).name, path, result, prediction)
                    # Guarda en el catálogo la identidad, el nombre, la ruta, el análisis y la sugerencia.
                existing.add(song_id)
                # Marca el archivo como procesado para reconocerlo si se repite en el lote.
                st.success(f"{Path(uploaded.name).name}: guardada · {result.duration:.1f} segundos.")
                # Muestra el nombre y la duración redondeada a una décima de segundo.
                if prediction:
                # Comprueba si se obtuvo una sugerencia automática para esta canción.
                    st.write(f"Género sugerido: **{LABELS[prediction['genre']]}**. Confírmalo en Biblioteca.")
                    # Traduce el género a su nombre visible y dirige al usuario a Biblioteca para revisarlo.
                else:
                # Atiende el caso en que se guardó el audio sin predicción.
                    st.write("Sin sugerencia automática: falta entrenar el modelo. Puedes asignar una etiqueta en Biblioteca.")
                    # Indica dónde asignar manualmente una etiqueta a la canción.
            # Ante estos errores de validación o archivo, elimina la copia de esta importación
            # y muestra el motivo. No modifica el archivo original del usuario.
            except (ValueError, OSError) as exc:
            # Captura errores de validación o del sistema de archivos durante el procesamiento.
                path.unlink(missing_ok=True)
                # Borra la copia local del archivo fallido si todavía existe.
                st.error(f"{uploaded.name}: {exc}")
                # Muestra el nombre del archivo y el error para poder identificar el problema.

# 6. LECTURA Y REVISIÓN
# Recarga el catálogo tras la importación; lookup permite localizar cada registro
# por su id sin depender de que los títulos sean únicos.
# Importar y consultar el informe no necesitan descargar la biblioteca completa.
songs = catalog.songs(include_features=False) if page in ("Biblioteca", "Crear una lista") else []
# Consulta las canciones solo en Biblioteca y Crear una lista, omitiendo sus vectores para reducir la carga.
lookup = {s["id"]: s for s in songs}
# Construye un diccionario que permite encontrar una canción por su identificador.
if page == "Biblioteca":
# Activa la vista de consulta y revisión de canciones.
    st.caption("TU COLECCIÓN")
    # Añade el rótulo de colección sobre el encabezado de la biblioteca.
    st.header("Biblioteca")
    # Muestra el título de la colección.
    total_minutes = sum(s["duration"] for s in songs) / 60
    # Suma las duraciones en segundos y convierte el resultado a minutos.
    pending = sum(not s["confirmed_genre"] for s in songs)
    # Cuenta las canciones cuyo género sigue pendiente de confirmación.
    c1, c2, c3 = st.columns(3)
    # Distribuye los tres indicadores del catálogo en columnas.
    c1.metric("Canciones", len(songs))
    # Muestra el número de registros de canciones.
    c2.metric("Minutos de música", f"{total_minutes:,.0f}")
    # Muestra la duración acumulada en minutos redondeada al entero más cercano.
    c3.metric("Por revisar", pending)
    # Muestra cuántas canciones requieren revisión.
    st.write("Escucha, revisa el género y prepara tu próxima lista.")
    # Describe las acciones disponibles en esta pantalla.
    if not songs:
    # Detecta si la consulta devolvió una colección vacía.
        st.info("Tu colección está vacía. Abre el menú ☰ y elige Añadir música para importar tus archivos.")
        # Explica cómo importar el primer archivo mediante el menú.
    else:
    # Dibuja las herramientas de revisión cuando hay canciones disponibles.
        # Buscar y filtrar cambia solo la vista; el catálogo completo permanece intacto.
        left, right = st.columns([3, 1])
        # Reserva más espacio para el buscador que para el filtro de género.
        query = left.text_input("Buscar una canción", placeholder="Nombre de la canción")
        # Lee el texto que el usuario quiere buscar en los nombres de las canciones.
        genre_filter = right.selectbox("Filtrar género", ["Todos", "Pendientes", *LABELS.values()])
        # Ofrece todos los géneros, las canciones pendientes o la colección completa.
        visible = [s for s in songs if query.casefold() in s["name"].casefold()
        # Empieza a seleccionar las canciones cuyo nombre contiene la búsqueda, ignorando mayúsculas.
                   and (genre_filter == "Todos" or (genre_filter == "Pendientes" and not s["confirmed_genre"])
                   # Comprueba si el filtro admite todos los géneros o pide canciones pendientes de revisión.
                        or LABELS.get(s["confirmed_genre"]) == genre_filter)]
                        # También acepta una canción si su género confirmado coincide con el filtro elegido.
        st.dataframe([{"Canción": s["name"], "Duración (s)": round(s["duration"], 1),
        # Empieza a construir la tabla con el nombre y los segundos de cada canción visible.
                       "Sugerencia": LABELS[s["prediction"]["genre"]] if s["prediction"] else "Sin modelo al importar",
                       # Añade el género sugerido durante la importación o indica que faltaba el modelo.
                       "Confirmación": LABELS.get(s["confirmed_genre"], "Pendiente"),
                       # Añade el nombre del género confirmado o la etiqueta Pendiente.
                       "Índice acústico": round(s["intensity"], 3)} for s in visible], hide_index=True, width="stretch")
                       # Añade la intensidad con tres decimales y muestra las filas filtradas a todo el ancho.
        # Selecciona por id, pero presenta nombre y hash abreviado para facilitar la revisión.
        chosen = st.selectbox("Canción para revisar", list(lookup), format_func=lambda k: lookup[k]["name"] + " · " + k[:6])
        # Selecciona una canción por identificador mostrando su nombre y seis caracteres de la huella.
        song = lookup[chosen]
        # Recupera del diccionario los datos correspondientes a la selección.
        # Do not read and transmit a complete audio file just to open the library.
        # Only prepare the player after an explicit request for this song.
        if st.button("Escuchar canción", key=f"listen_{chosen}"):
        # Prepara la reproducción solo al pulsar el botón asociado con esta canción.
            if Path(song["path"]).exists():
            # Comprueba que la ruta del audio seleccionado siga existiendo.
                st.audio(song["path"])
                # Inserta el reproductor con el audio guardado en esa ruta.
            else:
            # Atiende el caso en que la copia local ya no se encuentra.
                st.warning("La copia de audio ya no está disponible.")
                # Avisa al usuario de que no se puede reproducir ese archivo.
        # El control muestra la etiqueta confirmada o queda vacío. Su clave incluye el id
        # para que la selección de una canción no se confunda con la de otra.
        genre = st.selectbox("Género principal confirmado", GENRES, format_func=LABELS.get,
        # Crea el selector de género confirmado con los cuatro nombres visibles.
                             index=GENRES.index(song["confirmed_genre"]) if song["confirmed_genre"] else None,
                             # Preselecciona el género ya confirmado o deja vacía la selección inicial.
                             key=f"genre_{chosen}",
                             # Distingue el estado de este selector del de las demás canciones mediante su identificador.
                             placeholder="Selecciona después de revisar")
                             # Muestra una indicación mientras el usuario todavía no haya elegido un género.
        # Guarda la confirmación y su historial en MySQL; no reemplaza la predicción.
        # Descarta la lista temporal y vuelve a ejecutar la página para reflejar la corrección.
        if st.button("Guardar confirmación", disabled=genre is None):
        # Habilita el guardado al seleccionar un género y detecta el clic de confirmación.
            catalog.confirm(chosen, genre)
            # Actualiza la etiqueta de la canción y registra la revisión en el catálogo.
            st.session_state.pop("playlist", None)
            # Descarta la lista temporal anterior porque sus géneros podrían haber cambiado.
            st.rerun()
            # Ejecuta de nuevo la interfaz para reflejar la confirmación recién guardada.
        # Hace una inferencia nueva con el vector guardado. Solo muestra la respuesta:
        # no sobrescribe la sugerencia histórica ni confirma automáticamente el género.
        if MODEL_PATH.exists() and st.button("Consultar modelo actual para esta canción"):
        # Habilita la consulta del modelo existente y espera a que el usuario la solicite.
            model = load_model(MODEL_PATH)
            # Recupera el modelo vigente al solicitar la predicción.
            st.json(predict(model, catalog.get(chosen)["features"]))
            # Lee las características de esta canción, calcula su predicción actual y muestra los campos en JSON.
            st.caption("Consulta nueva; la predicción original y la confirmación se conservan.")
            # Aclara que la consulta muestra un resultado nuevo sin sustituir el registro original.
        # Muestra la diferencia entre predicción original y decisiones posteriores del usuario.
        with st.expander("Predicción original e historial de correcciones"):
        # Crea un panel desplegable para revisar la predicción guardada y las correcciones.
            if st.button("Cargar historial", key=f"history_{chosen}"):
            # Espera a que se solicite el historial de la canción seleccionada.
                st.json({"prediction": song["prediction"], "history": catalog.history(chosen)})
                # Consulta las revisiones de esa canción y las presenta junto con la sugerencia original.
        # Exporta el catálogo a CSV en memoria. Incluye etiquetas y versión del modelo;
        # no incluye el audio. utf-8-sig facilita leer los acentos en hojas de cálculo.
        output = io.StringIO(newline="")
        # Crea un espacio temporal en memoria donde se escribirán los encabezados y una fila por canción para la descarga de catalogo.csv.
        writer = csv.writer(output)
        # Prepara el escritor que aplica las reglas de separación y comillas de CSV.
        writer.writerow(["id", "name", "duration", "confirmed_genre", "suggested_genre", "model_id"])
        # Escribe la fila de encabezados que describe las seis columnas exportadas.
        for s in songs:
        # Recorre todo el catálogo para incluir cada registro en la descarga.
            # Evita interpretar nombres como fórmulas al abrir el CSV en una hoja de cálculo.
            name = "'" + s["name"] if s["name"].startswith(("=", "+", "-", "@")) else s["name"]
            # Ante ciertos prefijos de fórmula, antepone una comilla al nombre para exportarlo como texto.
            writer.writerow([s["id"], name, s["duration"], s["confirmed_genre"],
            # Empieza una fila con la huella, el nombre preparado, la duración y el género confirmado.
                             s["prediction"]["genre"] if s["prediction"] else "",
                             # Añade la sugerencia histórica o un campo vacío si no hubo predicción.
                             s["prediction"]["model_id"] if s["prediction"] else ""])
                             # Termina la fila con la versión del modelo o con un campo vacío.
        st.download_button("Exportar etiquetas CSV", output.getvalue().encode("utf-8-sig"), "catalogo.csv", "text/csv")
        # Ofrece el archivo catalogo.csv con codificación UTF-8 y marca inicial para hojas de cálculo.

# 7. BÚSQUEDA DE LISTAS
# Los controles definen duración, tolerancia, curva acústica, géneros y exclusiones.
# beam conserva un haz limitado de candidatos; exact enumera hasta 10 elegibles.
if page == "Crear una lista":
# Activa el formulario de búsqueda cuando se elige Crear una lista.
    st.caption("SELECCIÓN PERSONALIZADA")
    # Muestra el rótulo que identifica la selección personalizada.
    st.header("Crear una lista")
    # Dibuja el título de la sección de listas.
    st.write("Elige cuánto tiempo tienes y qué géneros quieres escuchar.")
    # Indica que la búsqueda se configura mediante duración y géneros.
    c1, c2, c3 = st.columns(3)
    # Alinea en tres columnas los controles básicos de la solicitud.
    minutes = c1.number_input("Duración objetivo (minutos)", min_value=0.1, max_value=180.0, value=25.0, step=1.0)
    # Lee la duración deseada en minutos, con 25 como valor inicial.
    tolerance = c2.number_input("Tolerancia (segundos)", min_value=0, max_value=600, value=15)
    # Lee el margen de duración aceptable en segundos, inicialmente 15.
    progression = c3.selectbox("Progresión de intensidad", ["creciente", "decreciente", "constante"])
    # Recoge la forma deseada de la intensidad a lo largo de la lista.
    allowed = st.multiselect("Géneros permitidos", GENRES, default=list(GENRES), format_func=LABELS.get)
    # Permite marcar varios géneros y comienza con todos seleccionados.
    excluded = st.multiselect("Excluir canciones", list(lookup), format_func=lambda k: lookup[k]["name"])
    # Recoge los identificadores que deben excluirse, mostrando sus nombres para elegirlos.
    method = st.radio("Método", ["beam", "exact"], format_func=lambda m: "Búsqueda por haz" if m == "beam" else "Enumeración exacta (máximo 10 elegibles)", horizontal=True)
    # Permite elegir entre búsqueda por haz y enumeración exacta.
    width = st.slider("Ancho del haz", 1, 100, 25, disabled=method != "beam")
    # Ajusta cuántas alternativas conserva el haz; el deslizador se desactiva en el método exacto.
    st.caption("El índice acústico resume amplitud de fragmentos: no mide emoción. Solo se incluyen etiquetas confirmadas.")
    # Explica el significado acústico de la intensidad y el requisito de confirmación.
    # Cada canción se evalúa con reglas explícitas: archivo disponible, confirmación,
    # género permitido y ausencia de exclusión. Conserva también el motivo de la regla.
    decisions = [(s, *eligibility(s, allowed, excluded)) for s in songs]
    # Evalúa cada canción y reúne sus datos, su elegibilidad y la razón de la decisión.
    # Esta huella representa etiquetas/elegibilidad y parámetros actuales. Permite
    # detectar cambios frente a la lista que quedó guardada en la sesión del navegador.
    snapshot = sha256(json.dumps({"songs": [(s["id"], s["confirmed_genre"], ok) for s, ok, _ in decisions],
    # Empieza la huella de la búsqueda incluyendo identificadores, confirmaciones y elegibilidad.
                                 "request": [minutes, tolerance, progression, allowed, excluded, method, width]}, sort_keys=True).encode()).hexdigest()
                                 # Incorpora los parámetros, ordena las claves JSON y calcula una huella reproducible.
    if st.button("Generar lista", disabled=not songs, type="primary"):
    # Atiende la orden de generar una lista cuando existen canciones en el catálogo.
        try:
        # Prepara el manejo de errores en las restricciones o los parámetros de búsqueda.
            # Convierte solo las canciones elegibles a Track: id, duración e índice acústico.
            # Request recibe segundos, por eso se multiplican los minutos por 60.
            tracks = [Track(s["id"], s["duration"], s["intensity"]) for s, ok, _ in decisions if ok]
            # Convierte las canciones admitidas en pistas con identificador, duración e intensidad.
            with st.spinner("Explorando combinaciones…"):
            # Muestra un indicador mientras el algoritmo explora combinaciones.
                result = search(tracks, Request(minutes * 60, tolerance, progression), method, width)
                # Pasa el objetivo en segundos, la tolerancia, la progresión, el método y el ancho al buscador.
            # Conserva huella y resultado entre ejecuciones de Streamlit. Es estado de sesión,
            # no una lista persistida automáticamente en MySQL.
            st.session_state["playlist"] = (snapshot, result.to_dict())
            # Conserva el resultado y la huella de sus condiciones en la sesión actual.
        except ValueError as exc:
        # Captura una solicitud inválida comunicada por el buscador.
            st.error(str(exc))
            # Muestra el detalle del error para que el usuario pueda ajustar los controles.
    stored = st.session_state.get("playlist")
    # Recupera el último resultado de búsqueda guardado en esta sesión.
    # Solo muestra/exporta el resultado mientras su huella coincida con las condiciones.
    # Los estados distinguen optimalidad, solución sin garantía e imposibilidad comprobada
    # de una búsqueda limitada que simplemente no encontró solución.
    if stored and stored[0] == snapshot:
    # Comprueba que exista un resultado y que corresponda a las condiciones actuales.
        result = stored[1]
        # Obtiene el diccionario de la lista guardada, situado tras su huella.
        statuses = {"optimal": "Solución óptima para este catálogo y costo.", "found": "Lista válida encontrada; no se garantiza que sea la mejor.",
        # Define los mensajes para una solución óptima y una solución válida sin garantía de optimalidad.
                    "infeasible": "No existe una lista que cumpla estas condiciones en el catálogo elegible.",
                    # Define el mensaje para una búsqueda que pudo demostrar la inviabilidad.
                    "not_found": "No se encontró una lista con este presupuesto. Podría existir una solución."}
                    # Define el mensaje para una búsqueda limitada que terminó sin hallar una lista.
        st.info(statuses[result["status"]])
        # Muestra la explicación que corresponde al estado devuelto por el algoritmo.
        if result["ids"]:
        # Comprueba si el resultado incluye al menos un identificador de canción.
            st.metric("Duración de la lista", f"{result['duration'] / 60:.2f} min")
            # Convierte la duración total a minutos y la muestra con dos decimales.
            st.dataframe([{"Orden": i + 1, "Canción": lookup[k]["name"], "Segundos": lookup[k]["duration"],
            # Empieza la tabla de resultados con posición, nombre y segundos de cada pista.
                           "Género": LABELS[lookup[k]["confirmed_genre"]]} for i, k in enumerate(result["ids"])], hide_index=True)
                           # Añade el género confirmado siguiendo el orden de los identificadores del resultado.
            # M3U8 es texto UTF-8 con rutas a los audios en orden; no empaqueta las canciones.
            m3u = "#EXTM3U\n" + "\n".join(lookup[k]["path"] for k in result["ids"]) + "\n"
            # Construye el texto M3U8 con su cabecera y una ruta de audio por línea.
            st.download_button("Exportar lista M3U8", m3u.encode("utf-8"), "lista.m3u8", "audio/x-mpegurl")
            # Ofrece descargar la lista de rutas en el archivo lista.m3u8.
        # Exporta condiciones, reglas y resultado en JSON para documentar la búsqueda,
        # incluso cuando no se encontró una lista. No cambia la base de datos.
        st.download_button("Exportar evidencia de búsqueda", json.dumps({"request": {"target_seconds": minutes * 60, "tolerance": tolerance, "progression": progression, "allowed": allowed, "excluded": excluded, "width": width}, "catalog_snapshot": snapshot, "result": result,
        # Prepara la descarga de las condiciones de búsqueda, la huella del catálogo y el resultado.
            "candidates": [{"id": s["id"], "duration": s["duration"], "intensity": s["intensity"], "genre": s["confirmed_genre"], "eligible": ok, "rule": why} for s, ok, why in decisions]}, ensure_ascii=False, indent=2), "busqueda.json", "application/json")
            # Incluye los candidatos y sus reglas, y descarga el JSON con sangría y acentos legibles.
        with st.expander("Estadísticas de búsqueda"):
        # Crea el panel desplegable de estadísticas de la búsqueda.
            st.json(result)
            # Presenta el diccionario del resultado, incluidos sus contadores y estado.
    elif stored:
    # Detecta un resultado anterior cuya huella ya no corresponde a la solicitud.
        st.caption("Cambiaste las condiciones o el catálogo. Genera otra lista para ver un resultado actualizado.")
        # Pide volver a generar la lista al cambiar sus condiciones.
    with st.expander("Reglas de elegibilidad"):
    # Crea el panel que explica qué canciones pueden participar.
        # Construye una tabla de presentación: los redondeos no alteran los datos guardados.
        st.dataframe([{"Canción": s["name"], "Elegible": ok, "Razón": why} for s, ok, why in decisions], hide_index=True)
        # Muestra por canción la decisión de elegibilidad y el motivo de la regla aplicada.

# 8. MODELO Y PROYECTO
# Se presentan cifras del informe guardado, no métricas inventadas ni calculadas
# con el catálogo. Esta pantalla consulta resultados; no inicia entrenamiento.
if page == "Entrenamiento y proyecto":
# Activa la página que explica el modelo y presenta sus resultados de entrenamiento.
    st.caption("DETRÁS DE LA MÚSICA")
    # Muestra el rótulo introductorio de la sección.
    st.header("Entrenamiento y proyecto")
    # Dibuja el encabezado del informe de entrenamiento.
    st.write("Cómo aprende Armonía, qué modelo utiliza y hasta dónde llegan sus resultados.")
    # Presenta el propósito de los datos y explicaciones que aparecen a continuación.
    report_path = MODEL_PATH.with_suffix(".report.json")
    # Deduce la ruta del informe sustituyendo el sufijo del modelo por .report.json.
    training_report = {}
    # Prepara un diccionario vacío para cuando falte el informe.
    if report_path.exists():
    # Comprueba si hay un informe de entrenamiento en disco.
        try:
        # Inicia la lectura con manejo de errores de archivo y formato.
            training_report = json.loads(report_path.read_text(encoding="utf-8"))
            # Lee el informe como texto UTF-8 y convierte el JSON en datos de Python.
        except (ValueError, OSError):
        # Captura un archivo ilegible o un JSON que no pueda interpretarse.
            st.warning("No se pudo leer el informe de entrenamiento.")
            # Informa de que el reporte no se pudo cargar.
    names = {"random_forest": "Bosque aleatorio", "svm": "SVM", "baseline": "Clase mayoritaria"}
    # Asocia las claves internas de los clasificadores con nombres para la pantalla.
    counts = training_report.get("counts", {})
    # Recupera los conteos por partición y género, o un diccionario vacío.
    with st.container(border=True):
    # Agrupa la información del modelo activo dentro de un recuadro.
        st.subheader("Modelo en uso")
        # Muestra el subtítulo del resumen del modelo.
        c1, c2, c3 = st.columns(3)
        # Reserva tres columnas para el clasificador, los géneros y las características.
        c1.metric("Clasificador", names.get(training_report.get("selected"), "Sin informe") if model else "No disponible")
        # Muestra qué clasificador se seleccionó según el informe y si hay modelo disponible.
        c2.metric("Géneros", "4")
        # Indica cuántos géneros admite esta aplicación.
        c3.metric("Características de audio", len(model.get("feature_names", [])) if model else "—")
        # Cuenta los nombres de características definidos en el modelo cargado.
        if model:
        # Comprueba si hay un modelo del que se pueda mostrar la versión.
            st.caption("Modelo cargado · Versión " + model["model_id"])
            # Muestra el identificador de la versión cargada.
        else:
        # Atiende el caso en que el modelo no está disponible.
            st.info("Todavía no hay un modelo disponible. Puedes organizar y etiquetar canciones manualmente.")
            # Informa de que todavía se puede organizar y etiquetar la colección manualmente.
        st.write("Al importar una canción se consulta este modelo. Confirmar un género guarda tu decisión; no vuelve a entrenarlo.")
        # Explica la diferencia entre solicitar una predicción y confirmar el género.
    st.subheader("Del audio a una sugerencia")
    # Introduce el recorrido del audio hasta su revisión.
    for col, title, body in zip(st.columns(3),
    # Empareja cada columna con un título y una explicación para dibujar las tres etapas.
        ["01 · Analizar", "02 · Comparar", "03 · Revisar"],
        # Proporciona los títulos de análisis, comparación y revisión en su orden de aparición.
        ["Se analizan hasta seis fragmentos de diez segundos. Se resumen timbre, contenido tonal y ritmo.",
        # Describe la extracción de características en los fragmentos como primera etapa.
         "El modelo compara esas características con patrones aprendidos de ejemplos etiquetados como pop, reguetón, rock o rap.",
         # Explica en la segunda tarjeta la comparación con patrones aprendidos.
         "Recibes una sugerencia. Escuchas la canción y confirmas o corriges el género en Biblioteca."]):
         # Completa la tercera explicación con la revisión humana y cierra la lista de textos.
        with col:
        # Dirige la tarjeta actual a la columna que le corresponde.
            with st.container(border=True):
            # Enmarca el contenido de la tarjeta dentro de esa columna.
                st.subheader(title)
                # Muestra el título asignado a la etapa actual.
                st.write(body)
                # Muestra la explicación asociada con ese título.
    st.subheader("Cómo se entrenó")
    # Introduce las particiones empleadas en el entrenamiento.
    st.write("Se compararon SVM y bosque aleatorio con una referencia que siempre elige la clase más frecuente. Los ejemplos se separaron según su función:")
    # Describe los dos modelos candidatos y la referencia basada en la clase más frecuente.
    st.dataframe([{"Etapa": label, "Ejemplos": sum(counts.get(key, {}).values()), "Para qué se utiliza": purpose}
    # Prepara cada fila sumando los ejemplos de los géneros de una misma etapa.
        for key, label, purpose in [("train", "Entrenamiento", "Ajustar los patrones del modelo"),
        # Asocia la clave train con el ajuste del modelo.
            ("validation", "Validación", "Comparar candidatos y seleccionar uno"),
            # Asocia la clave validation con la comparación y selección de candidatos.
            ("test", "Prueba", "Evaluar al modelo seleccionado")]], hide_index=True, width="stretch")
            # Asocia test con la evaluación final y muestra las tres filas sin índice.
    st.caption("Un ejemplo puede ser una canción o un fragmento de mezcla. Varios fragmentos del mismo origen no son canciones independientes.")
    # Aclara cómo interpretar el número de fragmentos procedentes de una mezcla.
    validation = training_report.get("validation_f1_macro", {})
    # Recupera el valor de F1 macro de validación de cada candidato.
    if validation:
    # Comprueba si existen métricas de validación que se puedan presentar.
        st.subheader("Comparación en validación")
        # Encabeza la sección de comparación de modelos.
        st.dataframe([{"Modelo": names.get(k, k), "F1 macro": round(v, 3)} for k, v in validation.items()], hide_index=True, width="stretch")
        # Muestra una fila por modelo con su nombre y F1 macro redondeado a tres decimales.
        st.caption("F1 macro combina precisión y cobertura de cada género y luego promedia los cuatro por igual. Va de 0 a 1; no es un porcentaje de certeza para una canción.")
        # Explica qué resume F1 macro y cómo interpretar su escala.
    st.subheader("Qué podemos concluir")
    # Introduce las conclusiones y limitaciones de la prueba.
    test = training_report.get("test", {})
    # Recupera las métricas de prueba del informe o un diccionario vacío.
    support = int(test.get("macro avg", {}).get("support", 0))
    # Lee el número de ejemplos evaluados y lo convierte a entero.
    if support and "accuracy" in test:
    # Verifica que haya ejemplos y un valor de exactitud antes de calcular los aciertos.
        st.write(f"La prueba registrada contiene {support} ejemplos. El modelo acertó {round(test['accuracy'] * support)} de ellos.")
        # Multiplica la exactitud por el total de ejemplos y redondea para mostrar el número de aciertos.
    missing = training_report.get("test_missing_genres", [])
    # Lee la lista de géneros ausentes de la prueba.
    if missing:
    # Comprueba si el informe registró géneros ausentes.
        st.write("La prueba no incluye ejemplos de: " + ", ".join(LABELS.get(g, g) for g in missing) + ".")
        # Convierte sus claves en nombres visibles y los reúne en un mensaje.
    st.warning("Evaluación provisional. La prueba es pequeña y se ha reutilizado; las mezclas pueden compartir canciones con validación o prueba. Estos resultados no demuestran la precisión con música nueva.")
    # Advierte sobre el tamaño de la prueba, su reutilización y el posible solapamiento de obras.
    st.write("El siguiente paso es reunir grabaciones variadas y reservar una prueba nueva de los cuatro géneros, separando obras y artistas. Las puntuaciones actuales requieren revisión humana.")
    # Presenta el siguiente paso experimental y recuerda revisar las sugerencias.
    with st.expander("Proceso de un nuevo entrenamiento"):
    # Abre una sección desplegable con el procedimiento de reentrenamiento.
        st.write("1. Revisar archivos y etiquetas del conjunto de datos.")
        # Muestra el primer paso, revisar el material y sus etiquetas.
        st.write("2. Separar entrenamiento, validación y prueba sin repetir obras entre grupos.")
        # Muestra el segundo paso, separar ejemplos y evitar obras compartidas entre particiones.
        st.write("3. Ejecutar el entrenamiento desde Python. No se inicia al abrir esta página.")
        # Explica que el tercer paso consiste en ejecutar el entrenamiento desde Python.
        st.write("4. Revisar el informe, guardar la versión y comprobarla antes de usarla.")
        # Describe la revisión del informe y la comprobación de la nueva versión.
        st.caption("Ejemplo para un manifiesto completo, con los cuatro géneros en cada partición:")
        # Presenta el comando ilustrativo para un conjunto con las cuatro clases.
        st.code(r".\.venv\Scripts\python.exe -m music_ai.training data\manifest.csv", language="powershell")
        # Muestra el comando como código de PowerShell para que se pueda leer o copiar.
    with st.expander("Informe técnico completo"):
    # Agrupa el informe detallado dentro de un panel desplegable.
        if training_report:
        # Comprueba si se recuperó contenido del informe.
            st.json(training_report)
            # Presenta el informe completo con un visor de datos JSON.
        else:
        # Atiende el caso en que el informe quedó vacío.
            st.info("No hay un informe guardado para este modelo.")
            # Muestra un aviso indicando que falta el informe del modelo.
    with st.expander("Acerca del proyecto"):
    # Agrupa la autoría y los datos del proyecto en un panel desplegable.
        st.write("**Circo digital · Grupo 1907**")
        # Muestra en negrita el equipo y el grupo escolar.
        st.write("Benitez Guerra Alison Aranza · García Muñoz Jesús · Lorrabaquio Rodríguez Joanna Lizbeth · Mendez Camacho Josue")
        # Muestra los nombres de los cuatro integrantes del equipo.
        st.write("Profesor Martin Romero Ugalde")
        # Muestra el nombre del profesor.
        st.write("Armonía integra clasificación supervisada, reglas de revisión y búsqueda de listas. La web y la aplicación de escritorio comparten el catálogo.")
        # Presenta el texto descriptivo sobre las funciones de Armonía y su catálogo.
        st.caption("Almacenamiento: MySQL local")
        # Muestra dónde se guardan las fichas de canciones y las revisiones de género.
