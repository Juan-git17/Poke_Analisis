import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier # ¡NUEVO CLASIFICADOR!
import requests # ¡ASEGÚRATE DE AGREGAR ESTA LÍNEA!

# --- Configuración de la Página ---
st.set_page_config(page_title="Clasificación de Roles (KNN)", layout="wide")
st.title("Clasificación de Roles de Combate 2D Independientes (KNN) ")

# --- Variables Globales y Definiciones ---
N_CLUSTERS = 8
FULL_FEATURE_NAMES = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
DATA_PATH = r"C:\Users\USUARIO\Downloads\pokemon.csv" 
GLOBAL_COLORS = plt.cm.get_cmap('tab10') 
K_NEIGHBORS = 5 

# --- ¡CRUCIAL! LISTAS DE ROLES (DEBES USAR TUS LISTAS CORREGIDAS) ---
CLASSIFIERS = {
    'att_speed': {
        'title': "1. Clasificación (KNN): Velocidad y Ataque",
        'features': ['attack', 'speed'],
        'model_path': 'knn_att_speed_model.pkl',
        'cluster_col': 'cluster_att_speed',
        # ¡IMPORTANTE! Usa tus nombres de roles ya corregidos aquí.
        'roles': [
            'Rol 0: BAJO RENDIMIENTO (Atq/Vel)', 
            'Rol 1: Atacante Rápido Élite', 
            'Rol 2: Speedster (Alta Vel/Bajo Atq)',
            'Rol 3: Ataque Extremo Lento',
            'Rol 4: Balanceado Rápido',
            'Rol 5: Tanque Lento (Error de Proyección)',
            'Rol 6: Velocidad Media/Ataque Bajo',
            'Rol 7: Ataque Medio/Velocidad Baja'
        ]
    },
    'sp_att_sp_def': {
        'title': "2. Clasificación (KNN): Ataque y Defensa Especial",
        'features': ['special-attack', 'special-defense'],
        'model_path': 'knn_sp_att_sp_def_model.pkl',
        'cluster_col': 'cluster_sp_att_sp_def',
        'roles': [
            'Rol 0: BAJO ESPECIAL GENERAL',
            'Rol 1: Muralla Especial Extrema',
            'Rol 2: Cañón de Cristal (Alto Atq/Baja Def)',
            'Rol 3: Defensa Especial Media',
            'Rol 4: Especialista Mixto (Alto Atq/Def)',
            'Rol 5: Ataque Especial Lento/Medio',
            'Rol 6: Defensa Baja/Ataque Medio',
            'Rol 7: Balanceado Especial'
        ]
    },
    'att_def': {
        'title': "3. Clasificación (KNN): Ataque y Defensa Físicas",
        'features': ['attack', 'defense'],
        'model_path': 'knn_att_def_model.pkl',
        'cluster_col': 'cluster_att_def',
        'roles': [
            'Rol 0: POCA RESISTENCIA/ATAQUE',
            'Rol 1: Tanque Físico Puro (Bajo Atq/Alta Def)',
            'Rol 2: Atacante Físico Frágil (Alto Atq/Baja Def)',
            'Rol 3: Defensor/Atacante Mixto (Alto Atq/Def)',
            'Rol 4: Defensa Media/Ataque Bajo',
            'Rol 5: Ataque Medio/Defensa Baja',
            'Rol 6: Ataque Extremo Lento',
            'Rol 7: Defensa Media/Ataque Extremo'
        ]
    }
}


# --- 1. Carga y CLUSTERING INDEPENDIENTE (Sin Cambios) ---

@st.cache_data
def load_and_cluster_data():
    """Carga el dataset y realiza un K-Means independiente para cada par de features."""
    try:
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=FULL_FEATURE_NAMES, inplace=True)
        
        for key, config in CLASSIFIERS.items():
            features = config['features']
            cluster_col = config['cluster_col']
            X_subset = df[features]
            
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
            y_clusters = kmeans.fit_predict(X_subset)
            df[cluster_col] = y_clusters
            
        return df

    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos en la ruta: {DATA_PATH}.")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar o preprocesar los datos: {e}")
        st.stop()

DF_POKEMON = load_and_cluster_data()


# --- 2. Carga/Entrenamiento de Modelos (KNN) (Sin Cambios) ---

@st.cache_resource(hash_funcs={pd.DataFrame: id})
def train_and_save_model(df_clustered, features, cluster_col, model_path):
    """Entrena y guarda un modelo KNN (KNeighborsClassifier) para un subconjunto específico."""
    
    X = df_clustered[features]
    y = df_clustered[cluster_col] 
    
    if os.path.exists(model_path):
        st.sidebar.info(f"Cargando modelo KNN existente para {features[0]} vs {features[1]}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.sidebar.info(f"Entrenando nuevo modelo KNN (K={K_NEIGHBORS}) para {features[0]} vs {features[1]}...")
        model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS) 
        model.fit(X, y)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.sidebar.success("Modelo entrenado y guardado correctamente.")
        
        return model

MODELOS = {}
for key, config in CLASSIFIERS.items():
    MODELOS[key] = train_and_save_model(
        DF_POKEMON, 
        config['features'], 
        config['cluster_col'], 
        config['model_path']
    )


# --- 3. Funciones de Interacción y Visualización (Sin Cambios) ---

def get_pokemon_data_from_api(nombre_o_id: str):
    """
    Consulta la PokeAPI para obtener el nombre oficial, la URL de la imagen y los tipos.
    Retorna (nombre_oficial, url_imagen, tipos).
    """
    try:
        # La API acepta el nombre en minúsculas
        clean_input = str(nombre_o_id).lower().replace(" ", "-")
        url = f"https://pokeapi.co/api/v2/pokemon/{clean_input}"
        response = requests.get(url, timeout=5) # Añadir timeout para evitar cuelgues
        response.raise_for_status()
        data = response.json()
        
        # 1. Obtener Nombre
        pokemon_name = data['name'].title()
        
        # 2. Obtener URL de la Imagen (Usando official-artwork, la mejor calidad)
        # Esto maneja mejor las formas que el link estático
        sprite_url = data['sprites']['other']['official-artwork']['front_default']
        
        # 3. Obtener Tipos
        types = [t['type']['name'].title() for t in data['types']]
        
        return pokemon_name, sprite_url, types
        
    except requests.exceptions.HTTPError:
        # Pokémon no encontrado (Error 404)
        return None, None, None
    except Exception:
        # Otros errores (Red, formato)
        return None, None, None

def get_slider_input(key, features):
    """Crea sliders en la sección principal para las features de un modelo."""
    data = {}
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        feature = features[0]
        min_val, max_val, mean_val = DF_POKEMON[feature].min(), DF_POKEMON[feature].max(), DF_POKEMON[feature].mean()
        data[feature] = st.slider(
            f'🦖 {feature.replace("-", " ").title()}', 
            min_val.item(), max_val.item(), round(mean_val.item()), key=f'{key}_{feature}'
        )
        
    with col_input2:
        feature = features[1]
        min_val, max_val, mean_val = DF_POKEMON[feature].min(), DF_POKEMON[feature].max(), DF_POKEMON[feature].mean()
        data[feature] = st.slider(
            f'🦖 {feature.replace("-", " ").title()}', 
            min_val.item(), max_val.item(), round(mean_val.item()), key=f'{key}_{feature}'
        )
        
    nuevo_dato_np = np.array([[data[f] for f in features]])
    return nuevo_dato_np, data


def crear_seccion_clasificacion(key, config):
    """Maneja la lógica de predicción y visualización para una sección."""
    
    st.subheader(config['title'])
    
    features = config['features']
    modelo = MODELOS[key]
    target_names = config['roles']
    
    nuevo_dato_np, nuevo_dato_dict = get_slider_input(key, features)
    
    prediccion_clase_idx = modelo.predict(nuevo_dato_np)[0]
    
    try:
        prediccion_proba = modelo.predict_proba(nuevo_dato_np)
        confianza = prediccion_proba[0].max() * 100
    except AttributeError:
        confianza = 100.0 

    prediccion_nombre = target_names[prediccion_clase_idx]

    col_pred, col_conf = st.columns(2)
    with col_pred:
        st.markdown(f"**Rol Predicho:** <p style='font-size:24px;'>{prediccion_nombre}</p>", unsafe_allow_html=True)
    with col_conf:
        st.markdown(f"**N° Vecinos (K={K_NEIGHBORS})** <p style='font-size:24px;'>{confianza:.2f}%</p>", unsafe_allow_html=True)
        
    fig = crear_grafico_cluster_individual(DF_POKEMON, features[0], features[1], nuevo_dato_dict, prediccion_nombre, target_names, config['cluster_col'])
    st.pyplot(fig)
    st.markdown("---")


def crear_grafico_cluster_individual(df, x_col, y_col, input_values, prediccion_nombre, target_names, cluster_col):
    """Crea y devuelve un gráfico de Matplotlib para las columnas especificadas."""
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for i, cluster_name in enumerate(target_names):
        subset = df[df[cluster_col] == i]
        color = GLOBAL_COLORS(i) 
        
        ax.scatter(subset[x_col], subset[y_col], c=[color], label=cluster_name, alpha=0.7, edgecolors='w', s=40)

    x_nuevo = input_values[x_col]
    y_nuevo = input_values[y_col]

    ax.scatter(
        x_nuevo, y_nuevo, marker='*', c='red', edgecolors='black', 
        s=250, label=f'Nuevo Pokémon ({prediccion_nombre})', zorder=5 
    )

    ax.set_xlabel(x_col.replace("-", " ").title())
    ax.set_ylabel(y_col.replace("-", " ").title())
    ax.set_title(f'Proyección: {x_col.title()} vs {y_col.title()}')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


# --- 4.0. HERRAMIENTA DE MAEPEO (¡MODIFICADO!) ---

st.header("Herramienta de Mapeo de Roles (Centroides y Nombres)")
st.success("""
    La tabla ahora incluye el **Nombre del Rol** a la derecha de las estadísticas. 
    Usa esta tabla para verificar que tu nombre de Rol (ej: 'Atacante Rápido') esté asignado al Índice de Cluster (0-7) correcto (ej: el que tiene alta Ataque/Velocidad).
""")

def generar_medias_clusters(df, cluster_col, features, target_names):
    """Calcula la media de las features para cada cluster y añade la columna de rol."""
    # 1. Calcular las medias (Centroides)
    medias = df.groupby(cluster_col)[features].mean()
    
    # 2. Convertir a DataFrame y restablecer el índice para acceder al ID del Cluster
    medias_df = medias.reset_index().rename(columns={cluster_col: 'ID_Cluster'})
    
    # 3. Mapear los nombres de los roles a la tabla
    # Creamos una columna temporal de los nombres de roles
    rol_mapping = {i: name for i, name in enumerate(target_names)}
    
    # Asignamos el nombre del rol usando el ID_Cluster (que es el índice K-Means)
    medias_df['Nombre_Rol_Asignado'] = medias_df['ID_Cluster'].map(rol_mapping)
    
    # 4. Reordenar las columnas y establecer ID_Cluster como índice
    medias_df = medias_df.set_index('ID_Cluster')
    
    # 5. Ordenar por las features para facilitar la visualización del perfil
    medias_df = medias_df.sort_values(by=features[0]) 
    
    return medias_df

with st.expander("Calcular y Mostrar Medias de Centroides por Proyección"):
    for key, config in CLASSIFIERS.items():
        st.markdown(f"**Análisis de Medias para {config['title']}**")
        
        # LLAMADA A LA FUNCIÓN MODIFICADA
        medias_df = generar_medias_clusters(
            DF_POKEMON, 
            config['cluster_col'], 
            config['features'], 
            config['roles']
        )
        
        st.dataframe(medias_df.style.background_gradient(cmap='YlOrRd'))
        st.caption(f"""
            **Instrucciones de Verificación:**
            Asegúrate de que la columna **Nombre_Rol_Asignado** tenga sentido en relación con las medias de las columnas **{config['features'][0]}** y **{config['features'][1]}**. Si no, edita la lista 'roles' en el código.
        """)


# --- 5. Ejecución y Despliegue de Secciones (Sin Cambios) ---

st.header("Clasificaciones Independientes")

crear_seccion_clasificacion('att_speed', CLASSIFIERS['att_speed'])
crear_seccion_clasificacion('sp_att_sp_def', CLASSIFIERS['sp_att_sp_def'])
crear_seccion_clasificacion('att_def', CLASSIFIERS['att_def'])

# 6. Leyenda (Sin Cambios)
with st.expander("Ver Leyendas de Roles"):
    for key, config in CLASSIFIERS.items():
        st.markdown(f"**{config['title']}**")
        
        fig_legend, ax_legend = plt.subplots(figsize=(1, 1))
        
        handles = []
        labels = []
        for i, cluster_name in enumerate(config['roles']):
            color = GLOBAL_COLORS(i)
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.7))
            labels.append(cluster_name)

        ax_legend.legend(handles, labels, loc='center', ncol=1, title=f"Roles: {config['features'][0]} vs {config['features'][1]}", fontsize=8)
        ax_legend.axis('off')
        st.pyplot(fig_legend)

# --- NOTA IMPORTANTE ---
# Este bloque asume que el DataFrame cargado se llama DF_POKEMON 
# y que las variables FULL_FEATURE_NAMES y GLOBAL_COLORS están definidas.
# Asegúrate de que esas variables existan en el resto de tu script.
# La función display_team utiliza GLOBAL_COLORS.

# --- Inicialización de Session State para persistencia ---

# Importaciones necesarias (asegúrate de que estén al principio de tu script principal)
import streamlit as st
import pandas as pd
import requests 
import numpy as np # Necesario para generate_random_team

# Equipo aleatorio
if 'random_team' not in st.session_state:
    st.session_state.random_team = pd.DataFrame() 
# Equipo manual
if 'manual_team' not in st.session_state:
    st.session_state.manual_team = pd.DataFrame()

# --- FUNCIONES DE ASISTENCIA Y API ---

# GLOBAL_COLORS debe estar definida en la parte superior de tu script principal.
# Si no la tienes, puedes usar esta:
def GLOBAL_COLORS(index):
    colors = [
        '#6C7B8B', '#B8860B', '#A52A2A', '#006400', '#483D8B', 
        '#8B008B', '#FFD700', '#FF4500', '#00CED1', '#6A5ACD'
    ]
    return colors[index % len(colors)]


@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_pokemon_data_from_api(nombre_o_id: str):
    """
    Consulta la PokeAPI para obtener el nombre oficial, la URL de la imagen y los tipos.
    Retorna (nombre_oficial, url_imagen, tipos).
    """
    try:
        clean_input = str(nombre_o_id).lower().replace(" ", "-")
        url = f"https://pokeapi.co/api/v2/pokemon/{clean_input}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pokemon_name = data['name'].title()
        
        # Obtener imagen (probar diferentes fuentes en orden de preferencia)
        sprite_url = (
            data['sprites']['other']['official-artwork']['front_default'] or
            data['sprites']['other']['home']['front_default'] or
            data['sprites']['front_default']
        )
        
        types = [t['type']['name'].title() for t in data['types']]
        
        return pokemon_name, sprite_url, types
        
    except requests.exceptions.HTTPError:
        return None, None, None
    except Exception as e:
        st.error(f"Error obteniendo datos de {nombre_o_id}: {e}")
        return None, None, None

def generate_random_team():
    """Genera un equipo de 6 Pokémon aleatorios y lo guarda en session_state."""
    # **Asegúrate que DF_POKEMON esté disponible globalmente.**
    if not DF_POKEMON.empty:
        st.session_state.random_team = DF_POKEMON.sample(n=6, random_state=np.random.randint(0, 10000))
        st.success("¡Equipo aleatorio de 6 Pokémon generado con éxito!")


def display_team(team_df, team_type):
    """Muestra el equipo de Pokémon con todas las predicciones de roles."""
    
    if team_df.empty:
        st.info(f"El equipo {team_type} está vacío. ¡Genera o ingresa Pokémon!")
        return

    st.markdown(f"### Equipo {team_type} (Total: {team_df.shape[0]})")
    st.markdown("---")

    cols = st.columns(3)
    
    for i, (index, poke) in enumerate(team_df.iterrows()):
        col = cols[i % 3]
        
        with col:
            # Obtener datos básicos
            pokemon_name = str(poke.get('name', f'pokemon_{i}')).lower()
            official_name, image_url, types = get_pokemon_data_from_api(pokemon_name)
            
            if not official_name:
                official_name = poke.get('name', f'Pokémon {i+1}').title()
                
            if not types:
                type1 = poke.get('type1', 'Unknown')
                type2 = poke.get('type2')
                types = [type1.title()]
                if pd.notna(type2) and str(type2).strip().lower() not in ('nan', '', 'none'):
                    types.append(type2.title())
            
            # Manejo seguro del ID
            raw_id = poke.get('id')
            if pd.isna(raw_id) or raw_id is None:
                poke_id = i + 1
            else:
                try:
                    poke_id = int(float(raw_id))
                except (ValueError, TypeError):
                    poke_id = i + 1
            
            type_text = ' / '.join(types) if types else 'Desconocido'
            
            # --- DISEÑO MEJORADO CON PREDICCIONES ---
            
            # 1. Nombre del Pokémon (lo primero que se ve)
            st.markdown(f"**{official_name}**")
            
            # 2. Imagen del Pokémon
            if image_url:
                try:
                    st.image(image_url, width=100)
                except:
                    st.write("❌")
            else:
                st.write("🖼️")
            
            # 3. Información básica
            st.markdown(f"**Tipo:** {type_text}")
            st.markdown(f"*ID: #{poke_id}*")
            
            # 4. PREDICCIONES DE ROLES (¡NUEVO!)
            st.markdown("---")
            st.markdown("**Predicciones de Roles:**")
            
            # Predicción para Velocidad y Ataque
            att_speed_role = CLASSIFIERS['att_speed']['roles'][int(poke.get('cluster_att_speed', 0))]
            st.markdown(f"• **Ataque/Vel:** {att_speed_role.split(': ')[1]}")
            
            # Predicción para Ataque y Defensa Especial
            sp_att_sp_def_role = CLASSIFIERS['sp_att_sp_def']['roles'][int(poke.get('cluster_sp_att_sp_def', 0))]
            st.markdown(f"• **Especial:** {sp_att_sp_def_role.split(': ')[1]}")
            
            # Predicción para Ataque y Defensa Físicas
            att_def_role = CLASSIFIERS['att_def']['roles'][int(poke.get('cluster_att_def', 0))]
            st.markdown(f"• **Físico:** {att_def_role.split(': ')[1]}")
            
            # 5. ESTADÍSTICAS (¡NUEVO!)
            st.markdown("---")
            st.markdown("**Estadísticas:**")
            
            # Mostrar las estadísticas base
            stats = {
                'HP': int(poke.get('hp', 0)),
                'Ataque': int(poke.get('attack', 0)),
                'Defensa': int(poke.get('defense', 0)),
                'Atq. Esp.': int(poke.get('special-attack', 0)),
                'Def. Esp.': int(poke.get('special-defense', 0)),
                'Velocidad': int(poke.get('speed', 0))
            }
            
            for stat, value in stats.items():
                st.markdown(f"• **{stat}:** {value}")
            
            # Separador entre Pokémon
            if i < len(team_df) - 1:
                st.markdown("---")

# --- SECCIÓN PRINCIPAL: GENERADOR DE EQUIPOS (Sin cambios) ---
# **Asegúrate de que DF_POKEMON esté definido antes de esta sección.**

st.header("🛠️ Generador de Equipos Pokémon")

# Creación de pestañas para dividir la funcionalidad
tab_random, tab_manual = st.tabs(["1️⃣ Generador Aleatorio", "2️⃣ Selector Manual"])

# --- 1. GENERADOR ALEATORIO ---

with tab_random:
    st.markdown("Presiona el botón para seleccionar **6 Pokémon** al azar de todo el dataset.")
    
    if st.button("Generar Equipo de 6 Pokémon Aleatorios", key='btn_random', use_container_width=True):
        generate_random_team()
        
    display_team(st.session_state.random_team, "Aleatorio")

# --- 2. SELECTOR MANUAL ---

with tab_manual:
    st.markdown("Ingresa hasta 6 nombres de Pokémon. **Asegúrate de escribirlos correctamente** (e.g., 'charmander', 'pikachu').")

    manual_names = []
    
    # Inputs para 6 Pokémon en 3 columnas
    input_cols = st.columns(3)
    
    for i in range(6):
        with input_cols[i % 3]:
            # Guardamos el nombre ingresado
            name = st.text_input(f"Pokémon {i+1}", key=f'manual_name_{i}', value="", placeholder="Nombre...", max_chars=30)
            if name:
                manual_names.append(name.lower().strip())
    
    # Botón para procesar y validar el equipo manual
    if st.button("Crear y Validar Equipo Manual", key='btn_manual', use_container_width=True):
        
        # 1. Filtrar solo los nombres válidos que existen en el DF
        valid_team_df = DF_POKEMON[DF_POKEMON['name'].str.lower().isin(manual_names)]
        
        if valid_team_df.shape[0] > 0:
            
            # 2. Identificar nombres no encontrados
            found_names = valid_team_df['name'].str.lower().tolist()
            not_found_names = set(manual_names) - set(found_names)
            
            if not_found_names:
                st.warning(f"⚠️**Advertencia:** No se encontraron los siguientes Pokémon en el dataset: {', '.join(not_found_names)}. Se mostrarán solo los encontrados.")
            
            # Guardar el equipo válido (si el usuario ingresó más de 6, solo se guardan los primeros 6 válidos)
            st.session_state.manual_team = valid_team_df.head(6) 
            st.success(f"🦖 Equipo manual procesado. Se encontraron {st.session_state.manual_team.shape[0]} Pokémon válidos.")
        
        else:
            st.error("❌ No se encontró ninguno de los Pokémon ingresados o la lista está vacía. ¡Verifica la ortografía!")
            st.session_state.manual_team = pd.DataFrame()

    display_team(st.session_state.manual_team, "Manual")