import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import requests

# --- 0. CONFIGURACIÓN DE PÁGINAS Y ESTADO ---

# Constantes de las páginas
START_PAGE = "Página de Inicio"
CLASSIFICATION_PAGE = "Clasificación de Roles"
TEAM_BUILDER_PAGE = "Generador de Equipos"

# Inicialización de Session State para persistencia
if 'page' not in st.session_state:
    st.session_state.page = START_PAGE
if 'random_team' not in st.session_state:
    st.session_state.random_team = pd.DataFrame() 
if 'manual_team' not in st.session_state:
    st.session_state.manual_team = pd.DataFrame()

# URL de tu imagen de fondo
BACKGROUND_IMAGE_URL = "https://i.pinimg.com/originals/cd/f2/a5/cdf2a5aa0a40469d23873928f128336f.jpg"

# --- CSS PERSONALIZADO (STREAMLIT POKÉMON THEME) ---
st.markdown(f"""
<style>
/* 1. CONFIGURACIÓN DEL FONDO */
.stApp {{
    background-image: url("{BACKGROUND_IMAGE_URL}");
    background-size: cover; 
    background-attachment: fixed; 
    background-position: center; 
    color: #000000; 
}}

/* 2. CONTENEDOR PRINCIPAL (Visibilidad Máxima) */
.main {{
    /* Mantenemos una ligera transparencia para ver el fondo */
    background-color: rgba(255, 255, 255, 0.95); 
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); 
}}

/* 3. ENCABEZADOS - Estilo Pokedex */
h1, h2, h3, h4 {{
    color: #CC0000; 
    font-family: 'Arial Black', sans-serif;
    border-bottom: 2px solid #3B4CCA; 
    padding-bottom: 5px;
    margin-top: 20px;
    /* Evitamos que los encabezados hereden el fondo de texto genérico */
    background-color: transparent !important; 
    padding-left: 0 !important;
    padding-right: 0 !important;
    display: block; /* Asegura que el borde inferior se vea bien */
}}
h1 {{
    font-size: 2.5em;
    color: #3B4CCA; 
    text-shadow: 2px 2px #CC0000;
}}


/* 4. BOTONES - Estilo Clásico de Consola */
.stButton>button {{
    background-color: #FFDE00; 
    color: #3B4CCA; 
    border: 3px solid #CC0000; 
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 1.1em;
    transition: all 0.2s;
    box-shadow: 3px 3px #888888;
}}
/* Efecto hover y Botones activos (seleccionados) */
.stButton>button:hover {{
    background-color: #CC0000; 
    color: #FFDE00; 
    border-color: #3B4CCA;
    box-shadow: 4px 4px #555555;
}}
.stButton>button.active {{ 
    background-color: #CC0000; 
    color: #FFDE00; 
    border-color: #3B4CCA;
    box-shadow: 4px 4px #555555;
}}

/* 5. SLIDERS - Barra de vida */
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: #FFDE00;
    border: 1px solid #CC0000;
}}
div[data-testid="stSlider"] div[role="slider"] + div {{
    background-color: #3B4CCA;
}}

/* 6. TEXTO GENERAL Y MARKDOWN (MEJORA DE LEGIBILIDAD: CAMBIO CLAVE) */

/* 6a. FORZAR COLOR NEGRO Y TAMAÑO */
div.stMarkdown p, 
li, 
label, 
div[data-testid^="stText"], 
div[data-testid="stCaption"], 
div[data-testid="stSidebarContent"] * {{
    color: #000000 !important; /* ¡NEGRO FORZADO! */
    font-size: 1.05em; 
    line-height: 1.5;
}}

/* 6b. ¡APLICAR FONDO BLANCO SEMI-TRANSPARENTE A TODO EL TEXTO SIMPLE! */
p, li, div[data-testid^="stText"], div[data-testid="stCaption"], label {{
    background-color: rgba(255, 255, 255, 0.7) !important; /* Fondo semi-transparente blanco */
    padding: 2px 5px !important; /* Pequeño relleno para que el fondo se vea bien */
    border-radius: 3px; /* Esquinas suaves */
}}

/* Ajustes de display para que el fondo de las etiquetas (sliders/inputs) se vea bien */
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {{
    display: block !important; 
    width: 100% !important;
}}

/* Sobrescribe el color para que los enlaces no sean negros */
a {{
    color: #3B4CCA !important;
}}

/* 7. TEXTO DE CLASIFICACIÓN IMPORTANTE (ROLES Y CONFIANZA) */
/* Forzar el color a negro para máxima legibilidad en los resultados de predicción */
p[style*="font-size:24px"] {{
    color: #000000 !important; 
    font-weight: 900; 
    text-shadow: 1px 1px 1px rgba(255, 255, 255, 0.5); 
    background-color: transparent !important; /* Evitamos doble fondo, ya que se le aplica en show_classification_page */
}}

/* Ajuste para el texto que ya está dentro de un contenedor con fondo (e.g., el inicio) */
.stMarkdown > div {{
    background-color: transparent !important;
    padding: 0 !important;
}}

</style>
""", unsafe_allow_html=True)

# --- Configuración de la Página ---
st.set_page_config(page_title="Clasificación de Roles Pokémon", layout="wide")


# --- Variables Globales y Definiciones ---
N_CLUSTERS = 8
FULL_FEATURE_NAMES = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
DATA_PATH = "pokemon.csv" 
GLOBAL_COLORS = plt.cm.get_cmap('tab10') 
K_NEIGHBORS = 5 

# --- ¡CRUCIAL! LISTAS DE ROLES ---
CLASSIFIERS = {
    'att_speed': {
        'title': "1. Clasificación (KNN): Velocidad y Ataque",
        'features': ['attack', 'speed'],
        'model_path': 'knn_att_speed_model.pkl',
        'cluster_col': 'cluster_att_speed',
        'roles': [
            'Rol 0: Atacante Rápido Élite', 
            'Rol 1: Velocidad Media/Ataque Bajo', 
            'Rol 2: Ataque Medio/Velocidad Baja',
            'Rol 3: Speedster (Alta Vel/Bajo Atq)',
            'Rol 4: Bajo Rendimiento (Atq/Vel)',
            'Rol 5: Balanceado Rápido',
            'Rol 6: Ataque Extermo Lento',
            'Rol 7: Ataque Formidable/Velocidad Media'
        ]
    },
    'sp_att_sp_def': {
        'title': "2. Clasificación (KNN): Ataque y Defensa Especial",
        'features': ['special-attack', 'special-defense'],
        'model_path': 'knn_sp_att_sp_def_model.pkl',
        'cluster_col': 'cluster_sp_att_sp_def',
        'roles': [
            'Rol 0: Balanceado Especial',
            'Rol 1: Special Dangerous Barrier (Extrema Def/Extremo Atq)',
            'Rol 2: Defensor Élite Mixto (Extrema Def/Alto Atq)',
            'Rol 3: Bajo Especial General',
            'Rol 4: Defensor Especialista (Medio Atq/Alta Def)',
            'Rol 5: Atacante Élite Mixto (Extremo Atq/Alta Def)',
            'Rol 6: Especial Común',
            'Rol 7: Atacante Especialista (Alto Atq/Media Def)'
        ]
    },
    'att_def': {
        'title': "3. Clasificación (KNN): Ataque y Defensa Físicas",
        'features': ['attack', 'defense'],
        'model_path': 'knn_att_def_model.pkl',
        'cluster_col': 'cluster_att_def',
        'roles': [
            'Rol 0: Atacante Puro (Extremo Atq/Media Def)',
            'Rol 1: Defensor/Atacante Mixto (Alto Atq/Alta Def)',
            'Rol 2: Bajo Físico General',
            'Rol 3: Defensor Puro (Medio Atq/Extrema Def)',
            'Rol 4: Defensor Ligero (Medio Atq/Alta Def)',
            'Rol 5: Atacante Físico Frágil (Alto Atq/Baja Def)',
            'Rol 6: Balanceado Físico',
            'Rol 7: Striker Tank (Extremo Atq/Extrema Def)'
        ]
    }
}


# --- 1. Carga y CLUSTERING INDEPENDIENTE ---

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
        st.error(f"Error: No se encontró el archivo de datos en la ruta: {DATA_PATH}. ¡Asegúrate de subir 'pokemon.csv' al repositorio!")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar o preprocesar los datos: {e}")
        st.stop()

DF_POKEMON = load_and_cluster_data()


# --- 2. Carga/Entrenamiento de Modelos (KNN) ---

@st.cache_resource(hash_funcs={pd.DataFrame: id})
def train_and_save_model(df_clustered, features, cluster_col, model_path):
    """Entrena y guarda un modelo KNN (KNeighborsClassifier) para un subconjunto específico."""
    
    X = df_clustered[features]
    y = df_clustered[cluster_col] 
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.sidebar.info(f"Entrenando nuevo modelo KNN (K={K_NEIGHBORS}) para {features[0]} vs {features[1]}...")
        model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS) 
        model.fit(X, y)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model

MODELOS = {}
for key, config in CLASSIFIERS.items():
    MODELOS[key] = train_and_save_model(
        DF_POKEMON, 
        config['features'], 
        config['cluster_col'], 
        config['model_path']
    )


# --- 3. FUNCIONES DE UTILIDAD ---
@st.cache_data(ttl=3600) 
def get_pokemon_data_from_api(nombre_o_id: str):
    """Consulta la PokeAPI para obtener el nombre oficial, la URL de la imagen y los tipos."""
    try:
        clean_input = str(nombre_o_id).lower().replace(" ", "-")
        url = f"https://pokeapi.co/api/v2/pokemon/{clean_input}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pokemon_name = data['name'].title()
        
        sprite_url = (
            data['sprites']['other']['official-artwork']['front_default'] or
            data['sprites']['other']['home']['front_default'] or
            data['sprites']['front_default']
        )
        
        types = [t['type']['name'].title() for t in data['types']]
        
        return pokemon_name, sprite_url, types
        
    except requests.exceptions.HTTPError:
        return None, None, None
    except Exception:
        return None, None, None

def get_slider_input(key, features):
    """Crea sliders en la sección principal para las features de un modelo."""
    data = {}
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        feature = features[0]
        min_val, max_val, mean_val = DF_POKEMON[feature].min(), DF_POKEMON[feature].max(), DF_POKEMON[feature].mean()
        # Los labels de los sliders se ven bien gracias al CSS global
        data[feature] = st.slider(
            f'⚔️ {feature.replace("-", " ").title()}', 
            min_val.item(), max_val.item(), round(mean_val.item()), key=f'{key}_{feature}'
        )
        
    with col_input2:
        feature = features[1]
        min_val, max_val, mean_val = DF_POKEMON[feature].min(), DF_POKEMON[feature].max(), DF_POKEMON[feature].mean()
        # Los labels de los sliders se ven bien gracias al CSS global
        data[feature] = st.slider(
            f'🛡️ {feature.replace("-", " ").title()}', 
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
        # Muestra el rol limpio (sin 'Rol X:')
        rol_limpio = prediccion_nombre.split(': ')[-1]
        # El CSS global fuerza el color de este p tag a NEGRO y tiene un fondo especial
        st.markdown(f"**Rol Predicho:** <p style='font-size:24px; color:#CC0000;'> {rol_limpio}</p>", unsafe_allow_html=True)
    with col_conf:
        # El CSS global fuerza el color de este p tag a NEGRO y tiene un fondo especial
        st.markdown(f"**N° Vecinos (K={K_NEIGHBORS})** <p style='font-size:24px; color:#3B4CCA;'>{confianza:.2f}%</p>", unsafe_allow_html=True)
        
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
        s=250, label=f'Nuevo Pokémon ({prediccion_nombre.split(": ")[-1]})', zorder=5 
    )

    ax.set_xlabel(x_col.replace("-", " ").title())
    ax.set_ylabel(y_col.replace("-", " ").title())
    ax.set_title(f'Proyección: {x_col.title()} vs {y_col.title()}')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def generar_medias_clusters(df, cluster_col, features, target_names):
    """Calcula la media de las features para cada cluster y añade la columna de rol."""
    medias = df.groupby(cluster_col)[features].mean()
    medias_df = medias.reset_index().rename(columns={cluster_col: 'ID_Cluster'})
    
    rol_mapping = {i: name for i, name in enumerate(target_names)}
    medias_df['Nombre_Rol_Asignado'] = medias_df['ID_Cluster'].map(rol_mapping)
    
    medias_df = medias_df.set_index('ID_Cluster')
    medias_df = medias_df.sort_values(by=features[0]) 
    
    return medias_df


def generate_random_team():
    """Genera un equipo de 6 Pokémon aleatorios y lo guarda en session_state."""
    if not DF_POKEMON.empty:
        st.session_state.random_team = DF_POKEMON.sample(n=6, random_state=np.random.randint(0, 10000))
        st.success("¡Equipo aleatorio de 6 Pokémon generado con éxito!")


def display_team(team_df, team_type):
    """Muestra el equipo de Pokémon con todas las predicciones de roles, forzando la visualización de la imagen con HTML y agregando un fondo de texto."""
    
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
            
            # --- MANEJO SEGURO DEL ID ---
            raw_id = poke.get('id')
            poke_id_display = str(i + 1) # Fallback

            if pd.notna(raw_id) and raw_id is not None:
                try:
                    poke_id_int = int(float(raw_id))
                    poke_id_display = str(poke_id_int)
                except (ValueError, TypeError):
                    pass 
            
            type_text = ' / '.join(types) if types else 'Desconocido'
            
            # --- DISEÑO MEJORADO CON FONDO DE TEXTO ---
            
            # 1. Nombre del Pokémon
            st.markdown(f'<span style="font-weight: bold; font-size: 1.2em; color:#000000;">{official_name}</span>', unsafe_allow_html=True)
            
            # 2. Imagen del Pokémon
            if image_url:
                image_html = f'<img src="{image_url}" width="100" style="display:block; margin-left: auto; margin-right: auto;"/>'
                st.markdown(image_html, unsafe_allow_html=True)
            else:
                st.write("🖼️") # Emoji fallback si no hay URL
            
            # --- INICIO DEL CONTENEDOR CON FONDO (SOLO PARA LA INFORMACIÓN) ---
            # Fondo blanco semi-transparente para la sección de detalles del Pokémon
            # Usamos un div con fondo y el texto adentro hereda el estilo de color negro de p/span.
            st.markdown('<div style="background-color: rgba(255, 255, 255, 0.7); padding: 5px; border-radius: 5px;">', unsafe_allow_html=True)
            
            # 3. Información básica
            # Al usar p, hereda el fondo semi-transparente del CSS global, pero se lo quitamos con el div de arriba
            st.markdown(f'<p style="color:#000000; background-color: transparent !important; padding: 0 !important; margin-bottom: 0;"><b>Tipo:</b> {type_text}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#000000; background-color: transparent !important; padding: 0 !important; margin-top:-10px; margin-bottom: 0;"><i>ID: #{poke_id_display}</i></p>', unsafe_allow_html=True)
            
            # 4. PREDICCIONES DE ROLES
            st.markdown('<hr style="border-top: 1px solid #CCC; margin: 5px 0;">', unsafe_allow_html=True)
            st.markdown(f'<span style="font-weight: bold; color:#000000; background-color: transparent !important; padding: 0 !important;">Predicciones de Roles:</span>', unsafe_allow_html=True)
            
            # Predicción para Velocidad y Ataque
            att_speed_role = CLASSIFIERS['att_speed']['roles'][int(poke.get('cluster_att_speed', 0))]
            st.markdown(f'<p style="color:#000000; background-color: transparent !important; padding: 0 !important; margin:0;">• <b>Ataque/Vel:</b> {att_speed_role.split(": ")[1]}</p>', unsafe_allow_html=True)
            
            # Predicción para Ataque y Defensa Especial
            sp_att_sp_def_role = CLASSIFIERS['sp_att_sp_def']['roles'][int(poke.get('cluster_sp_att_sp_def', 0))]
            st.markdown(f'<p style="color:#000000; background-color: transparent !important; padding: 0 !important; margin:0;">• <b>Especial:</b> {sp_att_sp_def_role.split(": ")[1]}</p>', unsafe_allow_html=True)
            
            # Predicción para Ataque y Defensa Físicas
            att_def_role = CLASSIFIERS['att_def']['roles'][int(poke.get('cluster_att_def', 0))]
            st.markdown(f'<p style="color:#000000; background-color: transparent !important; padding: 0 !important; margin:0;">• <b>Físico:</b> {att_def_role.split(": ")[1]}</p>', unsafe_allow_html=True)
            
            # 5. ESTADÍSTICAS
            st.markdown('<hr style="border-top: 1px solid #CCC; margin: 5px 0;">', unsafe_allow_html=True)
            st.markdown(f'<span style="font-weight: bold; color:#000000; background-color: transparent !important; padding: 0 !important;">Estadísticas:</span>', unsafe_allow_html=True)
            
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
                st.markdown(f'<p style="color:#000000; background-color: transparent !important; padding: 0 !important; margin:0;">• <b>{stat}:</b> {value}</p>', unsafe_allow_html=True)
            
            # --- CIERRE DEL CONTENEDOR CON FONDO ---
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Separador entre Pokémon
            if i < len(team_df) - 1:
                st.markdown("<br>", unsafe_allow_html=True)


# --- 4. FUNCIONES DE PÁGINA ---

def show_start_page():
    """Muestra la pantalla de inicio con un mensaje de bienvenida."""
    # Este div ya tiene un fondo semi-transparente, por lo que el texto dentro se ve bien
    st.markdown("""
        <div style="
            padding: 30px; 
            border: 5px solid #FFDE00; 
            border-radius: 10px; 
            background-color: rgba(255, 255, 255, 0.95);
            text-align: center;
        ">
            <h2 style="color: #3B4CCA; border-bottom: 3px solid #CC0000; padding-bottom: 10px;">¡Bienvenido al Centro de Análisis de Roles Pokémon!</h2>
            <p style="font-size: 1.2em; margin-top: 20px; background-color: transparent !important; padding: 0 !important;">
                Esta aplicación utiliza un modelo de clasificación K-Nearest Neighbors (KNN)
                para analizar las estadísticas base de los Pokémon y asignarles un <b>Rol de Combate</b>
                (ej. Defensor Puro, Atacante Rápido).
            </p>
            <h3 style="color: #CC0000; border-bottom: 1px dashed #3B4CCA;">Pasos a seguir:</h3>
            <ul style="list-style-type: none; padding: 0; text-align: left; margin: 0 auto; width: fit-content;">
                <li style="margin-bottom: 10px; font-size: 1.1em; background-color: transparent !important; padding: 0 !important;">
                    <span style="font-weight: bold; color: #3B4CCA;">1. Clasificar Nuevo Pokémon:</span> 
                    Utiliza los <i>sliders</i> en esa sección para simular un nuevo Pokémon y ver la predicción de su rol.
                </li>
                <li style="margin-bottom: 10px; font-size: 1.1em; background-color: transparent !important; padding: 0 !important;">
                    <span style="font-weight: bold; color: #3B4CCA;">2. Generador de Equipos:</span> 
                    Crea un equipo de 6 Pokémon (aleatorio o manual) y visualiza las clasificaciones de roles de cada uno.
                </li>
            </ul>
            <img src="https://pbs.twimg.com/profile_images/1297556087114280966/B3k1hur3.jpg" width="80" style="margin-top: 20px; border-radius: 50%;"> 
        </div>
    """, unsafe_allow_html=True)

def show_classification_page():
    """Muestra la interfaz para clasificar un Pokémon nuevo."""
    st.header("Clasificaciones de Roles (KNN)")
    st.markdown("Utiliza los *sliders* para ajustar las estadísticas del Pokémon y observa en qué **Rol de Combate** lo clasifica el modelo K-Nearest Neighbors.")
    st.markdown("---")
    
    # 4.0. HERRAMIENTA DE MAEPEO (Centroides)
    with st.expander("Calcular y Mostrar Medias de Centroides por Proyección"):
        st.success("""
            La tabla te ayuda a verificar si el **Nombre del Rol** que definiste tiene sentido 
            con las **medias** (centroides) de las estadísticas en ese cluster.
        """)
        for key, config in CLASSIFIERS.items():
            st.markdown(f"**Análisis de Medias para {config['title']}**")
            
            medias_df = generar_medias_clusters(
                DF_POKEMON, 
                config['cluster_col'], 
                config['features'], 
                config['roles']
            )
            
            # El estilo de la tabla se mejora con la clase .dataframe del CSS global
            st.dataframe(medias_df.style.background_gradient(cmap='YlOrRd'))
            st.caption(f"Verifica la media de **{config['features'][0]}** y **{config['features'][1]}** frente al rol asignado.")
    
    # 5. Ejecución y Despliegue de Secciones (Clasificación)
    crear_seccion_clasificacion('att_speed', CLASSIFIERS['att_speed'])
    crear_seccion_clasificacion('sp_att_sp_def', CLASSIFIERS['sp_att_sp_def'])
    crear_seccion_clasificacion('att_def', CLASSIFIERS['att_def'])

    # 6. Leyenda
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


def show_team_builder_page():
    """Muestra la interfaz para crear equipos aleatorios o manuales."""
    st.header("Generador de Equipos Pokémon")
    
    # Creación de pestañas para dividir la funcionalidad
    tab_random, tab_manual = st.tabs(["1️⃣ Generador Aleatorio", "2️⃣ Selector Manual"])

    # --- 1. GENERADOR ALEATORIO ---
    with tab_random:
        st.markdown("Presiona el botón para seleccionar **6 Pokémon** al azar de todo el dataset y ver su clasificación de roles.")
        
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
                # El texto de este input se ve bien gracias al CSS global
                name = st.text_input(f"Pokémon {i+1}", key=f'manual_name_{i}', value="", placeholder="Nombre...", max_chars=30)
                if name:
                    manual_names.append(name.lower().strip())
        
        # Botón para procesar y validar el equipo manual
        if st.button("Crear y Validar Equipo Manual", key='btn_manual', use_container_width=True):
            
            valid_team_df = DF_POKEMON[DF_POKEMON['name'].str.lower().isin(manual_names)]
            
            if valid_team_df.shape[0] > 0:
                found_names = valid_team_df['name'].str.lower().tolist()
                not_found_names = set(manual_names) - set(found_names)
                
                if not_found_names:
                    st.warning(f"⚠️**Advertencia:** No se encontraron los siguientes Pokémon en el dataset: {', '.join(not_found_names)}. Se mostrarán solo los encontrados.")
                
                st.session_state.manual_team = valid_team_df.head(6) 
                st.success(f"🦖 Equipo manual procesado. Se encontraron {st.session_state.manual_team.shape[0]} Pokémon válidos.")
            
            else:
                st.error("❌ No se encontró ninguno de los Pokémon ingresados o la lista está vacía. ¡Verifica la ortografía!")
                st.session_state.manual_team = pd.DataFrame()

        display_team(st.session_state.manual_team, "Manual")


# --- 5. ESTRUCTURA DE LA APLICACIÓN PRINCIPAL (MENU) ---

st.title("Clasificación de Roles de Combate Pokémon")

# Contenedor para los botones de navegación
menu_cols = st.columns(2)

# Función para cambiar la página al hacer clic
def set_page(page_name):
    st.session_state.page = page_name

# Botón 1: Clasificación de Nuevo Pokémon
with menu_cols[0]:
    if st.button("Clasificar Nuevo Pokémon", key='btn_nav_clasif', use_container_width=True):
        set_page(CLASSIFICATION_PAGE)
        st.rerun() 

# Botón 2: Generador de Equipos
with menu_cols[1]:
    if st.button("Generador de Equipos", key='btn_nav_teams', use_container_width=True):
        set_page(TEAM_BUILDER_PAGE)
        st.rerun()


st.markdown("---")


# --- 6. RENDERIZADO CONDICIONAL DEL CONTENIDO ---

if st.session_state.page == CLASSIFICATION_PAGE:
    show_classification_page()
elif st.session_state.page == TEAM_BUILDER_PAGE:
    show_team_builder_page()
elif st.session_state.page == START_PAGE:
    show_start_page()
