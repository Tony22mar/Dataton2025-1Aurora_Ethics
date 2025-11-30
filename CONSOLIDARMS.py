# =============================================================================
# CONSOLIDADOR MASIVO DE ARCHIVOS JSON
# Sistema de Carga y Unificación de Datos
# Versión Corregida - Sin errores de memoria
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Consolidador JSON Masivo",
    layout="wide",
    page_icon="📁",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS PERSONALIZADO - DISEÑO AURORA-ETHICS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: white !important;
    }
    
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #667eea;
    }
    
    .hero-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: fadeIn 1s ease-in;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .brand-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 3px;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .quote-text {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-style: italic;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .author-text {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    
    .signature-text {
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.9rem;
        font-weight: 300;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #48bb78;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(237, 137, 54, 0.1) 0%, rgba(221, 107, 32, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ed8936;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.2);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.1) 0%, rgba(229, 62, 62, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #f56565;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #667eea;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    div[data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .file-counter {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        padding: 1rem 2rem;
        border-radius: 50px;
        color: white;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .processing-indicator {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        padding: 1rem 2rem;
        border-radius: 50px;
        color: white;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.4);
        animation: pulse 1s ease-in-out infinite;
    }
    
    .download-section {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid rgba(72, 187, 120, 0.3);
        margin: 2rem 0;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INICIALIZACIÓN DE SESSION STATE
# =============================================================================

if 'df_consolidado' not in st.session_state:
    st.session_state.df_consolidado = None

if 'archivos_procesados' not in st.session_state:
    st.session_state.archivos_procesados = 0

if 'archivos_con_error' not in st.session_state:
    st.session_state.archivos_con_error = []

if 'total_registros' not in st.session_state:
    st.session_state.total_registros = 0

if 'consolidacion_completada' not in st.session_state:
    st.session_state.consolidacion_completada = False

if 'campo_personalizado_nombre' not in st.session_state:
    st.session_state.campo_personalizado_nombre = ""

if 'campo_personalizado_valor' not in st.session_state:
    st.session_state.campo_personalizado_valor = ""

if 'campo_agregado' not in st.session_state:
    st.session_state.campo_agregado = False

if 'historial_procesamiento' not in st.session_state:
    st.session_state.historial_procesamiento = []

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def limpiar_estado():
    """Limpia el estado de la sesión para nueva carga"""
    st.session_state.df_consolidado = None
    st.session_state.archivos_procesados = 0
    st.session_state.archivos_con_error = []
    st.session_state.total_registros = 0
    st.session_state.consolidacion_completada = False
    st.session_state.campo_agregado = False


def flatten_json(nested_json, parent_key='', sep='_'):
    """
    Aplana un JSON anidado recursivamente.
    Versión optimizada para evitar problemas de memoria.
    """
    items = {}
    
    if isinstance(nested_json, dict):
        for key, value in nested_json.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(flatten_json(value, new_key, sep))
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    for i, item in enumerate(value[:10]):  # Limitar a 10 items
                        items.update(flatten_json(item, f"{new_key}_{i}", sep))
                else:
                    items[new_key] = str(value) if value else ""
            else:
                items[new_key] = value
    elif isinstance(nested_json, list):
        for i, item in enumerate(nested_json[:10]):  # Limitar a 10 items
            if isinstance(item, dict):
                items.update(flatten_json(item, f"{parent_key}_{i}" if parent_key else str(i), sep))
            else:
                items[f"{parent_key}_{i}" if parent_key else str(i)] = item
    else:
        items[parent_key] = nested_json
    
    return items


def procesar_json_individual(contenido_json, nombre_archivo):
    """
    Procesa un archivo JSON individual y lo convierte en lista de diccionarios.
    """
    try:
        registros = []
        
        if isinstance(contenido_json, list):
            for item in contenido_json:
                if isinstance(item, dict):
                    registro_plano = flatten_json(item)
                    registro_plano['_nombre_archivo_origen'] = nombre_archivo
                    registros.append(registro_plano)
                else:
                    registros.append({
                        'valor': item,
                        '_nombre_archivo_origen': nombre_archivo
                    })
        
        elif isinstance(contenido_json, dict):
            registro_plano = flatten_json(contenido_json)
            registro_plano['_nombre_archivo_origen'] = nombre_archivo
            registros.append(registro_plano)
        
        else:
            registros.append({
                'valor': contenido_json,
                '_nombre_archivo_origen': nombre_archivo
            })
        
        return registros
    
    except Exception as e:
        st.session_state.archivos_con_error.append({
            'archivo': nombre_archivo,
            'error': str(e)
        })
        return []


def procesar_archivos_json(uploaded_files, progress_bar, status_text):
    """
    Procesa múltiples archivos JSON y los consolida en un DataFrame único.
    Versión optimizada para manejo de memoria.
    """
    todos_registros = []
    total_archivos = len(uploaded_files)
    archivos_exitosos = 0
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"📄 Procesando: {uploaded_file.name} ({idx + 1}/{total_archivos})")
            
            contenido = uploaded_file.read()
            
            try:
                contenido_decodificado = contenido.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    contenido_decodificado = contenido.decode('latin-1')
                except UnicodeDecodeError:
                    contenido_decodificado = contenido.decode('utf-8', errors='ignore')
            
            datos_json = json.loads(contenido_decodificado)
            
            registros = procesar_json_individual(datos_json, uploaded_file.name)
            
            if registros:
                todos_registros.extend(registros)
                archivos_exitosos += 1
            
            progress_bar.progress((idx + 1) / total_archivos)
            
        except json.JSONDecodeError as e:
            st.session_state.archivos_con_error.append({
                'archivo': uploaded_file.name,
                'error': f"Error de formato JSON: {str(e)}"
            })
        except Exception as e:
            st.session_state.archivos_con_error.append({
                'archivo': uploaded_file.name,
                'error': str(e)
            })
    
    st.session_state.archivos_procesados = archivos_exitosos
    
    if todos_registros:
        status_text.text("🔄 Consolidando datos...")
        
        # Crear DataFrame de forma eficiente
        df_consolidado = pd.DataFrame(todos_registros)
        
        # Reordenar columnas de forma segura
        if '_nombre_archivo_origen' in df_consolidado.columns:
            cols = [col for col in df_consolidado.columns if col != '_nombre_archivo_origen']
            cols.insert(0, '_nombre_archivo_origen')
            df_consolidado = df_consolidado.reindex(columns=cols)
        
        st.session_state.total_registros = len(df_consolidado)
        
        return df_consolidado
    
    return None


def agregar_campo_personalizado(df, nombre_columna, valor):
    """
    Agrega una columna personalizada al DataFrame con un valor específico.
    """
    if df is not None and nombre_columna and valor:
        df[nombre_columna] = valor
        
        # Reordenar para poner el campo después del nombre de archivo
        if '_nombre_archivo_origen' in df.columns:
            cols = df.columns.tolist()
            cols.remove(nombre_columna)
            idx = cols.index('_nombre_archivo_origen') + 1
            cols.insert(idx, nombre_columna)
            df = df.reindex(columns=cols)
        
        return df
    
    return df


def convertir_df_a_csv(df):
    """Convierte DataFrame a CSV en bytes"""
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def convertir_df_a_excel(df):
    """Convierte DataFrame a Excel en bytes"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos_Consolidados')
    output.seek(0)
    return output.getvalue()


def obtener_estadisticas_df(df):
    """Obtiene estadísticas del DataFrame"""
    if df is None:
        return {}
    
    stats = {
        'total_registros': len(df),
        'total_columnas': len(df.columns),
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'valores_nulos': int(df.isnull().sum().sum()),
        'archivos_unicos': df['_nombre_archivo_origen'].nunique() if '_nombre_archivo_origen' in df.columns else 0
    }
    
    return stats
# =============================================================================
# HEADER PRINCIPAL
# =============================================================================

st.markdown("""
<div class="hero-header">
    <h1 class="brand-title">📁 CONSOLIDADOR JSON</h1>
    <h2 style="color: white; font-size: 1.8rem; margin: 0.5rem 0;">Carga Masiva de Archivos</h2>
    <p class="quote-text">"Unifica miles de archivos JSON en segundos"</p>
    <p class="author-text">Sistema de Consolidación de Datos</p>
    <p class="signature-text">Capacidad: Hasta 5,000+ archivos JSON</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - INFORMACIÓN Y CONFIGURACIÓN
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #667eea;">📁 Consolidador JSON</h2>
        <p style="color: #764ba2;">Carga Masiva de Datos</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Estado del Sistema")
    
    if st.session_state.df_consolidado is not None:
        stats = obtener_estadisticas_df(st.session_state.df_consolidado)
        
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{stats['total_registros']:,}</p>
            <p class="stat-label">Registros Totales</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
            <p class="stat-number">{stats['archivos_unicos']:,}</p>
            <p class="stat-label">Archivos Procesados</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);">
            <p class="stat-number">{stats['total_columnas']}</p>
            <p class="stat-label">Columnas</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);">
            <p class="stat-number">{len(st.session_state.archivos_con_error)}</p>
            <p class="stat-label">Archivos con Error</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**💾 Memoria:** {stats['memoria_mb']:.2f} MB")
    else:
        st.info("📭 Sin datos cargados")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Información")
    
    st.markdown("""
    <div class="info-box">
    <h5>📋 Características:</h5>
    <p>✅ Carga de hasta 5,000+ archivos</p>
    <p>✅ Soporte JSON anidado</p>
    <p>✅ Consolidación automática</p>
    <p>✅ Campo personalizado</p>
    <p>✅ Exportación CSV/Excel</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Limpiar Todo", use_container_width=True, key='btn_limpiar_sidebar'):
        limpiar_estado()
        st.rerun()

# =============================================================================
# CONTENIDO PRINCIPAL - TABS
# =============================================================================

tabs = st.tabs([
    "📤 Cargar Archivos",
    "📊 Vista de Datos",
    "➕ Campo Personalizado",
    "📥 Descargar Datos",
    "📋 Resumen"
])

# =============================================================================
# TAB 1: CARGA DE ARCHIVOS
# =============================================================================

with tabs[0]:
    st.markdown("## 📤 Carga Masiva de Archivos JSON")
    
    st.markdown("""
    <div class="info-box">
    <h4>📁 Instrucciones de Carga</h4>
    <p>Arrastra o selecciona múltiples archivos JSON para consolidar. El sistema:</p>
    <ul>
        <li>✅ Lee múltiples JSON en lote (hasta 5,000+ archivos)</li>
        <li>✅ Combina todos los datos automáticamente</li>
        <li>✅ Convierte estructuras anidadas en columnas planas</li>
        <li>✅ Unifica todo en una tabla consolidada</li>
        <li>✅ Agrega columna con nombre del archivo de origen</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Selecciona tus archivos JSON")
    
    uploaded_files = st.file_uploader(
        "Arrastra archivos JSON aquí o haz clic para seleccionar",
        type=['json'],
        accept_multiple_files=True,
        key='json_uploader',
        help="Puedes seleccionar múltiples archivos JSON. Capacidad: hasta 5,000+ archivos."
    )
    
    if uploaded_files:
        total_archivos = len(uploaded_files)
        
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <span class="file-counter">📁 {total_archivos:,} archivo(s) seleccionado(s)</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Archivos", f"{total_archivos:,}")
        
        with col2:
            tamano_total = sum(f.size for f in uploaded_files) / 1024 / 1024
            st.metric("💾 Tamaño Total", f"{tamano_total:.2f} MB")
        
        with col3:
            st.metric("📊 Estado", "Listo para procesar")
        
        st.markdown("---")
        
        st.markdown("### 📋 Vista Previa de Archivos")
        
        with st.expander("👁️ Ver lista de archivos seleccionados", expanded=False):
            archivos_info = []
            for f in uploaded_files[:100]:
                archivos_info.append({
                    'Nombre': f.name,
                    'Tamaño (KB)': f"{f.size / 1024:.2f}"
                })
            
            df_archivos = pd.DataFrame(archivos_info)
            st.dataframe(df_archivos, use_container_width=True, hide_index=True)
            
            if total_archivos > 100:
                st.info(f"📝 Mostrando los primeros 100 de {total_archivos:,} archivos")
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            if st.button("🚀 PROCESAR Y CONSOLIDAR ARCHIVOS", type="primary", use_container_width=True, key='btn_procesar'):
                
                limpiar_estado()
                
                st.markdown("""
                <div style="text-align: center; margin: 1rem 0;">
                    <span class="processing-indicator">⏳ Procesando archivos...</span>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                tiempo_inicio = time.time()
                
                df_consolidado = procesar_archivos_json(uploaded_files, progress_bar, status_text)
                
                tiempo_fin = time.time()
                tiempo_total = tiempo_fin - tiempo_inicio
                
                progress_bar.empty()
                status_text.empty()
                
                if df_consolidado is not None and not df_consolidado.empty:
                    st.session_state.df_consolidado = df_consolidado
                    st.session_state.consolidacion_completada = True
                    
                    st.session_state.historial_procesamiento.append({
                        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'archivos_procesados': st.session_state.archivos_procesados,
                        'registros_generados': len(df_consolidado),
                        'tiempo_segundos': round(tiempo_total, 2)
                    })
                    
                    st.markdown("""
                    <div class="success-box">
                    <h3>✅ ¡Consolidación Exitosa!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("✅ Archivos Procesados", f"{st.session_state.archivos_procesados:,}")
                    
                    with col2:
                        st.metric("📊 Registros Totales", f"{len(df_consolidado):,}")
                    
                    with col3:
                        st.metric("📋 Columnas", f"{len(df_consolidado.columns)}")
                    
                    with col4:
                        st.metric("⏱️ Tiempo", f"{tiempo_total:.2f} seg")
                    
                    if st.session_state.archivos_con_error:
                        st.markdown(f"""
                        <div class="warning-box">
                        <h4>⚠️ {len(st.session_state.archivos_con_error)} archivo(s) con errores</h4>
                        <p>Algunos archivos no pudieron ser procesados. Ver detalles en la pestaña "Resumen".</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.success("👉 Ve a la pestaña '📊 Vista de Datos' para ver los datos consolidados")
                    st.info("👉 Ve a la pestaña '➕ Campo Personalizado' para agregar una columna adicional")
                    
                    st.balloons()
                
                else:
                    st.markdown("""
                    <div class="error-box">
                    <h3>❌ Error en la Consolidación</h3>
                    <p>No se pudieron procesar los archivos. Verifica que sean archivos JSON válidos.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.3);">
            <h2 style="color: #667eea;">📁 Selecciona archivos JSON</h2>
            <p style="color: #764ba2;">Arrastra archivos aquí o haz clic en "Browse files"</p>
            <p style="color: #a0aec0;">Capacidad: hasta 5,000+ archivos JSON</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: VISTA DE DATOS
# =============================================================================

with tabs[1]:
    st.markdown("## 📊 Vista de Datos Consolidados")
    
    if st.session_state.df_consolidado is not None and not st.session_state.df_consolidado.empty:
        df_mostrar = st.session_state.df_consolidado
        
        st.markdown("### 📈 Estadísticas del Dataset")
        
        stats = obtener_estadisticas_df(df_mostrar)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Registros", f"{stats['total_registros']:,}")
        
        with col2:
            st.metric("📋 Columnas", f"{stats['total_columnas']}")
        
        with col3:
            st.metric("📁 Archivos", f"{stats['archivos_unicos']:,}")
        
        with col4:
            st.metric("❓ Valores Nulos", f"{stats['valores_nulos']:,}")
        
        with col5:
            st.metric("💾 Memoria", f"{stats['memoria_mb']:.2f} MB")
        
        st.markdown("---")
        
        st.markdown("### 🔍 Filtros y Búsqueda")
        
        col_filtro1, col_filtro2 = st.columns(2)
        
        with col_filtro1:
            if '_nombre_archivo_origen' in df_mostrar.columns:
                archivos_unicos = ['Todos'] + sorted(df_mostrar['_nombre_archivo_origen'].unique().tolist())
                filtro_archivo = st.selectbox(
                    "Filtrar por archivo de origen:",
                    archivos_unicos,
                    key='filtro_archivo'
                )
            else:
                filtro_archivo = 'Todos'
        
        with col_filtro2:
            max_registros = min(1000, len(df_mostrar))
            num_registros = st.slider(
                "Número de registros a mostrar:",
                min_value=10,
                max_value=max_registros,
                value=min(100, max_registros),
                step=10,
                key='num_registros_mostrar'
            )
        
        if '_nombre_archivo_origen' in df_mostrar.columns and filtro_archivo != 'Todos':
            df_filtrado = df_mostrar[df_mostrar['_nombre_archivo_origen'] == filtro_archivo].copy()
        else:
            df_filtrado = df_mostrar.copy()
        
        st.markdown("---")
        
        st.markdown("### 📋 Tabla de Datos")
        
        st.dataframe(
            df_filtrado.head(num_registros),
            use_container_width=True,
            height=500
        )
        
        st.caption(f"Mostrando {min(num_registros, len(df_filtrado)):,} de {len(df_filtrado):,} registros")
        
        st.markdown("---")
        
        st.markdown("### 📊 Información de Columnas")
        
        with st.expander("👁️ Ver detalles de columnas", expanded=False):
            columnas_info = []
            for col in df_mostrar.columns:
                col_data = df_mostrar[col]
                columnas_info.append({
                    'Columna': col,
                    'Tipo': str(col_data.dtype),
                    'No Nulos': int(col_data.notna().sum()),
                    'Nulos': int(col_data.isna().sum()),
                    '% Nulos': f"{(col_data.isna().sum() / len(df_mostrar)) * 100:.1f}%",
                    'Valores Únicos': int(col_data.nunique())
                })
            
            df_columnas_info = pd.DataFrame(columnas_info)
            st.dataframe(df_columnas_info, use_container_width=True, hide_index=True)
    
    else:
        st.markdown("""
        <div class="warning-box">
        <h3>📭 No hay datos cargados</h3>
        <p>Ve a la pestaña "📤 Cargar Archivos" para cargar y consolidar archivos JSON.</p>
        </div>
        """, unsafe_allow_html=True)
# =============================================================================
# TAB 3: CAMPO PERSONALIZADO
# =============================================================================

with tabs[2]:
    st.markdown("## ➕ Agregar Campo Personalizado")
    
    if st.session_state.df_consolidado is not None and not st.session_state.df_consolidado.empty:
        
        st.markdown("""
        <div class="info-box">
        <h4>🏷️ ¿Quieres agregar un campo personalizado a este conjunto de datos?</h4>
        <p>Puedes agregar una columna adicional con un valor que se aplicará a todos los registros.</p>
        <p><strong>Ejemplo:</strong> Si todos estos datos son de "Aguascalientes", puedes agregar una columna llamada "Estado" con el valor "Aguascalientes".</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📝 Configuración del Campo Personalizado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nombre_columna = st.text_input(
                "📋 Nombre de la nueva columna:",
                value=st.session_state.campo_personalizado_nombre,
                placeholder="Ej: Estado, Región, Categoría, Origen...",
                key='input_nombre_columna',
                help="Escribe el nombre que tendrá la nueva columna"
            )
        
        with col2:
            valor_columna = st.text_input(
                "✏️ Valor para todos los registros:",
                value=st.session_state.campo_personalizado_valor,
                placeholder="Ej: Aguascalientes, Norte, Tipo A...",
                key='input_valor_columna',
                help="Este valor se asignará a todos los registros del dataset"
            )
        
        st.markdown("---")
        
        if nombre_columna and valor_columna:
            st.markdown("### 👁️ Vista Previa")
            
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Configuración Lista</h4>
            <p><strong>Nueva columna:</strong> "{nombre_columna}"</p>
            <p><strong>Valor asignado:</strong> "{valor_columna}"</p>
            <p><strong>Se aplicará a:</strong> {len(st.session_state.df_consolidado):,} registros</p>
            </div>
            """, unsafe_allow_html=True)
            
            df_preview = st.session_state.df_consolidado.head(5).copy()
            df_preview[nombre_columna] = valor_columna
            
            # Reordenar columnas para vista previa
            if '_nombre_archivo_origen' in df_preview.columns:
                cols = df_preview.columns.tolist()
                cols.remove(nombre_columna)
                idx = cols.index('_nombre_archivo_origen') + 1
                cols.insert(idx, nombre_columna)
                df_preview = df_preview.reindex(columns=cols)
            
            st.markdown("#### 📋 Vista previa de los primeros 5 registros:")
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            
            with col_btn2:
                if st.button("✅ AGREGAR CAMPO AL DATASET", type="primary", use_container_width=True, key='btn_agregar_campo'):
                    
                    st.session_state.df_consolidado = agregar_campo_personalizado(
                        st.session_state.df_consolidado,
                        nombre_columna,
                        valor_columna
                    )
                    
                    st.session_state.campo_personalizado_nombre = nombre_columna
                    st.session_state.campo_personalizado_valor = valor_columna
                    st.session_state.campo_agregado = True
                    
                    st.markdown("""
                    <div class="success-box">
                    <h3>✅ ¡Campo Agregado Exitosamente!</h3>
                    <p>La nueva columna ha sido añadida a todos los registros del dataset.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success(f"✅ Columna '{nombre_columna}' agregada con valor '{valor_columna}' a {len(st.session_state.df_consolidado):,} registros")
                    
                    st.info("👉 Ve a la pestaña '📥 Descargar Datos' para exportar el dataset con la nueva columna")
                    
                    st.balloons()
        
        else:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Completa los campos</h4>
            <p>Ingresa tanto el nombre de la columna como el valor que deseas asignar.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.session_state.campo_agregado:
            st.markdown("### 📊 Campos Personalizados Agregados")
            
            st.markdown(f"""
            <div class="info-box">
            <p>✅ <strong>Columna:</strong> {st.session_state.campo_personalizado_nombre}</p>
            <p>✅ <strong>Valor:</strong> {st.session_state.campo_personalizado_valor}</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="warning-box">
        <h3>📭 No hay datos cargados</h3>
        <p>Primero debes cargar y consolidar archivos JSON en la pestaña "📤 Cargar Archivos".</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 4: DESCARGAR DATOS
# =============================================================================

with tabs[3]:
    st.markdown("## 📥 Descargar Datos Consolidados")
    
    if st.session_state.df_consolidado is not None and not st.session_state.df_consolidado.empty:
        
        df_descarga = st.session_state.df_consolidado
        
        st.markdown("""
        <div class="success-box">
        <h4>✅ Datos listos para descargar</h4>
        <p>Tu dataset consolidado está listo para ser exportado.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Resumen del Dataset a Descargar")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Registros", f"{len(df_descarga):,}")
        
        with col2:
            st.metric("📋 Columnas", f"{len(df_descarga.columns)}")
        
        with col3:
            if '_nombre_archivo_origen' in df_descarga.columns:
                archivos_origen = df_descarga['_nombre_archivo_origen'].nunique()
            else:
                archivos_origen = 0
            st.metric("📁 Archivos Origen", f"{archivos_origen:,}")
        
        with col4:
            memoria_mb = df_descarga.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Tamaño Aprox.", f"{memoria_mb:.2f} MB")
        
        st.markdown("---")
        
        st.markdown("### 📋 Columnas Incluidas")
        
        with st.expander("👁️ Ver lista de columnas", expanded=False):
            for i, col in enumerate(df_descarga.columns, 1):
                if col == '_nombre_archivo_origen':
                    st.markdown(f"**{i}. {col}** 📁 *(Archivo de origen)*")
                elif st.session_state.campo_agregado and col == st.session_state.campo_personalizado_nombre:
                    st.markdown(f"**{i}. {col}** 🏷️ *(Campo personalizado: {st.session_state.campo_personalizado_valor})*")
                else:
                    st.markdown(f"{i}. {col}")
        
        st.markdown("---")
        
        st.markdown("### 📥 Opciones de Descarga")
        
        st.markdown("""
        <div class="download-section">
        <h4>🎯 Selecciona el formato de descarga</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_csv, col_excel = st.columns(2)
        
        with col_csv:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%); border-radius: 15px; margin-bottom: 1rem;">
                <h3 style="color: #48bb78;">📄 Formato CSV</h3>
                <p style="color: #38a169;">Archivo de texto separado por comas</p>
                <p style="font-size: 0.8rem; color: #718096;">Compatible con Excel, Google Sheets, bases de datos</p>
            </div>
            """, unsafe_allow_html=True)
            
            nombre_archivo_csv = st.text_input(
                "Nombre del archivo CSV:",
                value=f"datos_consolidados_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key='nombre_csv'
            )
            
            csv_data = convertir_df_a_csv(df_descarga)
            
            st.download_button(
                label="⬇️ DESCARGAR CSV",
                data=csv_data,
                file_name=f"{nombre_archivo_csv}.csv",
                mime='text/csv',
                use_container_width=True,
                key='btn_download_csv'
            )
        
        with col_excel:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; margin-bottom: 1rem;">
                <h3 style="color: #667eea;">📊 Formato Excel</h3>
                <p style="color: #764ba2;">Archivo de hoja de cálculo (.xlsx)</p>
                <p style="font-size: 0.8rem; color: #718096;">Formato nativo de Microsoft Excel</p>
            </div>
            """, unsafe_allow_html=True)
            
            nombre_archivo_excel = st.text_input(
                "Nombre del archivo Excel:",
                value=f"datos_consolidados_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key='nombre_excel'
            )
            
            excel_data = convertir_df_a_excel(df_descarga)
            
            st.download_button(
                label="⬇️ DESCARGAR EXCEL",
                data=excel_data,
                file_name=f"{nombre_archivo_excel}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True,
                key='btn_download_excel'
            )
        
        st.markdown("---")
        
        st.markdown("### 👁️ Vista Previa de Datos a Descargar")
        
        max_preview = min(100, len(df_descarga))
        num_preview = st.slider(
            "Registros a mostrar en vista previa:",
            min_value=5,
            max_value=max_preview,
            value=min(20, max_preview),
            key='preview_descarga'
        )
        
        st.dataframe(df_descarga.head(num_preview), use_container_width=True, hide_index=True)
        
        st.caption(f"Vista previa: {num_preview} de {len(df_descarga):,} registros totales")
    
    else:
        st.markdown("""
        <div class="warning-box">
        <h3>📭 No hay datos para descargar</h3>
        <p>Primero debes cargar y consolidar archivos JSON en la pestaña "📤 Cargar Archivos".</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 5: RESUMEN
# =============================================================================

with tabs[4]:
    st.markdown("## 📋 Resumen del Proceso")
    
    if st.session_state.consolidacion_completada:
        
        st.markdown("""
        <div class="success-box">
        <h3>✅ Consolidación Completada</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Estadísticas Generales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">{st.session_state.archivos_procesados:,}</p>
                <p class="stat-label">Archivos Procesados</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
                <p class="stat-number">{st.session_state.total_registros:,}</p>
                <p class="stat-label">Registros Totales</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.df_consolidado is not None:
                num_cols = len(st.session_state.df_consolidado.columns)
            else:
                num_cols = 0
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);">
                <p class="stat-number">{num_cols}</p>
                <p class="stat-label">Columnas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);">
                <p class="stat-number">{len(st.session_state.archivos_con_error)}</p>
                <p class="stat-label">Errores</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.session_state.campo_agregado:
            st.markdown("### 🏷️ Campo Personalizado Agregado")
            
            st.markdown(f"""
            <div class="info-box">
            <p><strong>Columna:</strong> {st.session_state.campo_personalizado_nombre}</p>
            <p><strong>Valor:</strong> {st.session_state.campo_personalizado_valor}</p>
            <p><strong>Aplicado a:</strong> {st.session_state.total_registros:,} registros</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        if st.session_state.archivos_con_error:
            st.markdown("### ⚠️ Archivos con Errores")
            
            st.markdown(f"""
            <div class="error-box">
            <h4>Se encontraron {len(st.session_state.archivos_con_error)} archivo(s) con errores</h4>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("👁️ Ver detalles de errores", expanded=True):
                df_errores = pd.DataFrame(st.session_state.archivos_con_error)
                df_errores.columns = ['Archivo', 'Error']
                st.dataframe(df_errores, use_container_width=True, hide_index=True)
        
        else:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Todos los archivos procesados correctamente</h4>
            <p>No se encontraron errores durante el procesamiento.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.session_state.historial_procesamiento:
            st.markdown("### 📜 Historial de Procesamiento")
            
            df_historial = pd.DataFrame(st.session_state.historial_procesamiento)
            df_historial.columns = ['Fecha/Hora', 'Archivos', 'Registros', 'Tiempo (seg)']
            st.dataframe(df_historial, use_container_width=True, hide_index=True)
    
    else:
        st.markdown("""
        <div class="warning-box">
        <h3>📭 Sin datos procesados</h3>
        <p>Aún no has consolidado ningún archivo. Ve a la pestaña "📤 Cargar Archivos" para comenzar.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📋 Pasos para usar el sistema:</h4>
        <ol>
            <li><strong>Cargar:</strong> Selecciona tus archivos JSON en la pestaña "📤 Cargar Archivos"</li>
            <li><strong>Procesar:</strong> Haz clic en "PROCESAR Y CONSOLIDAR ARCHIVOS"</li>
            <li><strong>Personalizar:</strong> Opcionalmente, agrega un campo personalizado</li>
            <li><strong>Descargar:</strong> Exporta tus datos en CSV o Excel</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 2rem; color: #667eea;">
    <h3>📁 Consolidador Masivo de Archivos JSON</h3>
    <p>Sistema de Carga y Unificación de Datos</p>
    <p><strong>Capacidad: Hasta 5,000+ archivos JSON</strong></p>
    <p style="font-size: 0.9rem; color: #a0aec0;">
        Basado en diseño AURORA-ETHICS
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FIN DEL CÓDIGO
# =============================================================================