import chardet
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import json
import zipfile
import os
import tempfile
from collections import Counter
import string
import random
import hashlib
import math
import time
import warnings
import unicodedata
import re
import pickle
from io import BytesIO
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from scipy.stats import gaussian_kde, stats, chi2, zscore
from scipy.spatial.distance import cdist
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


UMBRAL_ALERTA_INCREMENTO = 100.0  # Porcentaje de incremento patrimonial inusual
UMBRAL_RATIO_BIENES_INGRESOS = 3.0  # Ratio bienes/ingresos que genera alerta
UMBRAL_RATIO_ADEUDOS_INGRESOS = 1.5  # Ratio adeudos/ingresos que genera alerta

# Lista de estados válidos para normalización
ESTADOS_VALIDOS = [
    'BAJA CALIFORNIA',
    'BAJA CALIFORNIA SUR',
    'SINALOA',
    'CIUDAD DE MEXICO',
    'ESTADO DE MEXICO',
    'ZACATECAS',
    'NAYARIT',
    'MICHOACAN',
    'SAN LUIS POTOSI',
    'GUANAJUATO',
    'TABASCO',
    'MORELOS',
    'GUERRERO',
    'CHIHUAHUA',
    'CHIAPAS',
    'AGUASCALIENTES',
    'JALISCO',
    'QUINTANA ROO'
]

# Coordenadas geográficas de estados
coordenadas_estados = {
    'BAJA CALIFORNIA': (30.8406, -115.2838),
    'BAJA CALIFORNIA SUR': (26.0444, -111.6661),
    'SINALOA': (25.0000, -107.5000),
    'CIUDAD DE MEXICO': (19.4326, -99.1332),
    'ESTADO DE MEXICO': (19.4969, -99.7233),
    'ZACATECAS': (22.7709, -102.5832),
    'NAYARIT': (21.7514, -104.8455),
    'MICHOACAN': (19.5665, -101.7068),
    'SAN LUIS POTOSI': (22.1565, -100.9855),
    'GUANAJUATO': (21.0190, -101.2574),
    'TABASCO': (17.8409, -92.6189),
    'MORELOS': (18.6813, -99.1013),
    'GUERRERO': (17.4392, -99.5451),
    'CHIHUAHUA': (28.6330, -106.0691),
    'CHIAPAS': (16.7569, -93.1292),
    'AGUASCALIENTES': (21.8853, -102.2916),
    'JALISCO': (20.6595, -103.3494),
    'QUINTANA ROO': (19.1817, -88.4791)
}   
st.set_page_config(
    page_title="AURORA-ETHICS M.A.S.C. & S.A.H.",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

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
    @keyframes colorPulse {
        0% { background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%); }
        25% { background: linear-gradient(135deg, rgba(0, 71, 187, 0.95) 0%, rgba(102, 126, 234, 0.95) 100%); }
        50% { background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%); }
        75% { background: linear-gradient(135deg, rgba(0, 71, 187, 0.95) 0%, rgba(102, 126, 234, 0.95) 100%); }
        100% { background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%); }
    }
    .hero-header {
        animation: colorPulse 4s ease-in-out infinite;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
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
</style>
""", unsafe_allow_html=True)

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'dataset_consolidado' not in st.session_state:
    st.session_state.dataset_consolidado = None
if 'dataset_seleccionado' not in st.session_state:
    st.session_state.dataset_seleccionado = None
if 'encoding_dicts' not in st.session_state:
    st.session_state.encoding_dicts = {}
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'df_procesado' not in st.session_state:
    st.session_state.df_procesado = None
if 'df_metricas_persona' not in st.session_state:
    st.session_state.df_metricas_persona = None
if 'df_servidores_unicos' not in st.session_state:
    st.session_state.df_servidores_unicos = None
if 'df_anomalias' not in st.session_state:
    st.session_state.df_anomalias = None
if 'df_alertas' not in st.session_state:
    st.session_state.df_alertas = None
if 'cargos_publicos' not in st.session_state:
    st.session_state.cargos_publicos = None
if 'denuncias' not in st.session_state:
    st.session_state.denuncias = []
if 'df_consolidado' not in st.session_state:
    st.session_state.df_consolidado = None

if 'archivos_procesados' not in st.session_state:
    st.session_state.archivos_procesados = 0

if 'total_registros' not in st.session_state:
    st.session_state.total_registros = 0

if 'archivos_con_error' not in st.session_state:
    st.session_state.archivos_con_error = []

if 'archivos_en_zip' not in st.session_state:
    st.session_state.archivos_en_zip = []

if 'campo_agregado' not in st.session_state:
    st.session_state.campo_agregado = False

if 'campo_personalizado_nombre' not in st.session_state:
    st.session_state.campo_personalizado_nombre = ""

if 'campo_personalizado_valor' not in st.session_state:
    st.session_state.campo_personalizado_valor = ""
COLUMNAS_REQUERIDAS = [
    '_nombre_archivo_origen',
    'Estado',
    'id',
    'anioEjercicio',
    'metadata_tipo',
    'metadata_actualizacion',
    'declaracion_situacionPatrimonial_datosGenerales_nombre',
    'declaracion_situacionPatrimonial_datosGenerales_primerApellido',
    'declaracion_situacionPatrimonial_datosGenerales_segundoApellido',
    'declaracion_situacionPatrimonial_datosGenerales_curp',
    'declaracion_situacionPatrimonial_datosGenerales_rfc_rfc',
    'declaracion_situacionPatrimonial_ingresos_remuneracionMensualCargoPublico_valor',
    'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor',
    'declaracion_situacionPatrimonial_ingresos_ingresoMensualNetoDeclarante_valor',
    'declaracion_situacionPatrimonial_ingresos_ingresoAnualNetoDeclarante_valor',
    'declaracion_situacionPatrimonial_ingresos_totalIngresosMensualesNetos_valor',
    'declaracion_situacionPatrimonial_ingresos_totalIngresosAnualesNetos_valor',
    'declaracion_situacionPatrimonial_ingresos_otrosIngresosMensualesTotal_valor',
    'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nivelEmpleoCargoComision',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_fechaTomaPosesion',
    'declaracion_situacionPatrimonial_domicilioDeclarante_calle',
    'declaracion_situacionPatrimonial_domicilioDeclarante_numeroExterior',
    'declaracion_situacionPatrimonial_domicilioDeclarante_numeroInterior',
    'declaracion_situacionPatrimonial_domicilioDeclarante_coloniaLocalidad',
    'declaracion_situacionPatrimonial_domicilioDeclarante_municipioAlcaldia',
    'declaracion_situacionPatrimonial_domicilioDeclarante_entidadFederativa',
    'declaracion_situacionPatrimonial_domicilioDeclarante_codigoPostal',
    'declaracion_situacionPatrimonial_domicilioDeclarante_pais',
    'declaracion_situacionPatrimonial_datosPareja_nombre',
    'declaracion_situacionPatrimonial_datosPareja_primerApellido',
    'declaracion_situacionPatrimonial_datosPareja_segundoApellido',
    'declaracion_situacionPatrimonial_datosPareja_fechaNacimiento',
    'declaracion_situacionPatrimonial_datosPareja_rfc_rfc',
    'declaracion_situacionPatrimonial_datosPareja_curp',
    'declaracion_situacionPatrimonial_datosPareja_relacionConDeclarante',
    'declaracion_situacionPatrimonial_datosPareja_ciudadanoExtranjero',
    'declaracion_situacionPatrimonial_datosPareja_habitaDomicilioDeclarante',
    'declaracion_situacionPatrimonial_datosPareja_lugarDondeReside',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_nombre',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_primerApellido',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_segundoApellido',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_fechaNacimiento',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_rfc_rfc',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_relacionConDeclarante',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_habitaDomicilioDeclarante',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_0_lugarDondeReside',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_nombre',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_primerApellido',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_segundoApellido',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_fechaNacimiento',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_rfc_rfc',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_relacionConDeclarante',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_habitaDomicilioDeclarante',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_1_lugarDondeReside',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_nombre',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_primerApellido',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_segundoApellido',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_fechaNacimiento',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_rfc_rfc',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_relacionConDeclarante',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_habitaDomicilioDeclarante',
    'declaracion_situacionPatrimonial_datosDependientesEconomicos_2_lugarDondeReside',
    'declaracion_situacionPatrimonial_ingresos_actividadIndustrial_0_remuneracionTotal_valor',
    'declaracion_situacionPatrimonial_ingresos_actividadIndustrial_0_remuneracionTotal_moneda',
    'declaracion_situacionPatrimonial_ingresos_actividadIndustrial_0_nombreRazonSocial',
    'declaracion_situacionPatrimonial_ingresos_actividadIndustrial_0_tipoNegocio',
    'declaracion_situacionPatrimonial_ingresos_actividadFinanciera_0_remuneracionTotal_valor',
    'declaracion_situacionPatrimonial_ingresos_actividadFinanciera_0_remuneracionTotal_moneda',
    'declaracion_situacionPatrimonial_ingresos_actividadFinanciera_0_tipoInstrumento',
    'declaracion_situacionPatrimonial_ingresos_serviciosProfesionales_0_remuneracionTotal_valor',
    'declaracion_situacionPatrimonial_ingresos_serviciosProfesionales_0_remuneracionTotal_moneda',
    'declaracion_situacionPatrimonial_ingresos_serviciosProfesionales_0_tipoServicio',
    'declaracion_situacionPatrimonial_ingresos_enajenacionBienes_0_remuneracionTotal_valor',
    'declaracion_situacionPatrimonial_ingresos_enajenacionBienes_0_remuneracionTotal_moneda',
    'declaracion_situacionPatrimonial_ingresos_enajenacionBienes_0_tipoBienEnajenado',
    'declaracion_situacionPatrimonial_ingresos_otrosIngresos_0_remuneracionTotal_valor',
    'declaracion_situacionPatrimonial_ingresos_otrosIngresos_0_remuneracionTotal_moneda',
    'declaracion_situacionPatrimonial_ingresos_otrosIngresos_0_tipoIngreso',
    'declaracion_situacionPatrimonial_ingresos_ingresoNetoParejaDependiente_valor',
    'declaracion_situacionPatrimonial_ingresos_ingresoNetoParejaDependiente_moneda',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_tipoInmueble',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_titular',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_porcentajePropiedad',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_superficieTerreno_valor',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_superficieTerreno_unidad',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_superficieConstruccion_valor',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_superficieConstruccion_unidad',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_tercero_nombreRazonSocial',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_tercero_rfc',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_calle',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_numeroExterior',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_numeroInterior',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_coloniaLocalidad',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_municipioAlcaldia',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_entidadFederativa',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_codigoPostal',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_domicilio_pais',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_valorAdquisicion_valor',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_valorAdquisicion_moneda',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_fechaAdquisicion',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_formaAdquisicion',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_formaPago',
    'declaracion_situacionPatrimonial_bienesInmuebles_0_valorConformeA',
    'declaracion_situacionPatrimonial_bienesInmuebles_1_tipoInmueble',
    'declaracion_situacionPatrimonial_bienesInmuebles_1_titular',
    'declaracion_situacionPatrimonial_bienesInmuebles_1_porcentajePropiedad',
    'declaracion_situacionPatrimonial_bienesInmuebles_1_valorAdquisicion_valor',
    'declaracion_situacionPatrimonial_bienesInmuebles_1_valorAdquisicion_moneda',
    'declaracion_situacionPatrimonial_bienesInmuebles_1_fechaAdquisicion',
    'declaracion_situacionPatrimonial_vehiculos_0_tipoVehiculo',
    'declaracion_situacionPatrimonial_vehiculos_0_titular',
    'declaracion_situacionPatrimonial_vehiculos_0_marca',
    'declaracion_situacionPatrimonial_vehiculos_0_modelo',
    'declaracion_situacionPatrimonial_vehiculos_0_anio',
    'declaracion_situacionPatrimonial_vehiculos_0_numeroSerieRegistro',
    'declaracion_situacionPatrimonial_vehiculos_0_lugarRegistro',
    'declaracion_situacionPatrimonial_vehiculos_0_tercero_nombreRazonSocial',
    'declaracion_situacionPatrimonial_vehiculos_0_tercero_rfc',
    'declaracion_situacionPatrimonial_vehiculos_0_valorAdquisicion_valor',
    'declaracion_situacionPatrimonial_vehiculos_0_valorAdquisicion_moneda',
    'declaracion_situacionPatrimonial_vehiculos_0_fechaAdquisicion',
    'declaracion_situacionPatrimonial_vehiculos_0_formaAdquisicion',
    'declaracion_situacionPatrimonial_vehiculos_0_formaPago',
    'declaracion_situacionPatrimonial_vehiculos_1_tipoVehiculo',
    'declaracion_situacionPatrimonial_vehiculos_1_titular',
    'declaracion_situacionPatrimonial_vehiculos_1_marca',
    'declaracion_situacionPatrimonial_vehiculos_1_modelo',
    'declaracion_situacionPatrimonial_vehiculos_1_anio',
    'declaracion_situacionPatrimonial_vehiculos_1_valorAdquisicion_valor',
    'declaracion_situacionPatrimonial_vehiculos_1_valorAdquisicion_moneda',
    'declaracion_situacionPatrimonial_vehiculos_1_fechaAdquisicion',
    'declaracion_situacionPatrimonial_bienesMuebles_0_tipoBien',
    'declaracion_situacionPatrimonial_bienesMuebles_0_titular',
    'declaracion_situacionPatrimonial_bienesMuebles_0_descripcion',
    'declaracion_situacionPatrimonial_bienesMuebles_0_tercero_nombreRazonSocial',
    'declaracion_situacionPatrimonial_bienesMuebles_0_tercero_rfc',
    'declaracion_situacionPatrimonial_bienesMuebles_0_valorAdquisicion_valor',
    'declaracion_situacionPatrimonial_bienesMuebles_0_valorAdquisicion_moneda',
    'declaracion_situacionPatrimonial_bienesMuebles_0_fechaAdquisicion',
    'declaracion_situacionPatrimonial_bienesMuebles_0_formaAdquisicion',
    'declaracion_situacionPatrimonial_bienesMuebles_0_formaPago',
    'declaracion_situacionPatrimonial_bienesMuebles_1_tipoBien',
    'declaracion_situacionPatrimonial_bienesMuebles_1_titular',
    'declaracion_situacionPatrimonial_bienesMuebles_1_descripcion',
    'declaracion_situacionPatrimonial_bienesMuebles_1_valorAdquisicion_valor',
    'declaracion_situacionPatrimonial_bienesMuebles_1_valorAdquisicion_moneda',
    'declaracion_situacionPatrimonial_bienesMuebles_1_fechaAdquisicion',
    'declaracion_situacionPatrimonial_inversiones_0_tipoInversion',
    'declaracion_situacionPatrimonial_inversiones_0_bancoInstitucion',
    'declaracion_situacionPatrimonial_inversiones_0_tercero_nombreRazonSocial',
    'declaracion_situacionPatrimonial_inversiones_0_tercero_rfc',
    'declaracion_situacionPatrimonial_inversiones_0_titular',
    'declaracion_situacionPatrimonial_inversiones_0_localizacionInversion',
    'declaracion_situacionPatrimonial_inversiones_0_saldoSituacionActual_valor',
    'declaracion_situacionPatrimonial_inversiones_0_saldoSituacionActual_moneda',
    'declaracion_situacionPatrimonial_inversiones_1_tipoInversion',
    'declaracion_situacionPatrimonial_inversiones_1_bancoInstitucion',
    'declaracion_situacionPatrimonial_inversiones_1_titular',
    'declaracion_situacionPatrimonial_inversiones_1_saldoSituacionActual_valor',
    'declaracion_situacionPatrimonial_inversiones_1_saldoSituacionActual_moneda',
    'declaracion_situacionPatrimonial_adeudos_0_tipoAdeudo',
    'declaracion_situacionPatrimonial_adeudos_0_numeroCuentaContrato',
    'declaracion_situacionPatrimonial_adeudos_0_fechaAdquisicion',
    'declaracion_situacionPatrimonial_adeudos_0_montoOriginal_valor',
    'declaracion_situacionPatrimonial_adeudos_0_montoOriginal_moneda',
    'declaracion_situacionPatrimonial_adeudos_0_saldoInsoluto_valor',
    'declaracion_situacionPatrimonial_adeudos_0_saldoInsoluto_moneda',
    'declaracion_situacionPatrimonial_adeudos_0_titular',
    'declaracion_situacionPatrimonial_adeudos_0_tercero_nombreRazonSocial',
    'declaracion_situacionPatrimonial_adeudos_0_tercero_rfc',
    'declaracion_situacionPatrimonial_adeudos_0_localizacionAdeudo',
    'declaracion_situacionPatrimonial_adeudos_1_tipoAdeudo',
    'declaracion_situacionPatrimonial_adeudos_1_numeroCuentaContrato',
    'declaracion_situacionPatrimonial_adeudos_1_fechaAdquisicion',
    'declaracion_situacionPatrimonial_adeudos_1_montoOriginal_valor',
    'declaracion_situacionPatrimonial_adeudos_1_montoOriginal_moneda',
    'declaracion_situacionPatrimonial_adeudos_1_saldoInsoluto_valor',
    'declaracion_situacionPatrimonial_adeudos_1_saldoInsoluto_moneda',
    'declaracion_situacionPatrimonial_adeudos_1_titular',
    'declaracion_situacionPatrimonial_adeudos_2_tipoAdeudo',
    'declaracion_situacionPatrimonial_adeudos_2_saldoInsoluto_valor',
    'declaracion_situacionPatrimonial_adeudos_2_saldoInsoluto_moneda',
    'declaracion_situacionPatrimonial_prestamoComodato_0_tipoBien',
    'declaracion_situacionPatrimonial_prestamoComodato_0_tipoDueno',
    'declaracion_situacionPatrimonial_prestamoComodato_0_dueno_nombreRazonSocial',
    'declaracion_situacionPatrimonial_prestamoComodato_0_dueno_rfc',
    'declaracion_situacionPatrimonial_prestamoComodato_0_relacionConTitular',
    'declaracion_situacionPatrimonial_prestamoComodato_0_formaPago',
    'declaracion_situacionPatrimonial_prestamoComodato_0_valorConformeA',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_nombreEmpresa',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_rfc',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_puesto',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_fechaIngreso',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_fechaEgreso',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_sector',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_sectorOtro',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_ubicacion',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_salarioMensualNeto_valor',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_0_salarioMensualNeto_moneda',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_1_nombreEmpresa',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_1_puesto',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_1_fechaIngreso',
    'declaracion_situacionPatrimonial_actividadAnualAnterior_1_fechaEgreso',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_nombreEmpresa',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_rfc',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_porcentajeParticipacion',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_tipoParticipacion',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_recibeRemuneracion',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_montoMensual_valor',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_montoMensual_moneda',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_ubicacion',
    'declaracion_situacionPatrimonial_participacionEmpresa_0_sector',
    'declaracion_situacionPatrimonial_participacionEmpresa_1_nombreEmpresa',
    'declaracion_situacionPatrimonial_participacionEmpresa_1_porcentajeParticipacion',
    'declaracion_situacionPatrimonial_participacionEmpresa_1_tipoParticipacion',
    'declaracion_situacionPatrimonial_apoyos_0_nombrePrograma',
    'declaracion_situacionPatrimonial_apoyos_0_institucionOtorgante',
    'declaracion_situacionPatrimonial_apoyos_0_nivel',
    'declaracion_situacionPatrimonial_apoyos_0_tipoApoyo',
    'declaracion_situacionPatrimonial_apoyos_0_formaRecepcion',
    'declaracion_situacionPatrimonial_apoyos_0_monto_valor',
    'declaracion_situacionPatrimonial_apoyos_0_monto_moneda',
    'declaracion_situacionPatrimonial_representacion_0_tipoRepresentacion',
    'declaracion_situacionPatrimonial_representacion_0_nombreRazonSocial',
    'declaracion_situacionPatrimonial_representacion_0_rfc',
    'declaracion_situacionPatrimonial_representacion_0_fechaInicioRepresentacion',
    'declaracion_situacionPatrimonial_representacion_0_recibeRemuneracion',
    'declaracion_situacionPatrimonial_representacion_0_montoMensual_valor',
    'declaracion_situacionPatrimonial_representacion_0_montoMensual_moneda',
    'declaracion_situacionPatrimonial_representacion_0_ubicacion',
    'declaracion_situacionPatrimonial_representacion_0_sector',
    'declaracion_situacionPatrimonial_clientesPrincipales_0_clientePrincipal',
    'declaracion_situacionPatrimonial_clientesPrincipales_0_empresaDondeLabora',
    'declaracion_situacionPatrimonial_clientesPrincipales_0_rfc',
    'declaracion_situacionPatrimonial_clientesPrincipales_0_sector',
    'declaracion_situacionPatrimonial_clientesPrincipales_0_monto_valor',
    'declaracion_situacionPatrimonial_clientesPrincipales_0_monto_moneda',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_beneficiario',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_otorgante',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_formaRecepcion',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_especifiqueBeneficio',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_montoMensual_valor',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_montoMensual_moneda',
    'declaracion_situacionPatrimonial_beneficiosPrivados_0_sector',
    'declaracion_situacionPatrimonial_fideicomisos_0_tipoRelacion',
    'declaracion_situacionPatrimonial_fideicomisos_0_tipoFideicomiso',
    'declaracion_situacionPatrimonial_fideicomisos_0_participacion',
    'declaracion_situacionPatrimonial_fideicomisos_0_rfc',
    'declaracion_situacionPatrimonial_fideicomisos_0_fideicomitente',
    'declaracion_situacionPatrimonial_fideicomisos_0_fiduciario',
    'declaracion_situacionPatrimonial_fideicomisos_0_fideicomisario',
    'declaracion_situacionPatrimonial_fideicomisos_0_sector',
    'declaracion_situacionPatrimonial_fideicomisos_0_extranjero',
    # Columnas calculadas adicionales
    'Patrimonio_Total',
    'Ratio_Patrimonio_Ingreso',
    'Delta_Patrimonial_Anual',
    'Incremento_Porcentual_Patrimonio',
    'Años_en_Cargo',
    'Tiene_Conflicto_Interes',
    'Numero_Bienes_Totales',
    'Ratio_Deuda_Activos'
]
MODO_EXTRACCION = "TODAS"

COLUMNAS_ESPECIFICAS_BASICAS = [
    'id',
    'anioEjercicio',
    'metadata_tipo',
    'metadata_actualizacion',
    'declaracion_situacionPatrimonial_datosGenerales_nombre',
    'declaracion_situacionPatrimonial_datosGenerales_primerApellido',
    'declaracion_situacionPatrimonial_datosGenerales_segundoApellido',
    'declaracion_situacionPatrimonial_datosGenerales_curp',
    'declaracion_situacionPatrimonial_datosGenerales_rfc_rfc',
    'declaracion_situacionPatrimonial_ingresos_remuneracionMensualCargoPublico_valor',
    'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor',
    'declaracion_situacionPatrimonial_ingresos_ingresoMensualNetoDeclarante_valor',
    'declaracion_situacionPatrimonial_ingresos_ingresoAnualNetoDeclarante_valor',
    'declaracion_situacionPatrimonial_ingresos_totalIngresosMensualesNetos_valor',
    'declaracion_situacionPatrimonial_ingresos_totalIngresosAnualesNetos_valor',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nivelEmpleoCargoComision',
    'declaracion_situacionPatrimonial_datosEmpleoCargoComision_fechaTomaPosesion'
]

def limpiar_estado():
    """Limpia todas las variables de sesión"""
    st.session_state.df_consolidado = None
    st.session_state.archivos_procesados = 0
    st.session_state.total_registros = 0
    st.session_state.archivos_con_error = []
    st.session_state.archivos_en_zip = []
    st.session_state.campo_agregado = False
    st.session_state.campo_personalizado_nombre = ""
    st.session_state.campo_personalizado_valor = ""

def flatten_json(nested_json, parent_key='', sep='_'):
    """
    Aplana un JSON anidado recursivamente
    """
    items = {}
    
    if isinstance(nested_json, dict):
        for key, value in nested_json.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(flatten_json(value, new_key, sep))
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    for i, item in enumerate(value[:10]):
                        items.update(flatten_json(item, f"{new_key}_{i}", sep))
                else:
                    items[new_key] = str(value) if value else ""
            else:
                items[new_key] = value
    elif isinstance(nested_json, list):
        for i, item in enumerate(nested_json[:10]):
            if isinstance(item, dict):
                items.update(flatten_json(item, f"{parent_key}_{i}" if parent_key else str(i), sep))
            else:
                items[f"{parent_key}_{i}" if parent_key else str(i)] = item
    else:
        items[parent_key] = nested_json
    
    return items

def procesar_json_individual(contenido_json, nombre_archivo):
    """
    Procesa un archivo JSON individual y lo convierte en lista de diccionarios
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

def extraer_archivos_json_del_zip(uploaded_zip):
    """
    Extrae todos los archivos JSON de un archivo ZIP
    Soporta archivos grandes (>2000 MB)
    """
    archivos_json = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, "uploaded.zip")
            
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_zip.getvalue())
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                json_files = [f for f in file_list 
                             if f.lower().endswith('.json') 
                             and not f.startswith('__MACOSX')
                             and not f.startswith('.')]
                
                st.session_state.archivos_en_zip = json_files
                
                for json_file in json_files:
                    try:
                        with zip_ref.open(json_file) as file:
                            contenido = file.read()
                            
                            archivos_json.append({
                                'nombre': os.path.basename(json_file),
                                'contenido': contenido,
                                'ruta_completa': json_file
                            })
                    
                    except Exception as e:
                        st.session_state.archivos_con_error.append({
                            'archivo': json_file,
                            'error': f"Error al extraer: {str(e)}"
                        })
    
    except zipfile.BadZipFile:
        st.error("❌ El archivo no es un ZIP válido")
        return []
    except Exception as e:
        st.error(f"❌ Error al procesar el ZIP: {str(e)}")
        return []
    
    return archivos_json

def procesar_archivos_json_individuales(uploaded_files, progress_bar, status_text):
    """
    Procesa archivos JSON individuales (sin ZIP)
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
        status_text.text("📊 Consolidando datos...")
        
        df_consolidado = pd.DataFrame(todos_registros)
        
        if '_nombre_archivo_origen' in df_consolidado.columns:
            cols = [col for col in df_consolidado.columns if col != '_nombre_archivo_origen']
            cols.insert(0, '_nombre_archivo_origen')
            df_consolidado = df_consolidado.reindex(columns=cols)
        
        st.session_state.total_registros = len(df_consolidado)
        
        return df_consolidado
    
    return None

def procesar_archivos_desde_zip(archivos_json, progress_bar, status_text):
    """
    Procesa múltiples archivos JSON extraídos del ZIP y los consolida
    """
    todos_registros = []
    total_archivos = len(archivos_json)
    archivos_exitosos = 0
    
    for idx, archivo_info in enumerate(archivos_json):
        try:
            status_text.text(f"📄 Procesando: {archivo_info['nombre']} ({idx + 1}/{total_archivos})")
            
            contenido = archivo_info['contenido']
            
            try:
                contenido_decodificado = contenido.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    contenido_decodificado = contenido.decode('latin-1')
                except UnicodeDecodeError:
                    contenido_decodificado = contenido.decode('utf-8', errors='ignore')
            
            datos_json = json.loads(contenido_decodificado)
            
            registros = procesar_json_individual(datos_json, archivo_info['nombre'])
            
            if registros:
                todos_registros.extend(registros)
                archivos_exitosos += 1
            
            progress_bar.progress((idx + 1) / total_archivos)
            
        except json.JSONDecodeError as e:
            st.session_state.archivos_con_error.append({
                'archivo': archivo_info['nombre'],
                'error': f"Error de formato JSON: {str(e)}"
            })
        except Exception as e:
            st.session_state.archivos_con_error.append({
                'archivo': archivo_info['nombre'],
                'error': str(e)
            })
    
    st.session_state.archivos_procesados = archivos_exitosos
    
    if todos_registros:
        status_text.text("📊 Consolidando datos...")
        
        df_consolidado = pd.DataFrame(todos_registros)
        
        if '_nombre_archivo_origen' in df_consolidado.columns:
            cols = [col for col in df_consolidado.columns if col != '_nombre_archivo_origen']
            cols.insert(0, '_nombre_archivo_origen')
            df_consolidado = df_consolidado.reindex(columns=cols)
        
        st.session_state.total_registros = len(df_consolidado)
        
        return df_consolidado
    
    return None

def convertir_a_numero_seguro(valor):
    """Convierte un valor a número de forma segura"""
    try:
        if pd.isna(valor) or valor == '' or valor is None:
            return np.nan
        return float(valor)
    except:
        return np.nan

def calcular_metricas_patrimoniales(df):
    """
    Calcula las columnas derivadas patrimoniales
    """
    df_calc = df.copy()
    
    df_calc['Patrimonio_Total'] = np.nan
    df_calc['Ratio_Patrimonio_Ingreso'] = np.nan
    df_calc['Delta_Patrimonial_Anual'] = np.nan
    df_calc['Incremento_Porcentual_Patrimonio'] = np.nan
    df_calc['Años_en_Cargo'] = np.nan
    df_calc['Tiene_Conflicto_Interes'] = np.nan
    df_calc['Numero_Bienes_Totales'] = np.nan
    df_calc['Ratio_Deuda_Activos'] = np.nan
    
    try:
        bienes_inmuebles = 0
        for i in range(10):
            col_inmueble = f'declaracion_situacionPatrimonial_bienesInmuebles_{i}_valorAdquisicion_valor'
            if col_inmueble in df_calc.columns:
                bienes_inmuebles += df_calc[col_inmueble].apply(convertir_a_numero_seguro)
        
        vehiculos = 0
        for i in range(10):
            col_vehiculo = f'declaracion_situacionPatrimonial_vehiculos_{i}_valorAdquisicion_valor'
            if col_vehiculo in df_calc.columns:
                vehiculos += df_calc[col_vehiculo].apply(convertir_a_numero_seguro)
        
        bienes_muebles = 0
        for i in range(10):
            col_mueble = f'declaracion_situacionPatrimonial_bienesMuebles_{i}_valorAdquisicion_valor'
            if col_mueble in df_calc.columns:
                bienes_muebles += df_calc[col_mueble].apply(convertir_a_numero_seguro)
        
        inversiones = 0
        for i in range(10):
            col_inversion = f'declaracion_situacionPatrimonial_inversiones_{i}_saldoSituacionActual_valor'
            if col_inversion in df_calc.columns:
                inversiones += df_calc[col_inversion].apply(convertir_a_numero_seguro)
        
        adeudos = 0
        for i in range(10):
            col_adeudo = f'declaracion_situacionPatrimonial_adeudos_{i}_saldoInsoluto_valor'
            if col_adeudo in df_calc.columns:
                adeudos += df_calc[col_adeudo].apply(convertir_a_numero_seguro)
        
        df_calc['Patrimonio_Total'] = bienes_inmuebles + vehiculos + bienes_muebles + inversiones - adeudos
    except:
        pass
    
    try:
        if 'declaracion_situacionPatrimonial_ingresos_totalIngresosAnualesNetos_valor' in df_calc.columns:
            ingresos_anuales = df_calc['declaracion_situacionPatrimonial_ingresos_totalIngresosAnualesNetos_valor'].apply(convertir_a_numero_seguro)
            df_calc['Ratio_Patrimonio_Ingreso'] = df_calc['Patrimonio_Total'] / ingresos_anuales
    except:
        pass
    
    try:
        if 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_fechaTomaPosesion' in df_calc.columns:
            fecha_actual = pd.Timestamp.now()
            df_calc['Años_en_Cargo'] = df_calc['declaracion_situacionPatrimonial_datosEmpleoCargoComision_fechaTomaPosesion'].apply(
                lambda x: (fecha_actual - pd.to_datetime(x, errors='coerce')).days / 365.25 if pd.notna(x) else np.nan
            )
    except:
        pass
    
    try:
        conteo_bienes = 0
        for i in range(10):
            col = f'declaracion_situacionPatrimonial_bienesInmuebles_{i}_tipoInmueble'
            if col in df_calc.columns:
                conteo_bienes += df_calc[col].notna().astype(int)
        for i in range(10):
            col = f'declaracion_situacionPatrimonial_vehiculos_{i}_tipoVehiculo'
            if col in df_calc.columns:
                conteo_bienes += df_calc[col].notna().astype(int)
        for i in range(10):
            col = f'declaracion_situacionPatrimonial_bienesMuebles_{i}_tipoBien'
            if col in df_calc.columns:
                conteo_bienes += df_calc[col].notna().astype(int)
        
        df_calc['Numero_Bienes_Totales'] = conteo_bienes
    except:
        pass
    
    try:
        adeudos_total = 0
        for i in range(10):
            col = f'declaracion_situacionPatrimonial_adeudos_{i}_saldoInsoluto_valor'
            if col in df_calc.columns:
                adeudos_total += df_calc[col].apply(convertir_a_numero_seguro)
        
        activos_total = df_calc['Patrimonio_Total'] + adeudos_total
        df_calc['Ratio_Deuda_Activos'] = adeudos_total / activos_total
    except:
        pass
    
    try:
        tiene_participacion = False
        for i in range(10):
            col = f'declaracion_situacionPatrimonial_participacionEmpresa_{i}_nombreEmpresa'
            if col in df_calc.columns:
                tiene_participacion = tiene_participacion | df_calc[col].notna()
        df_calc['Tiene_Conflicto_Interes'] = tiene_participacion.astype(int)
    except:
        pass
    
    return df_calc

def filtrar_columnas_requeridas(df):
    """
    Filtra el DataFrame para mantener solo las columnas requeridas.
    Crea columnas faltantes con valores NaN.
    PRESERVA el campo personalizado agregado (como Estado).
    """
    df_filtrado = pd.DataFrame()
    
    # Primero agregar todas las columnas requeridas
    for columna in COLUMNAS_REQUERIDAS:
        if columna in df.columns:
            df_filtrado[columna] = df[columna]
        else:
            df_filtrado[columna] = np.nan
    
    # IMPORTANTE: Si hay un campo personalizado agregado, preservarlo
    if st.session_state.campo_agregado and st.session_state.campo_personalizado_nombre:
        nombre_campo = st.session_state.campo_personalizado_nombre
        if nombre_campo in df.columns and nombre_campo not in df_filtrado.columns:
            # Insertarlo después de _nombre_archivo_origen
            if '_nombre_archivo_origen' in df_filtrado.columns:
                pos = df_filtrado.columns.get_loc('_nombre_archivo_origen') + 1
                df_filtrado.insert(pos, nombre_campo, df[nombre_campo])
            else:
                # Si no existe _nombre_archivo_origen, ponerlo al inicio
                df_filtrado.insert(0, nombre_campo, df[nombre_campo])
    
    return df_filtrado

def agregar_campo_personalizado(df, nombre_columna, valor):
    """
    Agrega una columna personalizada al DataFrame
    """
    if df is not None and not df.empty:
        df_copy = df.copy()
        df_copy[nombre_columna] = valor
        return df_copy
    return df

def generar_estadisticas():
    """Genera estadísticas del procesamiento"""
    if st.session_state.df_consolidado is not None:
        return {
            'total_archivos': st.session_state.archivos_procesados,
            'total_registros': st.session_state.total_registros,
            'total_columnas': len(st.session_state.df_consolidado.columns),
            'errores': len(st.session_state.archivos_con_error),
            'memoria_mb': st.session_state.df_consolidado.memory_usage(deep=True).sum() / 1024**2
        }
    return None


def calcular_hash_md5(data):
    """Calcula hash MD5 de datos"""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False).encode()
    elif isinstance(data, bytes):
        pass
    else:
        data = str(data).encode()
    return hashlib.md5(data).hexdigest()

def normalizar_texto(texto):
    """Normaliza texto: mayúsculas, sin acentos, sin espacios múltiples"""
    if pd.isna(texto):
        return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def normalizar_estado(estado):
    """Normaliza nombre de estado y lo valida contra lista permitida"""
    if pd.isna(estado):
        return None
    estado_norm = normalizar_texto(estado)
    # Intentar match exacto
    if estado_norm in ESTADOS_VALIDOS:
        return estado_norm
    # Intentar match fuzzy si está disponible
    if FUZZY_AVAILABLE:
        match = process.extractOne(estado_norm, ESTADOS_VALIDOS, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            return match[0]
    return None
def limpiar_valor_monetario(valor):
    """Limpia y convierte valores monetarios a float"""
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    # Eliminar símbolos de moneda, comas, paréntesis
    valor_str = str(valor).replace('$', '').replace(',', '').replace('(', '').replace(')', '').strip()
    try:
        return float(valor_str)
    except:
        return 0.0
def detectar_encoding(archivo):
    """Detecta el encoding de un archivo"""
    try:
        raw_data = archivo.read(10000)
        archivo.seek(0)
        result = chardet.detect(raw_data)
        return result['encoding']
    except:
        return 'utf-8'
def eliminar_fila_encabezados_duplicada(df):
    """Elimina filas duplicadas de encabezados dentro de los datos"""
    if len(df) <= 1:
        return df
    headers_original = df.columns.astype(str).str.strip().str.upper()
    filas_a_eliminar = []
    for idx in range(len(df)):
        fila_valores = df.iloc[idx].astype(str).str.strip().str.upper().values
        if np.array_equal(fila_valores, headers_original):
            filas_a_eliminar.append(idx)
    if filas_a_eliminar:
        df = df.drop(index=filas_a_eliminar).reset_index(drop=True)
        st.info(f"🔄 {len(filas_a_eliminar)} fila(s) duplicada(s) de encabezados eliminada(s) automáticamente")
    return df
def procesar_archivo_cargado(uploaded_file):
    """Procesa archivo cargado con detección automática de encoding y limpieza robusta"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        file_size = uploaded_file.size / (1024 * 1024)
        status_text.text(f"📥 Cargando {uploaded_file.name}... ({file_size:.2f} MB)")
        progress_bar.progress(10)
        start_time = datetime.now()
        
        # Leer archivo según extensión
        if uploaded_file.name.endswith('.csv'):
            encoding = detectar_encoding(uploaded_file)
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
            except:
                df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False, encoding_errors='ignore')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("❌ Formato no soportado. Use CSV, XLS o XLSX.")
            return None
        
        progress_bar.progress(30)
        status_text.text(f"🔄 Procesando datos... ({len(df)} registros)")
        
        # Eliminar filas duplicadas de encabezados
        df = eliminar_fila_encabezados_duplicada(df)
        
        progress_bar.progress(50)
        
        # Convertir columnas de fecha
        fecha_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['fecha', 'actualizacion', 'date'])]
        for col in fecha_cols:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            except:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        progress_bar.progress(70)
        
        # Convertir columnas monetarias
        columnas_monetarias = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['valor', 'monto', 'saldo', 'ingreso', 'adquisicion', 'precio', 'costo'])
        ]
        for col in columnas_monetarias:
            if col in df.columns:
                df[col] = df[col].apply(limpiar_valor_monetario)
        
        progress_bar.progress(85)
        
        # Crear NombreCompleto normalizado
        if all(col in df.columns for col in ['declaracion_situacionPatrimonial_datosGenerales_nombre', 
                                               'declaracion_situacionPatrimonial_datosGenerales_primerApellido']):
            df['NombreCompleto'] = (
                df['declaracion_situacionPatrimonial_datosGenerales_nombre'].fillna('').astype(str) + ' ' +
                df['declaracion_situacionPatrimonial_datosGenerales_primerApellido'].fillna('').astype(str) + ' ' +
                df.get('declaracion_situacionPatrimonial_datosGenerales_segundoApellido', pd.Series([''] * len(df))).fillna('').astype(str)
            )
            df['NombreCompleto'] = df['NombreCompleto'].apply(normalizar_texto)
        
        # Normalizar estado
        estado_cols = ['declaracion_situacionPatrimonial_datosEmpleoCargoComision_entidadFederativa', 'entidadFederativa', 'estado', 'Estado']
        for col in estado_cols:
            if col in df.columns:
                df['EstadoNormalizado'] = df[col].apply(normalizar_estado)
                break
        
        # Extraer año de declaración
        if 'metadata_actualizacion' in df.columns:
            df['AnioDeclaracion'] = pd.to_datetime(df['metadata_actualizacion'], errors='coerce').dt.year
        
        progress_bar.progress(100)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        status_text.text(f"✅ Archivo cargado exitosamente en {elapsed_time:.2f} segundos")
        return df
    except Exception as e:
        st.error(f"❌ Error al procesar archivo: {str(e)}")
        return None

def consolidar_datasets():
    """Consolida todos los datasets cargados en uno solo"""
    if not st.session_state.datasets:
        return None
    
    try:
        df_list = []
        for nombre, df in st.session_state.datasets.items():
            df_copy = df.copy()
            df_copy['NombreArchivo'] = nombre
            df_list.append(df_copy)
        
        df_consolidado = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
        
        # Eliminar duplicados exactos
        len_antes = len(df_consolidado)
        df_consolidado = df_consolidado.drop_duplicates()
        len_despues = len(df_consolidado)
        
        if len_antes != len_despues:
            st.info(f"🔄 Se eliminaron {len_antes - len_despues} registros duplicados durante la consolidación")
        
        st.session_state.dataset_consolidado = df_consolidado
        st.success(f"✅ {len(st.session_state.datasets)} archivos consolidados exitosamente en {len(df_consolidado):,} registros")
        return df_consolidado
    except Exception as e:
        st.error(f"❌ Error consolidando datasets: {str(e)}")
        return None
def preparar_para_excel(df):
    """Convierte columnas datetime con timezone a timezone-unaware para Excel"""
    df = df.copy()
    for col in df.select_dtypes(include=['datetimetz', 'datetime']).columns:
        try:
            df[col] = df[col].dt.tz_localize(None)
        except:
            pass
    return df
def calcular_estadisticas_sin_ceros(serie):
    """Calcula estadísticas excluyendo valores cero"""
    valores = pd.to_numeric(serie, errors='coerce').dropna()
    valores_sin_cero = valores[valores != 0]
    
    if len(valores_sin_cero) == 0:
        return {
            'promedio': 0,
            'maximo': 0,
            'minimo': 0,
            'count_sin_cero': 0
        }
    
    return {
        'promedio': valores_sin_cero.mean(),
        'maximo': valores_sin_cero.max(),
        'minimo': valores_sin_cero.min(),
        'count_sin_cero': len(valores_sin_cero)
    }
def detectar_duplicados_sin_contar(serie):
    """Detecta valores únicos sin contar duplicados iguales"""
    valores = pd.to_numeric(serie, errors='coerce').dropna()
    valores_sin_cero = valores[valores != 0]
    valores_unicos = valores_sin_cero.drop_duplicates()
    return {
        'count_unicos': len(valores_unicos),
        'importe_total': valores_unicos.sum()
    }
def calcular_alertas_individuales(row):
    """Calcula alertas específicas para un registro individual"""
    alertas = []
    
    # Alerta por incremento inusual
    if row.get('delta_max_pct', 0) > UMBRAL_ALERTA_INCREMENTO:
        alertas.append(f"INCREMENTO_INUSUAL_{row.get('delta_max_pct', 0):.1f}%")
    
    # Alerta por ratio bienes/ingresos
    if row.get('ingresos_totales_promedio', 0) > 0:
        ratio_bienes = row.get('Importe_bienesInm', 0) / row.get('ingresos_totales_promedio', 1)
        if ratio_bienes > UMBRAL_RATIO_BIENES_INGRESOS:
            alertas.append(f"BIENES_ALTOS_{ratio_bienes:.1f}x")
    
    # Alerta por adeudos elevados
    if row.get('ingresos_totales_promedio', 0) > 0:
        ratio_adeudos = row.get('Importe_adeudos_total', 0) / row.get('ingresos_totales_promedio', 1)
        if ratio_adeudos > UMBRAL_RATIO_ADEUDOS_INGRESOS:
            alertas.append(f"ADEUDOS_ELEVADOS_{ratio_adeudos:.1f}x")
    
    # Alerta por otros ingresos desproporcionados
    if row.get('ingreso_sp_promedio', 0) > 0:
        ratio_otros = row.get('otros_ingresos_promedio', 0) / row.get('ingreso_sp_promedio', 1)
        if ratio_otros > 2.0:
            alertas.append(f"OTROS_INGRESOS_ALTOS_{ratio_otros:.1f}x")
    
    # Alerta por inversiones no justificadas
    if row.get('ingresos_totales_promedio', 0) > 0:
        ratio_inv = row.get('Importe_inversiones', 0) / row.get('ingresos_totales_promedio', 1)
        if ratio_inv > 1.5:
            alertas.append(f"INVERSIONES_ALTAS_{ratio_inv:.1f}x")
    
    # Alerta por vehículos costosos
    if row.get('No_autos', 0) > 0:
        promedio_auto = row.get('Importe_autos', 0) / row.get('No_autos', 1)
        if promedio_auto > row.get('ingreso_sp_promedio', 0) * 2:
            alertas.append(f"VEHICULOS_COSTOSOS")
    
    return '; '.join(alertas) if alertas else 'SIN_ALERTA'
def calcular_nivel_riesgo_corrupcion(row):
    """Calcula el nivel de riesgo de corrupción basado en múltiples indicadores"""
    puntos_riesgo = 0
    
    # Puntos por incremento patrimonial inusual
    if row.get('delta_max_pct', 0) > UMBRAL_ALERTA_INCREMENTO:
        puntos_riesgo += 3
    
    # Puntos por bienes inmuebles desproporcionados
    if row.get('ingresos_totales_promedio', 0) > 0:
        ratio_bienes = row.get('Importe_bienesInm', 0) / row.get('ingresos_totales_promedio', 1)
        if ratio_bienes > UMBRAL_RATIO_BIENES_INGRESOS:
            puntos_riesgo += 2
        if ratio_bienes > UMBRAL_RATIO_BIENES_INGRESOS * 2:
            puntos_riesgo += 2
    
    # Puntos por adeudos elevados
    if row.get('ingresos_totales_promedio', 0) > 0:
        ratio_adeudos = row.get('Importe_adeudos_total', 0) / row.get('ingresos_totales_promedio', 1)
        if ratio_adeudos > UMBRAL_RATIO_ADEUDOS_INGRESOS:
            puntos_riesgo += 2
    
    # Puntos por otros ingresos desproporcionados
    if row.get('ingreso_sp_promedio', 0) > 0:
        ratio_otros = row.get('otros_ingresos_promedio', 0) / row.get('ingreso_sp_promedio', 1)
        if ratio_otros > 2.0:
            puntos_riesgo += 2
        if ratio_otros > 5.0:
            puntos_riesgo += 2
    
    # Puntos por inversiones no justificadas
    if row.get('ingresos_totales_promedio', 0) > 0:
        ratio_inv = row.get('Importe_inversiones', 0) / row.get('ingresos_totales_promedio', 1)
        if ratio_inv > 1.5:
            puntos_riesgo += 1
    
    # Puntos por declaraciones con valores cero (posible ocultamiento)
    if row.get('tiene_valores_cero', False):
        puntos_riesgo += 1
    
    # Clasificar nivel de riesgo
    if puntos_riesgo >= 7:
        return 'ALTO'
    elif puntos_riesgo >= 4:
        return 'MEDIO'
    else:
        return 'BAJO'

def calcular_metricas_por_persona(df):
    """
    Calcula métricas patrimoniales exhaustivas por persona
    Incluye:
    - Conteos de declaraciones sin valores cero
    - Promedios, máximos y mínimos excluyendo ceros
    - Alertas de corrupción
    - Análisis de tabuladores salariales
    - Bienes, inversiones y adeudos sin duplicados
    """
    col_nombre = 'NombreCompleto'
    if col_nombre not in df.columns:
        st.error(f"❌ Columna '{col_nombre}' no encontrada")
        return pd.DataFrame()
    
    df_trabajo = df.copy()
    
    # Definir columnas clave
    col_ingreso_sp_anual = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
    col_ingreso_sp_mensual = 'declaracion_situacionPatrimonial_ingresos_remuneracionMensualCargoPublico_valor'
    col_otros_ing_anual = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
    col_otros_ing_mensual = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosMensualesTotal_valor'
    col_total_ing_anual = 'declaracion_situacionPatrimonial_ingresos_totalIngresosAnualesNetos_valor'
    col_total_ing_mensual = 'declaracion_situacionPatrimonial_ingresos_totalIngresosMensualesNetos_valor'
    col_fecha = 'metadata_actualizacion'
    col_puesto = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision'
    col_dependencia = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico'
    col_estado = 'EstadoNormalizado'
    
    # Asegurar columnas numéricas
    for col in [col_ingreso_sp_anual, col_ingreso_sp_mensual, col_otros_ing_anual, 
                col_otros_ing_mensual, col_total_ing_anual, col_total_ing_mensual]:
        if col in df_trabajo.columns:
            df_trabajo[col] = pd.to_numeric(df_trabajo[col], errors='coerce').fillna(0.0)
    
    # Calcular ingresos anualizados
    df_trabajo['ingreso_sp_anualizado'] = df_trabajo.get(col_ingreso_sp_anual, 0)
    mask_sp_zero = df_trabajo['ingreso_sp_anualizado'] == 0
    df_trabajo.loc[mask_sp_zero, 'ingreso_sp_anualizado'] = df_trabajo.loc[mask_sp_zero, col_ingreso_sp_mensual] * 12
    
    df_trabajo['otros_ingresos_anualizado'] = df_trabajo.get(col_otros_ing_anual, 0)
    mask_oi_zero = df_trabajo['otros_ingresos_anualizado'] == 0
    df_trabajo.loc[mask_oi_zero, 'otros_ingresos_anualizado'] = df_trabajo.loc[mask_oi_zero, col_otros_ing_mensual] * 12
    
    df_trabajo['total_ingresos_anualizado'] = df_trabajo.get(col_total_ing_anual, 0)
    mask_ti_zero = df_trabajo['total_ingresos_anualizado'] == 0
    df_trabajo.loc[mask_ti_zero, 'total_ingresos_anualizado'] = df_trabajo.loc[mask_ti_zero, col_total_ing_mensual] * 12
    
    # Si total sigue en cero, calcular como suma
    mask_ti_still_zero = df_trabajo['total_ingresos_anualizado'] == 0
    df_trabajo.loc[mask_ti_still_zero, 'total_ingresos_anualizado'] = (
        df_trabajo.loc[mask_ti_still_zero, 'ingreso_sp_anualizado'] + 
        df_trabajo.loc[mask_ti_still_zero, 'otros_ingresos_anualizado']
    )
    
    # Parsear fecha
    if col_fecha in df_trabajo.columns:
        df_trabajo[col_fecha] = pd.to_datetime(df_trabajo[col_fecha], utc=True, errors='coerce')
    else:
        df_trabajo[col_fecha] = pd.NaT
    
    # Ordenar por persona y fecha
    df_trabajo = df_trabajo.sort_values([col_nombre, col_fecha])
    
    # Calcular incrementos
    df_trabajo['ingreso_prev'] = df_trabajo.groupby(col_nombre)['total_ingresos_anualizado'].shift(1)
    df_trabajo['delta_abs'] = df_trabajo['total_ingresos_anualizado'] - df_trabajo['ingreso_prev']
    df_trabajo['delta_pct'] = ((df_trabajo['total_ingresos_anualizado'] - df_trabajo['ingreso_prev']) / 
                                (df_trabajo['ingreso_prev'] + 1)) * 100
    df_trabajo['alerta_incremento'] = df_trabajo['delta_pct'] > UMBRAL_ALERTA_INCREMENTO
    
    # Procesar bienes inmuebles (sin duplicados)
    bienes_cols = [col for col in df_trabajo.columns if 'bienesInmuebles' in col and 'valorAdquisicion' in col]
    for col in bienes_cols:
        df_trabajo[col] = pd.to_numeric(df_trabajo[col], errors='coerce').fillna(0)
    
    # Procesar vehículos
    vehiculos_cols = [col for col in df_trabajo.columns if 'vehiculos' in col.lower() and 'valorAdquisicion' in col]
    for col in vehiculos_cols:
        df_trabajo[col] = pd.to_numeric(df_trabajo[col], errors='coerce').fillna(0)
    
    # Procesar inversiones
    inversiones_cols = [col for col in df_trabajo.columns if 'inversiones' in col.lower() and 'valor' in col]
    for col in inversiones_cols:
        df_trabajo[col] = pd.to_numeric(df_trabajo[col], errors='coerce').fillna(0)
    
    # Procesar adeudos
    adeudos_cols = [col for col in df_trabajo.columns if 'adeudos' in col.lower() and ('monto' in col or 'saldo' in col)]
    for col in adeudos_cols:
        df_trabajo[col] = pd.to_numeric(df_trabajo[col], errors='coerce').fillna(0)
    
    # Función de agregación personalizada
    def agg_metricas(grupo):
        resultado = {}
        
        # Información básica
        resultado['n_declaraciones'] = len(grupo)
        
        # Conteos sin cero
        resultado['Decla_sin0SP'] = (grupo['ingreso_sp_anualizado'] != 0).sum()
        resultado['Decla_sin0OI'] = (grupo['otros_ingresos_anualizado'] != 0).sum()
        resultado['Decla_sin0TI'] = (grupo['total_ingresos_anualizado'] != 0).sum()
        
        # Calcular promedios excluyendo ceros
        stats_sp = calcular_estadisticas_sin_ceros(grupo['ingreso_sp_anualizado'])
        stats_oi = calcular_estadisticas_sin_ceros(grupo['otros_ingresos_anualizado'])
        stats_ti = calcular_estadisticas_sin_ceros(grupo['total_ingresos_anualizado'])
        
        resultado['ingreso_sp_promedio'] = stats_sp['promedio']
        resultado['max_ingresos_SP'] = stats_sp['maximo']
        resultado['min_ingresos_SP'] = stats_sp['minimo']
        
        resultado['otros_ingresos_promedio'] = stats_oi['promedio']
        resultado['max_ingresos_OI'] = stats_oi['maximo']
        resultado['min_ingresos_OI'] = stats_oi['minimo']
        
        resultado['ingresos_totales_promedio'] = stats_ti['promedio']
        resultado['max_ingresos_totales'] = stats_ti['maximo']
        resultado['min_ingresos_totales'] = stats_ti['minimo']
        
        # Verificar si tiene valores cero
        resultado['tiene_valores_cero'] = (
            (grupo['ingreso_sp_anualizado'] == 0).any() or
            (grupo['otros_ingresos_anualizado'] == 0).any() or
            (grupo['total_ingresos_anualizado'] == 0).any()
        )
        
        # Bienes inmuebles (sin duplicados)
        if bienes_cols:
            bienes_unicos = detectar_duplicados_sin_contar(pd.concat([grupo[col] for col in bienes_cols]))
            resultado['No_bienesInm'] = bienes_unicos['count_unicos']
            resultado['Importe_bienesInm'] = bienes_unicos['importe_total']
        else:
            resultado['No_bienesInm'] = 0
            resultado['Importe_bienesInm'] = 0
        
        # Vehículos (sin duplicados)
        if vehiculos_cols:
            vehiculos_unicos = detectar_duplicados_sin_contar(pd.concat([grupo[col] for col in vehiculos_cols]))
            resultado['No_autos'] = vehiculos_unicos['count_unicos']
            resultado['Importe_autos'] = vehiculos_unicos['importe_total']
        else:
            resultado['No_autos'] = 0
            resultado['Importe_autos'] = 0
        
        # Inversiones (sin duplicados)
        if inversiones_cols:
            inv_unicos = detectar_duplicados_sin_contar(pd.concat([grupo[col] for col in inversiones_cols]))
            resultado['No_inversiones'] = inv_unicos['count_unicos']
            resultado['Importe_inversiones'] = inv_unicos['importe_total']
        else:
            resultado['No_inversiones'] = 0
            resultado['Importe_inversiones'] = 0
        
        # Adeudos (sin duplicados)
        if adeudos_cols:
            adeudos_unicos = detectar_duplicados_sin_contar(pd.concat([grupo[col] for col in adeudos_cols]))
            resultado['No_deudas'] = adeudos_unicos['count_unicos']
            resultado['Importe_adeudos_total'] = adeudos_unicos['importe_total']
        else:
            resultado['No_deudas'] = 0
            resultado['Importe_adeudos_total'] = 0
        
        # Incrementos
        resultado['delta_max_abs'] = grupo['delta_abs'].max() if not grupo['delta_abs'].isna().all() else 0
        resultado['delta_max_pct'] = grupo['delta_pct'].max() if not grupo['delta_pct'].isna().all() else 0
        resultado['alerta_incremento_inusual'] = grupo['alerta_incremento'].any()
        
        # Información adicional
        resultado['fecha_ultima_actualizacion'] = grupo[col_fecha].max()
        
        # Estado, dependencia y cargo (más frecuente)
        if col_estado in grupo.columns:
            resultado['Estado'] = grupo[col_estado].mode()[0] if not grupo[col_estado].mode().empty else None
        if col_dependencia in grupo.columns:
            resultado['Dependencia'] = grupo[col_dependencia].mode()[0] if not grupo[col_dependencia].mode().empty else None
        if col_puesto in grupo.columns:
            # Tomar el puesto asociado al ingreso más alto
            idx_max_ing = grupo['total_ingresos_anualizado'].idxmax()
            resultado['Cargo'] = grupo.loc[idx_max_ing, col_puesto] if pd.notna(idx_max_ing) else None
        
        # Año de declaración
        if 'AnioDeclaracion' in grupo.columns:
            resultado['AnioDeclaracion'] = grupo['AnioDeclaracion'].mode()[0] if not grupo['AnioDeclaracion'].mode().empty else 2024
        else:
            resultado['AnioDeclaracion'] = 2024
        
        return pd.Series(resultado)
    
    # Aplicar agregación
    metricas_persona = df_trabajo.groupby(col_nombre).apply(agg_metricas).reset_index()
    
    
    # Calcular porcentaje de incremento anual promedio
    metricas_persona['pct_incremento_anual'] = metricas_persona['delta_max_pct']
    
    # Calcular alertas individuales
    metricas_persona['alertas_detalle'] = metricas_persona.apply(calcular_alertas_individuales, axis=1)
    
    # Calcular nivel de riesgo de corrupción
    metricas_persona['nivel_riesgo_corrupcion'] = metricas_persona.apply(calcular_nivel_riesgo_corrupcion, axis=1)
    
    # Alertas específicas por categoría
    metricas_persona['alerta_ingresos'] = metricas_persona['delta_max_pct'] > UMBRAL_ALERTA_INCREMENTO
    metricas_persona['alerta_otros_ingresos'] = (
        metricas_persona['otros_ingresos_promedio'] / (metricas_persona['ingreso_sp_promedio'] + 1)
    ) > 2.0
    metricas_persona['alerta_adeudos'] = (
        metricas_persona['Importe_adeudos_total'] / (metricas_persona['ingresos_totales_promedio'] + 1)
    ) > UMBRAL_RATIO_ADEUDOS_INGRESOS
    metricas_persona['alerta_bienes'] = (
        metricas_persona['Importe_bienesInm'] / (metricas_persona['ingresos_totales_promedio'] + 1)
    ) > UMBRAL_RATIO_BIENES_INGRESOS
    
    # Alerta global de riesgo patrimonial
    metricas_persona['alerta_global_riesgo'] = (
        metricas_persona['alerta_ingresos'] |
        metricas_persona['alerta_otros_ingresos'] |
        metricas_persona['alerta_adeudos'] |
        metricas_persona['alerta_bienes']
    )
    
    # Alerta específica por ingreso anual anómalo
    metricas_persona['alerta_ingreso_anomalo'] = (
        (metricas_persona['max_ingresos_totales'] / (metricas_persona['min_ingresos_totales'] + 1)) > 5.0
    )
    
    return metricas_persona
def seccion_inicio():
    if not st.session_state.datasets:
        st.info("ℹ️ Por favor, cargue archivos desde el panel lateral para comenzar.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown(f"### 📁 Resumen: {st.session_state.dataset_seleccionado}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Filas", f"{len(df):,}")
    with col2:
        st.metric("📋 Columnas", f"{len(df.columns):,}")
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("💾 Memoria", f"{memory_usage:.2f} MB")
    with col4:
        hash_md5 = calcular_hash_md5(df)
        st.metric("🔒 Hash MD5", hash_md5[:8])
    st.markdown("---")
    st.markdown("#### 📊 Tipos de Datos")
    tipos_datos = df.dtypes.value_counts()
    col1, col2 = st.columns(2)
    with col1:
        for tipo, cantidad in tipos_datos.items():
            st.write(f"**{tipo}**: {cantidad} columnas")
    with col2:
        fig = px.pie(
            values=tipos_datos.values,
            names=[str(x) for x in tipos_datos.index],
            title="Distribución de Tipos de Datos",
            color_discrete_sequence=px.colors.sequential.Purp
        )
        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_446")
    st.markdown("---")
    st.markdown("#### 🔍 Vista Previa de Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primeras 5 filas:**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.markdown("**Últimas 5 filas:**")
        st.dataframe(df.tail(), use_container_width=True)
    st.markdown("---")
    st.markdown("#### 📋 Información Detallada")
    info_df = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': df.dtypes.values,
        'No Nulos': df.count().values,
        '% Nulos': ((df.isnull().sum() / len(df)) * 100).values,
        'Valores Únicos': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)
    st.markdown("---")
    st.markdown("### 💾 Descargar Dataset con Cálculos")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generar Dataset con Métricas Calculadas"):
            with st.spinner("Calculando métricas patrimoniales..."):
                try:
                    df_exportar = df.copy()
                    # Aplicar limpieza de columnas monetarias
                    col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
                    col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
                    if col_ingreso_sp in df_exportar.columns:
                        df_exportar[col_ingreso_sp] = pd.to_numeric(df_exportar[col_ingreso_sp], errors='coerce').fillna(0)
                    if col_otros_ingresos in df_exportar.columns:
                        df_exportar[col_otros_ingresos] = pd.to_numeric(df_exportar[col_otros_ingresos], errors='coerce').fillna(0)
                    # Calcular Total de Ingresos correctamente
                    df_exportar['CALC_Total_Ingresos'] = (
                        df_exportar.get(col_ingreso_sp, 0) + 
                        df_exportar.get(col_otros_ingresos, 0)
                    )
                    st.session_state.df_procesado = df_exportar
                    st.success("✅ Dataset con métricas calculadas generado exitosamente")
                    st.markdown("#### 📊 Nuevas Columnas Agregadas:")
                    nuevas_cols = [col for col in df_exportar.columns if col.startswith('CALC_')]
                    for col in nuevas_cols:
                        st.write(f"✓ **{col}**")
                except Exception as e:
                    st.error(f"Error generando dataset: {e}")
    with col2:
        if st.session_state.df_procesado is not None:
            try:
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_procesado)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Datos con Calculos', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Dataset Procesado (Excel)",
                    data=output,
                    file_name=f"dataset_procesado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exportando: {e}")
        else:
            st.info("Primero genere el dataset con métricas usando el botón de la izquierda")

def seccion_eda():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🔍 EDA - Análisis Exploratorio de Datos")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Información General",
        "🔢 Valores Únicos",
        "❌ Análisis de Nulos",
        "📈 Distribuciones",
        "🔗 Correlaciones",
        "📊 Cargos e Ingresos"
    ])
    with tab1:
        st.markdown("### 📊 Información General del Dataset")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Total Registros", f"{len(df):,}")
        with col2:
            st.metric("📋 Total Columnas", f"{len(df.columns):,}")
        with col3:
            duplicados = df.duplicated().sum()
            st.metric("🔄 Duplicados", f"{duplicados:,}")
        with col4:
            memoria = df.memory_usage(deep=True).sum() / (1024**2)
            st.metric("💾 Memoria Total", f"{memoria:.2f} MB")
        with col5:
            nulos_totales = df.isnull().sum().sum()
            st.metric("❌ Nulos Totales", f"{nulos_totales:,}")
        st.markdown("---")
        st.markdown("### 📋 Resumen Estadístico")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Columnas Numéricas")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(f"**Total:** {len(numeric_cols)} columnas")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        with col2:
            st.markdown("#### Columnas Categóricas")
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            st.write(f"**Total:** {len(cat_cols)} columnas")
            if cat_cols:
                cat_summary = pd.DataFrame({
                    'Columna': cat_cols[:10],
                    'Valores Únicos': [df[col].nunique() for col in cat_cols[:10]],
                    'Valor Más Frecuente': [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in cat_cols[:10]]
                })
                st.dataframe(cat_summary, use_container_width=True)
        st.markdown("---")
        st.markdown("### 🎯 Información Detallada por Columna")
        try:
            memoria_por_col = df.memory_usage(deep=True)
            info_detallada = pd.DataFrame({
                'Columna': df.columns,
                'Tipo de Dato': df.dtypes.astype(str).values,
                'Valores No Nulos': df.count().values,
                'Valores Nulos': df.isnull().sum().values,
                '% Nulos': ((df.isnull().sum() / len(df)) * 100).round(2).values,
                'Valores Únicos': [df[col].nunique() for col in df.columns],
                'Memoria (KB)': (memoria_por_col[1:] / 1024).round(2).values
            })
            st.dataframe(info_detallada, use_container_width=True)
        except Exception as e:
            info_simple = pd.DataFrame({
                'Columna': df.columns,
                'Tipo de Dato': df.dtypes.astype(str).values,
                'Valores No Nulos': df.count().values,
                'Valores Únicos': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(info_simple, use_container_width=True)
    with tab2:
        st.markdown("### 🔢 Análisis de Valores Únicos")
        columna_seleccionada = st.selectbox(
            "Seleccione una columna para analizar:",
            df.columns.tolist(),
            key="valores_unicos_col"
        )
        valores_unicos = df[columna_seleccionada].value_counts()
        total_unicos = df[columna_seleccionada].nunique()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Valores Únicos", f"{total_unicos:,}")
        with col2:
            st.metric("📊 Total Registros", f"{len(df):,}")
        with col3:
            porcentaje_unicos = (total_unicos / len(df)) * 100
            st.metric("📈 % Unicidad", f"{porcentaje_unicos:.2f}%")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Tabla de Frecuencias (Top 50)")
            frecuencias_df = pd.DataFrame({
                'Valor': valores_unicos.index,
                'Frecuencia': valores_unicos.values,
                '% del Total': ((valores_unicos.values / len(df)) * 100).round(2)
            })
            st.dataframe(frecuencias_df.head(50), use_container_width=True)
        with col2:
            st.markdown("#### 📈 Gráfico de Distribución")
            if total_unicos <= 50:
                fig = px.bar(
                    x=valores_unicos.index[:20],
                    y=valores_unicos.values[:20],
                    title=f"Top 20 valores más frecuentes",
                    labels={'x': columna_seleccionada, 'y': 'Frecuencia'},
                    color=valores_unicos.values[:20],
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_623")
            else:
                fig = px.histogram(
                    df,
                    x=columna_seleccionada,
                    title=f"Distribución de {columna_seleccionada}",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_631")
        st.markdown("---")
        if df[columna_seleccionada].dtype == 'object' and WORDCLOUD_AVAILABLE:
            st.markdown("#### ☁️ Nube de Palabras")
            try:
                texto = ' '.join(df[columna_seleccionada].dropna().astype(str).tolist())
                if len(texto) > 0:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='twilight',
                        max_words=100
                    ).generate(texto)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Nube de Palabras: {columna_seleccionada}', fontsize=16, pad=20)
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"No se pudo generar la nube de palabras: {e}")
    with tab3:
        st.markdown("### ❌ Análisis de Valores Nulos")
        nulos_por_columna = df.isnull().sum()
        porcentaje_nulos = (nulos_por_columna / len(df)) * 100
        nulos_df = pd.DataFrame({
            'Columna': df.columns,
            'Valores Nulos': nulos_por_columna.values,
            '% Nulos': porcentaje_nulos.values
        }).sort_values('Valores Nulos', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Estadísticas de Nulos")
            columnas_con_nulos = (nulos_df['Valores Nulos'] > 0).sum()
            st.metric("📋 Columnas con Nulos", columnas_con_nulos)
            st.metric("❌ Total de Nulos", f"{nulos_df['Valores Nulos'].sum():,}")
            st.metric("📈 % Promedio de Nulos", f"{nulos_df['% Nulos'].mean():.2f}%")
        with col2:
            st.markdown("#### 🎯 Top 10 Columnas con Más Nulos")
            fig = px.bar(
                nulos_df.head(10),
                x='% Nulos',
                y='Columna',
                orientation='h',
                title="Porcentaje de Nulos por Columna",
                color='% Nulos',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_679")
        st.markdown("---")
        st.markdown("#### 📋 Tabla Completa de Nulos")
        st.dataframe(nulos_df, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 🔥 Heatmap de Nulos")
        if len(df.columns) <= 50:
            nulos_matrix = df.isnull().astype(int)
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(
                nulos_matrix,
                cmap='RdPu',
                cbar_kws={'label': 'Nulo (1) / No Nulo (0)'},
                ax=ax,
                yticklabels=False
            )
            ax.set_title('Mapa de Calor de Valores Nulos', fontsize=16, pad=20)
            st.pyplot(fig)
        else:
            st.info("Demasiadas columnas para mostrar heatmap completo. Mostrando muestra aleatoria de 50 columnas.")
            sample_cols = np.random.choice(df.columns, min(50, len(df.columns)), replace=False)
            nulos_matrix = df[sample_cols].isnull().astype(int)
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(
                nulos_matrix,
                cmap='RdPu',
                cbar_kws={'label': 'Nulo (1) / No Nulo (0)'},
                ax=ax,
                yticklabels=False
            )
            ax.set_title('Mapa de Calor de Valores Nulos (Muestra)', fontsize=16, pad=20)
            st.pyplot(fig)
    with tab4:
        st.markdown("### 📈 Análisis de Distribuciones")
        all_cols = df.columns.tolist()
        columna_distribucion = st.selectbox(
            "Seleccione una columna:",
            all_cols,
            key="dist_col"
        )
        if df[columna_distribucion].dtype in [np.number, 'int64', 'float64']:
            datos_validos = pd.to_numeric(df[columna_distribucion], errors='coerce').dropna()
            if len(datos_validos) == 0:
                st.warning("⚠️ No hay datos numéricos válidos en esta columna.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Media", f"{datos_validos.mean():.2f}")
                with col2:
                    st.metric("📍 Mediana", f"{datos_validos.median():.2f}")
                with col3:
                    st.metric("📉 Desv. Est.", f"{datos_validos.std():.2f}")
                with col4:
                    moda_val = datos_validos.mode()
                    st.metric("🎯 Moda", f"{moda_val[0] if len(moda_val) > 0 else 'N/A'}")
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📊 Histograma")
                    fig = px.histogram(
                        datos_validos,
                        nbins=50,
                        title=f"Distribución de {columna_distribucion}",
                        color_discrete_sequence=['#667eea'],
                        marginal='box'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_chart_745")
                with col2:
                    st.markdown("#### 📦 Box Plot")
                    fig = px.box(
                        y=datos_validos,
                        title=f"Box Plot de {columna_distribucion}",
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_chart_753")
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📈 Gráfico Q-Q")
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        stats.probplot(datos_validos, dist="norm", plot=ax)
                        ax.set_title(f"Q-Q Plot: {columna_distribucion}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"No se pudo generar Q-Q plot: {e}")
                with col2:
                    st.markdown("#### 📊 Estadísticas Detalladas")
                    try:
                        stats_df = pd.DataFrame({
                            'Métrica': ['Mínimo', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'Máximo', 'Rango', 'IQR', 'Asimetría', 'Curtosis'],
                            'Valor': [
                                datos_validos.min(),
                                datos_validos.quantile(0.25),
                                datos_validos.quantile(0.50),
                                datos_validos.quantile(0.75),
                                datos_validos.max(),
                                datos_validos.max() - datos_validos.min(),
                                datos_validos.quantile(0.75) - datos_validos.quantile(0.25),
                                datos_validos.skew(),
                                datos_validos.kurtosis()
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error calculando estadísticas: {e}")
        else:
            st.info("La columna seleccionada no es numérica. Mostrando distribución categórica.")
            valores_unicos = df[columna_distribucion].value_counts().head(20)
            fig = px.bar(
                x=valores_unicos.values,
                y=valores_unicos.index,
                orientation='h',
                title=f"Top 20 valores de {columna_distribucion}",
                labels={'x': 'Frecuencia', 'y': 'Valor'},
                color=valores_unicos.values,
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_798")
    with tab5:
        st.markdown("### 🔗 Análisis de Correlaciones")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")
            return
        # Limitar a top 20 por varianza si hay muchas
        if len(numeric_cols) > 20:
            varianzas = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = varianzas.head(20).index.tolist()
        metodo_correlacion = st.selectbox(
            "Seleccione el método de correlación:",
            ["Pearson", "Spearman", "Kendall"],
            key="metodo_corr"
        )
        metodo_map = {
            "Pearson": "pearson",
            "Spearman": "spearman",
            "Kendall": "kendall"
        }
        try:
            corr_matrix = df[numeric_cols].corr(method=metodo_map[metodo_correlacion])
            st.markdown("#### 🔥 Matriz de Correlación")
            fig, ax = plt.subplots(figsize=(16, 12))
            sns.heatmap(
                corr_matrix,
                annot=True if len(numeric_cols) <= 15 else False,
                fmt='.2f',
                cmap='twilight',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title(f'Matriz de Correlación ({metodo_correlacion})', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            st.markdown("---")
            st.markdown("#### 🎯 Correlaciones Más Fuertes")
            correlaciones_pares = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlaciones_pares.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlación': corr_matrix.iloc[i, j]
                    })
            corr_df = pd.DataFrame(correlaciones_pares)
            corr_df['Correlación Abs'] = corr_df['Correlación'].abs()
            corr_df = corr_df.sort_values('Correlación Abs', ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🔴 Top 10 Correlaciones Positivas")
                top_positivas = corr_df[corr_df['Correlación'] > 0].head(10)
                st.dataframe(top_positivas[['Variable 1', 'Variable 2', 'Correlación']], use_container_width=True)
            with col2:
                st.markdown("##### 🔵 Top 10 Correlaciones Negativas")
                top_negativas = corr_df[corr_df['Correlación'] < 0].head(10)
                st.dataframe(top_negativas[['Variable 1', 'Variable 2', 'Correlación']], use_container_width=True)
            st.markdown("---")
            st.markdown("#### 📊 Gráfico de Dispersión Interactivo")
            col1, col2 = st.columns(2)
            with col1:
                var_x = st.selectbox("Variable X:", numeric_cols, key="scatter_x")
            with col2:
                var_y = st.selectbox("Variable Y:", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
            fig = px.scatter(
                df,
                x=var_x,
                y=var_y,
                title=f"Dispersión: {var_x} vs {var_y}",
                trendline="ols",
                color_discrete_sequence=['#667eea'],
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_876")
        except Exception as e:
            st.error(f"Error calculando correlaciones: {e}")
    with tab6:
        st.markdown("### 📊 Análisis de Cargos e Ingresos")
        st.markdown("""
        <div class='info-box'>
            <p style='margin: 0; color: #667eea; font-weight: 600;'>
            📊 Análisis de relación entre cargos públicos e ingresos declarados
            </p>
        </div>
        """, unsafe_allow_html=True)
        cargo_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision'
        col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
        col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
        if cargo_col in df.columns and (col_ingreso_sp in df.columns or col_otros_ingresos in df.columns):
            df_temp = df.copy()
            # Calcular ingresos totales correctamente
            ingreso_sp_vals = pd.to_numeric(df_temp[col_ingreso_sp], errors='coerce').fillna(0) if col_ingreso_sp in df_temp.columns else 0
            otros_ingresos_vals = pd.to_numeric(df_temp[col_otros_ingresos], errors='coerce').fillna(0) if col_otros_ingresos in df_temp.columns else 0
            df_temp['Total_Ingresos'] = ingreso_sp_vals + otros_ingresos_vals
            st.markdown("#### 📊 Top 20 Cargos con Mayores Ingresos Promedio")
            cargos_ingresos = df_temp.groupby(cargo_col)['Total_Ingresos'].agg(['mean', 'count', 'sum']).reset_index()
            cargos_ingresos.columns = ['Cargo', 'Ingreso_Promedio', 'Cantidad', 'Total_Ingresos']
            cargos_ingresos = cargos_ingresos.sort_values('Ingreso_Promedio', ascending=False).head(20)
            fig = px.bar(
                cargos_ingresos,
                x='Ingreso_Promedio',
                y='Cargo',
                orientation='h',
                title='Top 20 Cargos con Mayor Ingreso Promedio',
                color='Ingreso_Promedio',
                color_continuous_scale='Purples',
                hover_data=['Cantidad', 'Total_Ingresos']
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_911")
            st.markdown("---")
            st.markdown("#### 📈 Dispersión: Cantidad de Servidores vs Ingreso Promedio por Cargo")
            fig = px.scatter(
                cargos_ingresos,
                x='Cantidad',
                y='Ingreso_Promedio',
                size='Total_Ingresos',
                hover_data=['Cargo'],
                title='Relación entre Cantidad de Servidores e Ingresos por Cargo',
                color='Ingreso_Promedio',
                color_continuous_scale='Viridis',
                labels={'Cantidad': 'Número de Servidores', 'Ingreso_Promedio': 'Ingreso Promedio (MXN)'}
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_925")
        else:
            st.warning("⚠️ No se encontraron las columnas necesarias para el análisis de cargos e ingresos.")
def seccion_visualizacion_global():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 📈 Visualización Global")
    tipo_grafico = st.selectbox(
        "Seleccione el tipo de gráfico:",
        ["Dispersión", "Líneas", "Barras", "Histograma", "Box Plot", "Pastel", "Violin", "Densidad", "Heatmap"],
        key="tipo_graf_global"
    )
    st.markdown("---")
    if tipo_grafico == "Dispersión":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="scatter_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="scatter_y_global")
        with col3:
            color_col = st.selectbox("Color (opcional):", [None] + df.columns.tolist(), key="scatter_color_global")
        size_col = st.selectbox("Tamaño (opcional):", [None] + df.columns.tolist(), key="scatter_size_global")
        try:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                title=f"Gráfico de Dispersión: {x_col} vs {y_col}",
                color_continuous_scale='Purples' if color_col and pd.api.types.is_numeric_dtype(df[color_col]) else None,
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_960")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Líneas":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="line_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="line_y_global")
        with col3:
            color_col = st.selectbox("Agrupar por (opcional):", [None] + df.columns.tolist(), key="line_color_global")
        try:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"Gráfico de Líneas: {y_col} por {x_col}",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_980")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Barras":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="bar_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="bar_y_global")
        with col3:
            orientacion = st.radio("Orientación:", ["Vertical", "Horizontal"], key="bar_orient_global")
        color_col = st.selectbox("Color por (opcional):", [None] + df.columns.tolist(), key="bar_color_global")
        try:
            fig = px.bar(
                df,
                x=x_col if orientacion == "Vertical" else y_col,
                y=y_col if orientacion == "Vertical" else x_col,
                color=color_col,
                orientation='v' if orientacion == "Vertical" else 'h',
                title=f"Gráfico de Barras: {y_col} por {x_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1002")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Histograma":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Variable:", df.columns.tolist(), key="hist_x_global")
        with col2:
            bins = st.slider("Número de bins:", 10, 100, 30, key="hist_bins_global")
        color_col = st.selectbox("Color por (opcional):", [None] + df.columns.tolist(), key="hist_color_global")
        marginal = st.selectbox("Marginal:", [None, "rug", "box", "violin"], key="hist_marginal_global")
        try:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=bins,
                color=color_col,
                marginal=marginal,
                title=f"Histograma: {x_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1023")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Variable numérica:", df.columns.tolist(), key="box_y_global")
        with col2:
            x_col = st.selectbox("Agrupar por (opcional):", [None] + df.columns.tolist(), key="box_x_global")
        color_col = st.selectbox("Color por (opcional):", [None] + df.columns.tolist(), key="box_color_global")
        try:
            fig = px.box(
                df,
                y=y_col,
                x=x_col,
                color=color_col,
                title=f"Box Plot: {y_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1042")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Pastel":
        col1, col2 = st.columns(2)
        with col1:
            values_col = st.selectbox("Valores:", df.columns.tolist(), key="pie_values_global")
        with col2:
            names_col = st.selectbox("Etiquetas:", df.columns.tolist(), key="pie_names_global")
        try:
            df_grouped = df.groupby(names_col)[values_col].sum().reset_index()
            fig = px.pie(
                df_grouped,
                values=values_col,
                names=names_col,
                title=f"Gráfico de Pastel: {values_col} por {names_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1060")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Violin":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Variable numérica:", df.columns.tolist(), key="violin_y_global")
        with col2:
            x_col = st.selectbox("Agrupar por (opcional):", [None] + df.columns.tolist(), key="violin_x_global")
        try:
            fig = px.violin(
                df,
                y=y_col,
                x=x_col,
                title=f"Violin Plot: {y_col}",
                color_discrete_sequence=['#667eea'],
                box=True,
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1079")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Densidad":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="density_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="density_y_global")
        try:
            fig = px.density_contour(
                df,
                x=x_col,
                y=y_col,
                title=f"Gráfico de Densidad: {x_col} vs {y_col}",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1096")
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 columnas numéricas para un heatmap.")
        else:
            cols_seleccionadas = st.multiselect(
                "Seleccione las columnas para el heatmap:",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))],
                key="heatmap_cols_global"
            )
            if cols_seleccionadas:
                try:
                    corr_matrix = df[cols_seleccionadas].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Mapa de Calor de Correlaciones",
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1120")
                except Exception as e:
                    st.error(f"Error generando heatmap: {e}")
def seccion_visualizacion_personalizada():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🎨 Visualización Personalizada")
    st.markdown("### ⚙️ Configuración de Gráficos")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_graficos = st.slider("Número de gráficos:", 1, 6, 2, key="num_graphs")
    with col2:
        layout_tipo = st.selectbox(
            "Tipo de layout:",
            ["Cuadrícula", "Vertical", "Horizontal"],
            key="layout_type"
        )
    with col3:
        tema = st.selectbox(
            "Tema del gráfico:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            key="theme_select"
        )
    st.markdown("---")
    if layout_tipo == "Cuadrícula":
        if num_graficos <= 2:
            rows, cols = 1, num_graficos
        elif num_graficos <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
    elif layout_tipo == "Vertical":
        rows, cols = num_graficos, 1
    else:
        rows, cols = 1, num_graficos
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Gráfico {i+1}" for i in range(num_graficos)],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    for i in range(num_graficos):
        with st.expander(f"⚙️ Configurar Gráfico {i+1}", expanded=i==0):
            col1, col2, col3 = st.columns(3)
            with col1:
                tipo_graf = st.selectbox(
                    "Tipo de gráfico:",
                    ["Scatter", "Bar", "Line", "Box", "Histogram", "Violin"],
                    key=f"tipo_{i}"
                )
            with col2:
                x_col = st.selectbox(
                    "Eje X:",
                    df.columns.tolist(),
                    key=f"x_{i}"
                )
            with col3:
                y_col = st.selectbox(
                    "Eje Y:",
                    df.columns.tolist(),
                    key=f"y_{i}",
                    index=min(1, len(df.columns)-1)
                )
            color_col = st.selectbox(
                "Color por (opcional):",
                [None] + df.columns.tolist(),
                key=f"color_{i}"
            )
            row_pos = (i // cols) + 1
            col_pos = (i % cols) + 1
            try:
                if tipo_graf == "Scatter":
                    scatter_data = df[[x_col, y_col]].dropna()
                    if color_col:
                        colors = df.loc[scatter_data.index, color_col]
                        fig.add_trace(
                            go.Scatter(
                                x=scatter_data[x_col],
                                y=scatter_data[y_col],
                                mode='markers',
                                name=f"{y_col} vs {x_col}",
                                marker=dict(
                                    color=colors if pd.api.types.is_numeric_dtype(colors) else None,
                                    size=8,
                                    opacity=0.6,
                                    colorscale='Purples'
                                )
                            ),
                            row=row_pos,
                            col=col_pos
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=scatter_data[x_col],
                                y=scatter_data[y_col],
                                mode='markers',
                                name=f"{y_col} vs {x_col}",
                                marker=dict(color='#667eea', size=8, opacity=0.6)
                            ),
                            row=row_pos,
                            col=col_pos
                        )
                elif tipo_graf == "Bar":
                    if df[x_col].dtype == 'object' or df[x_col].nunique() < 50:
                        df_grouped = df.groupby(x_col)[y_col].sum().reset_index()
                        fig.add_trace(
                            go.Bar(
                                x=df_grouped[x_col],
                                y=df_grouped[y_col],
                                name=f"{y_col} por {x_col}",
                                marker=dict(color='#764ba2')
                            ),
                            row=row_pos,
                            col=col_pos
                        )
                    else:
                        st.warning(f"Gráfico {i+1}: Demasiados valores únicos en X para gráfico de barras")
                elif tipo_graf == "Line":
                    line_data = df[[x_col, y_col]].dropna().sort_values(x_col)
                    fig.add_trace(
                        go.Scatter(
                            x=line_data[x_col],
                            y=line_data[y_col],
                            mode='lines+markers',
                            name=f"{y_col} vs {x_col}",
                            line=dict(color='#667eea', width=2)
                        ),
                        row=row_pos,
                        col=col_pos
                    )
                elif tipo_graf == "Box":
                    box_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
                    fig.add_trace(
                        go.Box(
                            y=box_data,
                            name=y_col,
                            marker=dict(color='#764ba2')
                        ),
                        row=row_pos,
                        col=col_pos
                    )
                elif tipo_graf == "Histogram":
                    hist_data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                    fig.add_trace(
                        go.Histogram(
                            x=hist_data,
                            name=x_col,
                            marker=dict(color='#667eea')
                        ),
                        row=row_pos,
                        col=col_pos
                    )
                elif tipo_graf == "Violin":
                    violin_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
                    fig.add_trace(
                        go.Violin(
                            y=violin_data,
                            name=y_col,
                            marker=dict(color='#764ba2'),
                            box_visible=True,
                            meanline_visible=True
                        ),
                        row=row_pos,
                        col=col_pos
                    )
            except Exception as e:
                st.warning(f"Error en gráfico {i+1}: {e}")
    fig.update_layout(
        height=400 * rows,
        showlegend=True,
        title_text="Dashboard Personalizado",
        title_font_size=24,
        template=tema
    )
    st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1298")
def seccion_machine_learning():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🤖 Machine Learning")
    tab1, tab2, tab3 = st.tabs([
        "🔧 Preparación de Datos",
        "🎯 Entrenamiento de Modelos",
        "📊 Resultados y Evaluación"
    ])
    with tab1:
        st.markdown("### 🔧 Preparación y Codificación de Datos")
        st.markdown("""
        <div class='info-box'>
            <h4 style='color: #667eea; margin-top: 0;'>📌 Codificación de Variables</h4>
            <p>Los algoritmos de Machine Learning requieren datos numéricos. Esta sección convierte 
            las variables categóricas (texto) en valores numéricos usando LabelEncoder.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Codificar Variables Categóricas", key="encode_button"):
            with st.spinner("Codificando variables..."):
                try:
                    df_encoded = df.copy()
                    encoding_dicts = {}
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    progress_bar = st.progress(0)
                    total_cols = len(categorical_cols)
                    columnas_codificadas = 0
                    columnas_omitidas = 0
                    for idx, col in enumerate(categorical_cols):
                        progress_bar.progress((idx + 1) / total_cols)
                        unique_values = df[col].nunique()
                        if unique_values < 100 and unique_values > 1:
                            le = LabelEncoder()
                            valid_data = df[col].dropna()
                            if len(valid_data) > 0:
                                le.fit(valid_data)
                                df_encoded[col] = df[col].map(
                                    lambda x: le.transform([x])[0] if pd.notna(x) else np.nan
                                )
                                encoding_dicts[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                                columnas_codificadas += 1
                        else:
                            columnas_omitidas += 1
                    st.session_state.df_encoded = df_encoded
                    st.session_state.encoding_dicts = encoding_dicts
                    progress_bar.progress(1.0)
                    st.success(f"✅ Codificación completada: {columnas_codificadas} columnas codificadas, {columnas_omitidas} omitidas (>100 valores únicos)")
                except Exception as e:
                    st.error(f"Error en la codificación: {e}")
        if st.session_state.df_encoded is not None:
            st.markdown("---")
            st.markdown("### 📊 Vista Previa del Dataset Codificado")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Columnas Codificadas", len(st.session_state.encoding_dicts))
            with col2:
                st.metric("📊 Total Columnas", len(st.session_state.df_encoded.columns))
            with col3:
                numeric_cols = st.session_state.df_encoded.select_dtypes(include=[np.number]).columns
                st.metric("🔢 Columnas Numéricas", len(numeric_cols))
            st.dataframe(st.session_state.df_encoded.head(10), use_container_width=True)
            st.markdown("---")
            st.markdown("### 📖 Diccionario de Codificación")
            if st.session_state.encoding_dicts:
                columna_dict = st.selectbox(
                    "Seleccione una columna para ver su diccionario:",
                    list(st.session_state.encoding_dicts.keys()),
                    key="dict_select_ml"
                )
                if columna_dict:
                    st.markdown(f"#### Codificación de: {columna_dict}")
                    dict_df = pd.DataFrame([
                        {"Valor Original": k, "Código": v}
                        for k, v in st.session_state.encoding_dicts[columna_dict].items()
                    ])
                    st.dataframe(dict_df, use_container_width=True)
            st.markdown("---")
            if st.button("💾 Exportar Dataset Codificado", key="export_encoded"):
                try:
                    output = BytesIO()
                    df_para_excel = preparar_para_excel(st.session_state.df_encoded)
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_para_excel.to_excel(writer, sheet_name='Datos Codificados', index=False)
                        dict_rows = []
                        for col, mappings in st.session_state.encoding_dicts.items():
                            for original, codigo in mappings.items():
                                dict_rows.append({
                                    'Columna': col,
                                    'Valor Original': original,
                                    'Código': codigo
                                })
                        if dict_rows:
                            dict_df = pd.DataFrame(dict_rows)
                            dict_df.to_excel(writer, sheet_name='Diccionario', index=False)
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Descargar Excel con Datos Codificados",
                        data=output,
                        file_name=f"dataset_codificado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error exportando: {e}")
    with tab2:
        st.markdown("### 🎯 Configuración y Entrenamiento de Modelos")
        datos_disponibles = st.radio(
            "Seleccione el dataset a utilizar:",
            ["Datos Originales", "Datos Codificados", "Servidores Públicos Únicos"],
            key="data_source_ml"
        )
        if datos_disponibles == "Datos Codificados":
            if st.session_state.df_encoded is None:
                st.warning("⚠️ Primero debe codificar las variables en la pestaña 'Preparación de Datos'")
                return
            df_ml = st.session_state.df_encoded
        elif datos_disponibles == "Servidores Públicos Únicos":
            if st.session_state.df_servidores_unicos is None:
                st.warning("⚠️ Primero debe generar la tabla de Servidores Públicos Únicos en la sección correspondiente")
                return
            df_ml = st.session_state.df_servidores_unicos
        else:
            df_ml = df
        st.markdown("---")
        tipo_modelo = st.selectbox(
            "Seleccione el tipo de modelo:",
            ["Clasificación", "Regresión", "Clustering"],
            key="model_type"
        )
        st.markdown("---")
        numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("❌ Se necesitan al menos 2 columnas numéricas para entrenar modelos.")
            return
        st.markdown("### 📊 Selección de Variables")
        features = st.multiselect(
            "Seleccione las variables independientes (Features):",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="features_select"
        )
        if tipo_modelo != "Clustering":
            target_options = [col for col in numeric_cols if col not in features]
            if not target_options:
                st.error("❌ Debe seleccionar al menos una variable como feature para tener opciones de target.")
                return
            target = st.selectbox(
                "Seleccione la variable objetivo (Target):",
                target_options,
                key="target_select"
            )
        if not features:
            st.warning("⚠️ Debe seleccionar al menos una variable independiente.")
            return
        st.markdown("---")
        if tipo_modelo == "Clasificación":
            if st.button("🚀 Entrenar TODOS los Modelos de Clasificación", key="train_all_clf"):
                with st.spinner("Entrenando todos los modelos..."):
                    try:
                        X = df_ml[features].dropna()
                        y = df_ml.loc[X.index, target].dropna()
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        if len(np.unique(y)) < 2:
                            st.error("❌ La variable objetivo debe tener al menos 2 clases diferentes.")
                            return
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        modelos = {
                            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                            "SVM": SVC(kernel='rbf', random_state=42, probability=True),
                            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
                        }
                        resultados = []
                        modelos_entrenados = {}
                        for nombre, modelo in modelos.items():
                            modelo.fit(X_train_scaled, y_train)
                            y_pred = modelo.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            resultados.append({
                                'Modelo': nombre,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1-Score': f1
                            })
                            modelos_entrenados[nombre] = {
                                'model': modelo,
                                'scaler': scaler,
                                'X_test': X_test_scaled,
                                'y_test': y_test,
                                'y_pred': y_pred,
                                'features': features,
                                'target': target,
                                'type': 'classification',
                                'algorithm': nombre
                            }
                        st.session_state.ml_models['all_models'] = modelos_entrenados
                        resultados_df = pd.DataFrame(resultados)
                        st.session_state.ml_models['resultados_clf'] = resultados_df
                        st.success("✅ Todos los modelos de clasificación entrenados exitosamente")
                        st.dataframe(resultados_df, use_container_width=True)
                        # Botón para descargar resultados
                        output = BytesIO()
                        df_para_excel = preparar_para_excel(resultados_df)
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_para_excel.to_excel(writer, sheet_name='Resultados Clasificación', index=False)
                        output.seek(0)
                        st.download_button(
                            label="⬇️ Descargar Resultados Clasificación (Excel)",
                            data=output,
                            file_name=f"resultados_clasificacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error entrenando modelos: {e}")
        elif tipo_modelo == "Regresión":
            st.markdown("### 📈 Algoritmos de Regresión")
            algoritmo = st.selectbox(
                "Seleccione el algoritmo:",
                ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Random Forest Regressor"],
                key="reg_algo"
            )
            if st.button("🚀 Entrenar Modelo de Regresión", key="train_reg"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        X = df_ml[features].dropna()
                        y = df_ml.loc[X.index, target].dropna()
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        if algoritmo == "Linear Regression":
                            model = LinearRegression()
                        elif algoritmo == "Ridge":
                            model = Ridge(alpha=1.0, random_state=42)
                        elif algoritmo == "Lasso":
                            model = Lasso(alpha=1.0, random_state=42)
                        elif algoritmo == "ElasticNet":
                            model = ElasticNet(alpha=1.0, random_state=42)
                        elif algoritmo == "Random Forest Regressor":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        st.session_state.ml_models['last_model'] = {
                            'model': model,
                            'scaler': scaler,
                            'X_test': X_test_scaled,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'features': features,
                            'target': target,
                            'type': 'regression',
                            'algorithm': algoritmo
                        }
                        st.success(f"✅ Modelo {algoritmo} entrenado exitosamente")
                    except Exception as e:
                        st.error(f"Error entrenando modelo: {e}")
        elif tipo_modelo == "Clustering":
            st.markdown("### 🔍 Algoritmos de Clustering")
            algoritmo = st.selectbox(
                "Seleccione el algoritmo:",
                ["K-Means", "DBSCAN", "Agglomerative Clustering"],
                key="cluster_algo"
            )
            if algoritmo == "K-Means":
                # Método del codo
                st.markdown("#### 📏 Método del Codo para seleccionar k")
                if st.button("📊 Calcular Método del Codo"):
                    try:
                        X = df_ml[features].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        k_range = range(1, 11)
                        inertias = []
                        for k in k_range:
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            kmeans.fit(X_scaled)
                            inertias.append(kmeans.inertia_)
                        fig = px.line(
                            x=list(k_range),
                            y=inertias,
                            title='Método del Codo',
                            labels={'x': 'Número de clusters (k)', 'y': 'Inercia'},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1597")
                        st.session_state.ml_models['codo_data'] = {'k': list(k_range), 'inercia': inertias}
                    except Exception as e:
                        st.error(f"Error calculando método del codo: {e}")
                n_clusters = st.slider("Número de clusters:", 2, 10, 3, key="kmeans_clusters")
            elif algoritmo == "DBSCAN":
                eps = st.slider("Epsilon:", 0.1, 5.0, 0.5, key="dbscan_eps")
                min_samples = st.slider("Min samples:", 2, 20, 5, key="dbscan_min")
            elif algoritmo == "Agglomerative Clustering":
                n_clusters = st.slider("Número de clusters:", 2, 10, 3, key="agg_clusters")
            if st.button("🚀 Entrenar Modelo de Clustering", key="train_cluster"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        X = df_ml[features].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        if algoritmo == "K-Means":
                            model = KMeans(n_clusters=n_clusters, random_state=42)
                        elif algoritmo == "DBSCAN":
                            model = DBSCAN(eps=eps, min_samples=min_samples)
                        elif algoritmo == "Agglomerative Clustering":
                            model = AgglomerativeClustering(n_clusters=n_clusters)
                        clusters = model.fit_predict(X_scaled)
                        # Crear DataFrame con clusters
                        df_result = X.copy()
                        df_result['Cluster'] = clusters
                        st.session_state.ml_models['clustering_result'] = df_result
                        st.session_state.ml_models['last_model'] = {
                            'model': model,
                            'scaler': scaler,
                            'X': X_scaled,
                            'clusters': clusters,
                            'features': features,
                            'type': 'clustering',
                            'algorithm': algoritmo
                        }
                        st.success(f"✅ Modelo {algoritmo} entrenado exitosamente")
                        # Mostrar resultados
                        st.markdown("#### 📊 Resultados del Clustering")
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        st.dataframe(cluster_counts.rename('Cantidad').reset_index().rename(columns={'index': 'Cluster'}), use_container_width=True)
                        # Botón para descargar
                        output = BytesIO()
                        df_para_excel = preparar_para_excel(df_result)
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_para_excel.to_excel(writer, sheet_name='Clusters Asignados', index=False)
                        output.seek(0)
                        st.download_button(
                            label="⬇️ Descargar Resultados Clustering (Excel)",
                            data=output,
                            file_name=f"resultados_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error entrenando modelo: {e}")
    with tab3:
        st.markdown("### 📊 Resultados y Evaluación del Modelo")
        if 'all_models' in st.session_state.ml_models:
            st.markdown("#### 📋 Resultados Comparativos de Clasificación")
            resultados_df = st.session_state.ml_models['resultados_clf']
            st.dataframe(resultados_df, use_container_width=True)
            selected_model = st.selectbox("Seleccione un modelo para ver detalles:", list(st.session_state.ml_models['all_models'].keys()))
            model_data = st.session_state.ml_models['all_models'][selected_model]
        elif 'last_model' in st.session_state.ml_models:
            model_data = st.session_state.ml_models['last_model']
        else:
            st.info("ℹ️ Primero entrene un modelo en la pestaña 'Entrenamiento de Modelos'")
            return
        st.markdown(f"""
        <div class='info-box'>
            <h4 style='color: #667eea; margin-top: 0;'>📌 Modelo Entrenado</h4>
            <p><strong>Tipo:</strong> {model_data['type'].title()}</p>
            <p><strong>Algoritmo:</strong> {model_data['algorithm']}</p>
            <p><strong>Features:</strong> {', '.join(model_data['features'][:5])}{'...' if len(model_data['features']) > 5 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if model_data['type'] == 'classification':
            st.markdown("### 🎯 Métricas de Clasificación")
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("🎯 Accuracy", f"{accuracy:.4f}")
            with col2:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("🎪 Precision", f"{precision:.4f}")
            with col3:
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("📡 Recall", f"{recall:.4f}")
            with col4:
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("⚖️ F1-Score", f"{f1:.4f}")
            st.markdown("---")
            st.markdown("### 📊 Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
            ax.set_title('Matriz de Confusión')
            ax.set_ylabel('Valor Real')
            ax.set_xlabel('Valor Predicho')
            st.pyplot(fig)
            st.markdown("---")
            if hasattr(model_data['model'], 'feature_importances_'):
                st.markdown("### 🎯 Importancia de Features")
                importances = model_data['model'].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': model_data['features'],
                    'Importancia': importances
                }).sort_values('Importancia', ascending=False)
                fig = px.bar(
                    feature_importance_df,
                    x='Importancia',
                    y='Feature',
                    orientation='h',
                    title='Importancia de Variables',
                    color='Importancia',
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1717")
        elif model_data['type'] == 'regression':
            st.markdown("### 📈 Métricas de Regresión")
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            col1, col2, col3 = st.columns(3)
            with col1:
                mse = mean_squared_error(y_test, y_pred)
                st.metric("📊 MSE", f"{mse:.4f}")
            with col2:
                rmse = np.sqrt(mse)
                st.metric("📉 RMSE", f"{rmse:.4f}")
            with col3:
                r2 = r2_score(y_test, y_pred)
                st.metric("📈 R² Score", f"{r2:.4f}")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Predicciones vs Real")
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    title='Valores Predichos vs Valores Reales',
                    labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
                    color_discrete_sequence=['#667eea'],
                    opacity=0.6
                )
                fig.add_trace(
                    go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Línea Perfecta',
                        line=dict(color='red', dash='dash')
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1753")
            with col2:
                st.markdown("### 📊 Distribución de Residuos")
                residuos = y_test - y_pred
                fig = px.histogram(
                    residuos,
                    nbins=50,
                    title='Distribución de Residuos',
                    color_discrete_sequence=['#764ba2']
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1763")
        elif model_data['type'] == 'clustering':
            st.markdown("### 🔍 Resultados de Clustering")
            clusters = model_data['clusters']
            col1, col2, col3 = st.columns(3)
            with col1:
                n_clusters_found = len(np.unique(clusters))
                st.metric("🎯 Clusters Encontrados", n_clusters_found)
            with col2:
                cluster_counts = pd.Series(clusters).value_counts()
                st.metric("📊 Cluster Más Grande", cluster_counts.max())
            with col3:
                if n_clusters_found > 1:
                    silhouette_avg = silhouette_score(model_data['X'], clusters)
                    st.metric("📈 Silhouette Score", f"{silhouette_avg:.4f}")
            st.markdown("---")
            st.markdown("### 📊 Distribución de Clusters")
            cluster_df = pd.DataFrame({
                'Cluster': clusters,
                'Count': 1
            })
            cluster_summary = cluster_df.groupby('Cluster').count().reset_index()
            fig = px.bar(
                cluster_summary,
                x='Cluster',
                y='Count',
                title='Número de Puntos por Cluster',
                color='Count',
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1793")
            st.markdown("---")
            st.markdown("### 🎨 Visualización en 2D (PCA)")
            if model_data['X'].shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(model_data['X'])
                fig = px.scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    color=clusters.astype(str),
                    title='Clusters en Espacio PCA',
                    labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1807")
            # Mostrar tabla si existe
            if 'clustering_result' in st.session_state.ml_models:
                st.markdown("### 📋 Tabla con Asignación de Clusters")
                df_result = st.session_state.ml_models['clustering_result']
                st.dataframe(df_result, use_container_width=True)
        st.markdown("---")
        st.markdown("### 💾 Exportar Modelo")
        if st.button("💾 Descargar Modelo Entrenado", key="download_model"):
            try:
                buffer = BytesIO()
                pickle.dump(model_data, buffer)
                buffer.seek(0)
                st.download_button(
                    label="⬇️ Descargar Modelo (.pkl)",
                    data=buffer,
                    file_name=f"modelo_{model_data['algorithm'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error exportando modelo: {e}")
def seccion_geo_inteligencia():
    """Análisis geoespacial de riesgos de corrupción"""
    if not st.session_state.datasets:
        st.info("ℹ️ Por favor, cargue archivos desde el panel lateral.")
        return
    
    if st.session_state.dataset_consolidado is not None:
        df = st.session_state.dataset_consolidado
    else:
        df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    
    st.markdown("### 🌍 Geo Inteligencia Anticorrupción")
    
    # Calcular métricas si no existen
    if st.session_state.df_metricas_persona is None:
        with st.spinner("Calculando métricas patrimoniales..."):
            st.session_state.df_metricas_persona = calcular_metricas_por_persona(df)
    
    df_metricas = st.session_state.df_metricas_persona
    
    tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Riesgo", "📊 Análisis por Estado", "📈 Métricas Geográficas"])
    
    with tab1:
        st.markdown("### 🗺️ Mapa de Riesgo por Entidad Federativa")
        
        if 'Estado' not in df_metricas.columns or df_metricas['Estado'].isna().all():
            st.warning("⚠️ No hay información de estados disponible.")
            return
        
        # Agregar por estado usando el promedio de ingresos totales
        df_estados = df_metricas.groupby('Estado').agg({
            'NombreCompleto': 'count',
            'ingresos_totales_promedio': 'mean',
            'nivel_riesgo_corrupcion': lambda x: (x == 'ALTO').sum()
        }).reset_index()
        
        df_estados.columns = ['Estado', 'Total_Servidores', 'Ingreso_Promedio', 'Casos_Alto_Riesgo']
        df_estados = df_estados[df_estados['Estado'].notna()]
        
        # Agregar coordenadas
        df_estados['Lat'] = df_estados['Estado'].map(lambda x: coordenadas_estados.get(x, (None, None))[0])
        df_estados['Lon'] = df_estados['Estado'].map(lambda x: coordenadas_estados.get(x, (None, None))[1])
        df_estados = df_estados.dropna(subset=['Lat', 'Lon'])
        
        if len(df_estados) == 0:
            st.warning("⚠️ No se pudieron georreferenciar los estados.")
            return
        
        # Clasificar nivel de riesgo
        percentil_33 = df_estados['Casos_Alto_Riesgo'].quantile(0.33)
        percentil_67 = df_estados['Casos_Alto_Riesgo'].quantile(0.67)
        
        def clasificar_riesgo(x):
            if x <= percentil_33:
                return 'Bajo'
            elif x <= percentil_67:
                return 'Medio'
            else:
                return 'Alto'
        
        df_estados['Nivel_Riesgo'] = df_estados['Casos_Alto_Riesgo'].apply(clasificar_riesgo)
        
        color_map = {'Bajo': '#4CAF50', 'Medio': '#ff9800', 'Alto': '#f44336'}
        
        fig = px.scatter_mapbox(
            df_estados,
            lat='Lat',
            lon='Lon',
            size='Total_Servidores',
            color='Nivel_Riesgo',
            hover_name='Estado',
            hover_data={
                'Total_Servidores': ':,',
                'Ingreso_Promedio': ':$,.2f',
                'Casos_Alto_Riesgo': True,
                'Lat': False,
                'Lon': False
            },
            color_discrete_map=color_map,
            size_max=50,
            zoom=4,
            center={'lat': 23.6345, 'lon': -102.5528},
            title='Mapa de Riesgo de Corrupción por Estado'
        )
        fig.update_layout(mapbox_style='open-street-map', height=600)
        st.plotly_chart(fig, use_container_width=True, key="geo_mapa_riesgo")
        
        # Resumen
        col1, col2, col3 = st.columns(3)
        alto_riesgo = len(df_estados[df_estados['Nivel_Riesgo'] == 'Alto'])
        medio_riesgo = len(df_estados[df_estados['Nivel_Riesgo'] == 'Medio'])
        bajo_riesgo = len(df_estados[df_estados['Nivel_Riesgo'] == 'Bajo'])
        
        with col1:
            st.markdown(f"""
            <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;'>
                <p style='margin: 0; font-weight: 600; color: #f44336;'>🔴 Alto Riesgo</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #f44336;'>{alto_riesgo} estados</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800;'>
                <p style='margin: 0; font-weight: 600; color: #ff9800;'>🟡 Medio Riesgo</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #ff9800;'>{medio_riesgo} estados</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                <p style='margin: 0; font-weight: 600; color: #4CAF50;'>🟢 Bajo Riesgo</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #4CAF50;'>{bajo_riesgo} estados</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Análisis Detallado por Estado")
        
        estados_disponibles = sorted(df_metricas['Estado'].dropna().unique().tolist())
        if not estados_disponibles:
            st.warning("⚠️ No hay estados disponibles.")
            return
        
        estado_sel = st.selectbox("Seleccione un estado:", estados_disponibles, key="geo_estado_sel")
        
        df_estado = df_metricas[df_metricas['Estado'] == estado_sel]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Servidores", f"{len(df_estado):,}")
        with col2:
            alto_riesgo = (df_estado['nivel_riesgo_corrupcion'] == 'ALTO').sum()
            st.metric("🔴 Alto Riesgo", f"{alto_riesgo:,}")
        with col3:
            ingreso_prom = df_estado['ingresos_totales_promedio'].mean()
            st.metric("💰 Ingreso Promedio", f"${ingreso_prom:,.2f}")
        with col4:
            patrimonio_prom = df_estado['Importe_bienesInm'].mean()
            st.metric("🏛️ Patrimonio Promedio", f"${patrimonio_prom:,.2f}")
        
        st.markdown("---")
        
        st.markdown("#### 🎯 Top 10 Servidores con Mayor Riesgo")
        top_riesgo = df_estado.nlargest(10, 'delta_max_pct')[
            ['NombreCompleto', 'Cargo', 'ingresos_totales_promedio', 'delta_max_pct', 'nivel_riesgo_corrupcion']
        ]
        st.dataframe(top_riesgo, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Métricas Geográficas Comparativas")
        
        if len(df_estados) > 0:
            st.markdown("#### 💰 Comparación de Ingresos Promedio por Estado")
            fig = px.bar(
                df_estados.sort_values('Ingreso_Promedio', ascending=False).head(18),
                x='Estado',
                y='Ingreso_Promedio',
                color='Nivel_Riesgo',
                color_discrete_map=color_map,
                title='Ingresos Promedio por Estado',
                labels={'Ingreso_Promedio': 'Ingreso Promedio ($)'}
            )
            st.plotly_chart(fig, use_container_width=True, key="geo_ingresos_bar")
            
            st.markdown("#### 🔴 Estados con Más Casos de Alto Riesgo")
            fig2 = px.bar(
                df_estados.sort_values('Casos_Alto_Riesgo', ascending=False).head(18),
                x='Estado',
                y='Casos_Alto_Riesgo',
                color='Nivel_Riesgo',
                color_discrete_map=color_map,
                title='Casos de Alto Riesgo por Estado'
            )
            st.plotly_chart(fig2, use_container_width=True, key="geo_casos_bar")
def seccion_evolucion_patrimonial():
    """Análisis de evolución patrimonial y alertas de corrupción"""
    if not st.session_state.datasets:
        st.info("ℹ️ Por favor, cargue archivos desde el panel lateral.")
        return
    
    if st.session_state.dataset_consolidado is not None:
        df = st.session_state.dataset_consolidado
    else:
        df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    
    st.markdown("### 💰 Evolución Patrimonial y Análisis de Riesgos")
    
    # Calcular métricas si no existen
    if st.session_state.df_metricas_persona is None:
        with st.spinner("Calculando métricas patrimoniales detalladas..."):
            st.session_state.df_metricas_persona = calcular_metricas_por_persona(df)
    
    df_metricas = st.session_state.df_metricas_persona
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tabla Principal",
        "🔍 Buscador Inteligente",
        "📈 Visualizaciones",
        "⚠️ Alertas de Corrupción"
    ])
    
    with tab1:
        st.markdown("#### 📊 Tabla Principal de Evolución Patrimonial")
        
        st.markdown(f"""
        <div class='info-box'>
            <p style='margin: 0; color: #667eea; font-weight: 600;'>
            📌 Total de servidores públicos analizados: {len(df_metricas):,}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            estados_filtro = ['TODOS'] + sorted(df_metricas['Estado'].dropna().unique().tolist())
            estado_filtro = st.selectbox("Estado:", estados_filtro, key="evol_estado_filtro")
        
        with col2:
            riesgo_filtro = st.multiselect(
                "Nivel de Riesgo:",
                ['BAJO', 'MEDIO', 'ALTO'],
                default=['ALTO'],
                key="evol_riesgo_filtro"
            )
        
        with col3:
            tiene_ceros = st.selectbox(
                "¿Tiene valores cero?",
                ['TODOS', 'SÍ', 'NO'],
                key="evol_ceros_filtro"
            )
        
        # Aplicar filtros
        df_filtrado = df_metricas.copy()
        
        if estado_filtro != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['Estado'] == estado_filtro]
        
        if riesgo_filtro:
            df_filtrado = df_filtrado[df_filtrado['nivel_riesgo_corrupcion'].isin(riesgo_filtro)]
        
        if tiene_ceros == 'SÍ':
            df_filtrado = df_filtrado[df_filtrado['tiene_valores_cero'] == True]
        elif tiene_ceros == 'NO':
            df_filtrado = df_filtrado[df_filtrado['tiene_valores_cero'] == False]
        
        # Seleccionar columnas para mostrar
        columnas_mostrar = [
            'NombreCompleto', 'Estado', 'Dependencia', 'Cargo',
            'n_declaraciones', 'Decla_sin0SP', 'Decla_sin0OI', 'Decla_sin0TI',
            'ingreso_sp_promedio', 'max_ingresos_SP', 'min_ingresos_SP',
            'otros_ingresos_promedio', 'max_ingresos_OI', 'min_ingresos_OI',
            'ingresos_totales_promedio', 'max_ingresos_totales', 'min_ingresos_totales',
            'tiene_valores_cero', 'pct_incremento_anual',
            'No_bienesInm', 'Importe_bienesInm',
            'No_autos', 'Importe_autos',
            'No_inversiones', 'Importe_inversiones',
            'No_deudas', 'Importe_adeudos_total',
            'alerta_ingresos', 'alerta_otros_ingresos', 'alerta_adeudos', 'alerta_bienes',
            'alerta_global_riesgo', 'alerta_ingreso_anomalo',
            'nivel_riesgo_corrupcion', 'alertas_detalle'
        ]
        
        # Filtrar solo columnas que existen
        columnas_mostrar = [col for col in columnas_mostrar if col in df_filtrado.columns]
        
        st.markdown(f"**Registros mostrados:** {len(df_filtrado):,}")
        st.dataframe(df_filtrado[columnas_mostrar], use_container_width=True, height=500)
        
        # Descargar
        output = BytesIO()
        df_para_excel = preparar_para_excel(df_filtrado[columnas_mostrar])
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_para_excel.to_excel(writer, sheet_name='Evolución Patrimonial', index=False)
        output.seek(0)
        st.download_button(
            label="⬇️ Descargar Tabla Completa (Excel)",
            data=output,
            file_name=f"evolucion_patrimonial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with tab2:
        st.markdown("#### 🔍 Buscador Inteligente de Servidores Públicos")
        
        if FUZZY_AVAILABLE:
            nombre_buscar = st.text_input(
                "Ingrese el nombre del servidor público:",
                key="evol_buscador"
            )
            
            if nombre_buscar:
                nombre_norm = normalizar_texto(nombre_buscar)
                matches = process.extract(
                    nombre_norm,
                    df_metricas['NombreCompleto'].tolist(),
                    limit=10,
                    scorer=fuzz.token_sort_ratio
                )
                
                st.markdown("##### 🎯 Coincidencias Encontradas:")
                for nombre, score in matches:
                    if score >= 70:
                        servidor = df_metricas[df_metricas['NombreCompleto'] == nombre].iloc[0]
                        
                        with st.expander(f"📋 {nombre} (Match: {score}%)", expanded=(score >= 90)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Información Básica:**")
                                st.write(f"**Estado:** {servidor.get('Estado', 'N/A')}")
                                st.write(f"**Dependencia:** {servidor.get('Dependencia', 'N/A')}")
                                st.write(f"**Cargo:** {servidor.get('Cargo', 'N/A')}")
                                st.write(f"**Declaraciones:** {servidor['n_declaraciones']}")
                                
                                st.markdown("**Datos Originales por Año:**")
                                # Aquí se mostraría el historial año por año si está disponible
                                st.info("📅 Historial detallado disponible en datos originales")
                            
                            with col2:
                                st.markdown("**Métricas Financieras:**")
                                st.metric("💰 Ingreso SP Promedio", f"${servidor['ingreso_sp_promedio']:,.2f}")
                                st.metric("📊 Otros Ingresos Promedio", f"${servidor['otros_ingresos_promedio']:,.2f}")
                                st.metric("💵 Ingresos Totales Promedio", f"${servidor['ingresos_totales_promedio']:,.2f}")
                                
                                st.markdown("**Máximos y Mínimos:**")
                                st.write(f"**MAX Ingresos:** ${servidor['max_ingresos_totales']:,.2f}")
                                st.write(f"**MIN Ingresos:** ${servidor['min_ingresos_totales']:,.2f}")
                                st.write(f"**Incremento Anual:** {servidor['pct_incremento_anual']:.1f}%")
                            
                            st.markdown("---")
                            
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                st.markdown("**Patrimonio:**")
                                st.write(f"🏠 **Bienes Inmuebles:** {servidor['No_bienesInm']} (${servidor['Importe_bienesInm']:,.2f})")
                                st.write(f"🚗 **Vehículos:** {servidor['No_autos']} (${servidor['Importe_autos']:,.2f})")
                                st.write(f"📈 **Inversiones:** {servidor['No_inversiones']} (${servidor['Importe_inversiones']:,.2f})")
                                st.write(f"💳 **Adeudos:** {servidor['No_deudas']} (${servidor['Importe_adeudos_total']:,.2f})")
                            
                            with col4:
                                st.markdown("**Alertas:**")
                                if servidor['alerta_global_riesgo']:
                                    st.error("⚠️ ALERTA GLOBAL DE RIESGO")
                                else:
                                    st.success("✅ Sin alerta global")
                                
                                if servidor['alerta_ingresos']:
                                    st.warning("📊 Incremento de ingresos inusual")
                                if servidor['alerta_otros_ingresos']:
                                    st.warning("💰 Otros ingresos desproporcionados")
                                if servidor['alerta_adeudos']:
                                    st.warning("💳 Adeudos elevados")
                                if servidor['alerta_bienes']:
                                    st.warning("🏠 Bienes desproporcionados")
                                
                                st.markdown(f"**Nivel de Riesgo:** `{servidor['nivel_riesgo_corrupcion']}`")
                                
                                if servidor.get('alertas_detalle', 'SIN_ALERTA') != 'SIN_ALERTA':
                                    st.markdown(f"**Detalles:** {servidor['alertas_detalle']}")
        else:
            st.warning("⚠️ Instale fuzzywuzzy para búsqueda inteligente: `pip install fuzzywuzzy python-Levenshtein`")
            
            # Búsqueda simple
            nombre_buscar = st.text_input("Ingrese el nombre:", key="evol_buscador_simple")
            if nombre_buscar:
                nombre_norm = normalizar_texto(nombre_buscar)
                resultados = df_metricas[df_metricas['NombreCompleto'].str.contains(nombre_norm, na=False)]
                st.dataframe(resultados, use_container_width=True)
    
    with tab3:
        st.markdown("#### 📈 Visualizaciones de Evolución Patrimonial")
        
        tipo_viz = st.selectbox(
            "Seleccione tipo de visualización:",
            ["Distribución", "Boxplot", "Treemap", "Ley de Benford", "Cascada"],
            key="evol_viz_tipo"
        )
        
        variable = st.selectbox(
            "Variable a analizar:",
            ["ingreso_sp_promedio", "otros_ingresos_promedio", "ingresos_totales_promedio"],
            key="evol_viz_var"
        )
        
        if tipo_viz == "Distribución":
            valores = df_metricas[variable].dropna()
            if len(valores) > 0:
                fig = px.histogram(
                    x=valores,
                    nbins=50,
                    title=f'Distribución de {variable}',
                    labels={'x': variable, 'y': 'Frecuencia'},
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True, key="evol_dist_hist")
        
        elif tipo_viz == "Boxplot":
            fig = px.box(
                df_metricas,
                y=variable,
                title=f'Boxplot de {variable}',
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True, key="evol_boxplot")
        
        elif tipo_viz == "Treemap":
            if 'Estado' in df_metricas.columns:
                df_tree = df_metricas.groupby('Estado')[variable].sum().reset_index()
                df_tree = df_tree[df_tree[variable] > 0]
                
                fig = px.treemap(
                    df_tree,
                    path=['Estado'],
                    values=variable,
                    title=f'Treemap de {variable} por Estado'
                )
                st.plotly_chart(fig, use_container_width=True, key="evol_treemap")
        
        elif tipo_viz == "Ley de Benford":
            valores = df_metricas[variable].dropna()
            valores = valores[valores > 0]
            
            if len(valores) > 0:
                primeros_digitos = [int(str(int(v))[0]) for v in valores if v > 0]
                conteo_digitos = pd.Series(primeros_digitos).value_counts().sort_index()
                
                # Ley de Benford teórica
                benford_teorico = {d: np.log10(1 + 1/d) * 100 for d in range(1, 10)}
                
                df_benford = pd.DataFrame({
                    'Dígito': list(range(1, 10)),
                    'Observado': [conteo_digitos.get(d, 0) / len(primeros_digitos) * 100 for d in range(1, 10)],
                    'Benford': [benford_teorico[d] for d in range(1, 10)]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_benford['Dígito'], y=df_benford['Observado'], name='Observado'))
                fig.add_trace(go.Scatter(x=df_benford['Dígito'], y=df_benford['Benford'], name='Ley de Benford', mode='lines+markers'))
                fig.update_layout(title='Análisis Ley de Benford', xaxis_title='Primer Dígito', yaxis_title='Porcentaje (%)')
                st.plotly_chart(fig, use_container_width=True, key="evol_benford")
        
        elif tipo_viz == "Cascada":
            df_cascade = df_metricas.nlargest(20, variable)[['NombreCompleto', variable]]
            
            fig = go.Figure(go.Waterfall(
                x=df_cascade['NombreCompleto'],
                y=df_cascade[variable],
                text=[f"${v:,.0f}" for v in df_cascade[variable]],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(title=f'Gráfico de Cascada - Top 20 {variable}', showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="evol_waterfall")
    
    with tab4:
        st.markdown("#### ⚠️ Panel de Alertas de Corrupción")
        
        # Resumen de alertas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            alto_riesgo = (df_metricas['nivel_riesgo_corrupcion'] == 'ALTO').sum()
            st.metric("🔴 Alto Riesgo", f"{alto_riesgo:,}")
        with col2:
            medio_riesgo = (df_metricas['nivel_riesgo_corrupcion'] == 'MEDIO').sum()
            st.metric("🟡 Medio Riesgo", f"{medio_riesgo:,}")
        with col3:
            con_alerta_global = df_metricas['alerta_global_riesgo'].sum()
            st.metric("⚠️ Alerta Global", f"{con_alerta_global:,}")
        with col4:
            con_incremento = df_metricas['alerta_ingresos'].sum()
            st.metric("📈 Incremento Inusual", f"{con_incremento:,}")
        
        st.markdown("---")
        
        # Top alertas
        st.markdown("##### 🔴 Top 20 Casos de Alto Riesgo")
        top_alertas = df_metricas[df_metricas['nivel_riesgo_corrupcion'] == 'ALTO'].nlargest(20, 'delta_max_pct')
        
        columnas_alertas = [
            'NombreCompleto', 'Estado', 'Cargo',
            'ingresos_totales_promedio', 'delta_max_pct',
            'Importe_bienesInm', 'Importe_adeudos_total',
            'alertas_detalle', 'nivel_riesgo_corrupcion'
        ]
        columnas_alertas = [col for col in columnas_alertas if col in top_alertas.columns]
        
        st.dataframe(top_alertas[columnas_alertas], use_container_width=True)

def seccion_nepotismo():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🕸 Detección de Nepotismo")
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #ff9800; margin-top: 0;'>🔍 Análisis de Relaciones Sospechosas</h3>
        <p>Detección de posibles relaciones familiares o de compadrazgo dentro del mismo ente público
        mediante análisis de similitud de nombres y coincidencias de apellidos.</p>
    </div>
    """, unsafe_allow_html=True)
    if 'NombreCompleto' not in df.columns:
        st.error("❌ No se encontró la columna 'NombreCompleto'. Verifique que los datos estén procesados correctamente.")
        return
    st.markdown("---")
    df_coincidencias = pd.DataFrame()
    tab1, tab2, tab3 = st.tabs([
        "📊 Análisis de Similitud",
        "🕸 Grafos de Relación",
        "📋 Tabla de Casos Sospechosos"
    ])
    with tab1:
        st.markdown("### 📊 Análisis de Similitud de Nombres")
        umbral_similitud = st.slider(
            "Umbral de similitud (%):",
            50, 100, 85,
            help="Porcentaje mínimo de similitud para considerar una relación",
            key="umbral_similitud"
        )
        with st.spinner("Analizando similitudes..."):
            if 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico' in df.columns:
                ente_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico'
                df_nombres = df[['NombreCompleto', ente_col]].drop_duplicates()
                coincidencias = []
                if FUZZY_AVAILABLE:
                    for ente in df_nombres[ente_col].unique():
                        if pd.isna(ente):
                            continue
                        df_ente = df_nombres[df_nombres[ente_col] == ente]
                        nombres = df_ente['NombreCompleto'].tolist()
                        for i in range(len(nombres)):
                            for j in range(i+1, len(nombres)):
                                if pd.notna(nombres[i]) and pd.notna(nombres[j]):
                                    similitud = fuzz.ratio(nombres[i], nombres[j])
                                    if similitud >= umbral_similitud:
                                        coincidencias.append({
                                            'Nombre 1': nombres[i],
                                            'Nombre 2': nombres[j],
                                            'Similitud (%)': similitud,
                                            'Ente Público': ente,
                                            'Tipo': 'Alta Similitud'
                                        })
                                    apellidos_1 = nombres[i].split()[1:] if len(nombres[i].split()) > 1 else []
                                    apellidos_2 = nombres[j].split()[1:] if len(nombres[j].split()) > 1 else []
                                    if apellidos_1 and apellidos_2:
                                        if any(ap in apellidos_2 for ap in apellidos_1):
                                            if similitud < umbral_similitud:
                                                coincidencias.append({
                                                    'Nombre 1': nombres[i],
                                                    'Nombre 2': nombres[j],
                                                    'Similitud (%)': similitud,
                                                    'Ente Público': ente,
                                                    'Tipo': 'Mismo Apellido'
                                                })
                    df_coincidencias = pd.DataFrame(coincidencias)
                    if len(df_coincidencias) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔍 Total Coincidencias", len(df_coincidencias))
                        with col2:
                            alta_similitud = len(df_coincidencias[df_coincidencias['Tipo'] == 'Alta Similitud'])
                            st.metric("📊 Alta Similitud", alta_similitud)
                        with col3:
                            mismo_apellido = len(df_coincidencias[df_coincidencias['Tipo'] == 'Mismo Apellido'])
                            st.metric("👥 Mismo Apellido", mismo_apellido)
                        st.markdown("---")
                        st.markdown("#### 📊 Distribución por Tipo de Coincidencia")
                        fig = px.pie(
                            df_coincidencias,
                            names='Tipo',
                            title='Distribución de Tipos de Coincidencia',
                            color='Tipo',
                            color_discrete_map={
                                'Alta Similitud': '#ff9800',
                                'Mismo Apellido': '#f44336'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_2716")
                        st.markdown("---")
                        st.markdown("#### 📋 Top 20 Coincidencias por Similitud")
                        df_top = df_coincidencias.sort_values('Similitud (%)', ascending=False).head(20)
                        st.dataframe(df_top, use_container_width=True)
                        st.markdown("---")
                        st.markdown("#### 🏢 Entes con Más Coincidencias")
                        entes_coincidencias = df_coincidencias['Ente Público'].value_counts().head(10)
                        fig = px.bar(
                            x=entes_coincidencias.values,
                            y=entes_coincidencias.index,
                            orientation='h',
                            title='Top 10 Entes con Más Coincidencias',
                            labels={'x': 'Número de Coincidencias', 'y': 'Ente Público'},
                            color=entes_coincidencias.values,
                            color_continuous_scale='Oranges'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_2733")
                    else:
                        st.info("ℹ️ No se encontraron coincidencias con el umbral seleccionado.")
                else:
                    st.warning("⚠️ FuzzyWuzzy no está disponible. Instale con: pip install fuzzywuzzy python-Levenshtein")
            else:
                st.warning("⚠️ No se encontró la columna de Ente Público en el dataset.")
    with tab2:
        st.markdown("### 🕸 Grafos de Relación")
        if len(df_coincidencias) > 0 and NETWORKX_AVAILABLE:
            st.markdown("""
            <div class='info-box'>
                <p style='margin: 0; color: #ff9800; font-weight: 600;'>
                📊 Visualización de red de relaciones entre servidores públicos con coincidencias detectadas
                </p>
            </div>
            """, unsafe_allow_html=True)
            ente_seleccionado = st.selectbox(
                "Seleccione un ente público para visualizar:",
                df_coincidencias['Ente Público'].unique(),
                key="ente_grafo"
            )
            df_ente_grafo = df_coincidencias[df_coincidencias['Ente Público'] == ente_seleccionado]
            G = nx.Graph()
            for _, row in df_ente_grafo.iterrows():
                G.add_edge(
                    row['Nombre 1'], 
                    row['Nombre 2'],
                    weight=row['Similitud (%)'],
                    tipo=row['Tipo']
                )
            st.markdown(f"#### 🕸 Red de Relaciones: {ente_seleccionado}")
            st.markdown(f"**Nodos:** {G.number_of_nodes()} | **Conexiones:** {G.number_of_edges()}")
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=2, iterations=50)
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#ff9800'),
                    hoverinfo='none',
                    mode='lines',
                    opacity=0.5
                )
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_size.append(G.degree(node) * 20 + 20)
                # Corregir colorbar sin titleside
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    textfont=dict(size=10, color='#333'),
                    marker=dict(
                        showscale=True,
                        colorscale='Oranges',
                        size=node_size,
                        color=[G.degree(node) for node in G.nodes()],
                        colorbar=dict(
                            thickness=15,
                            title=dict(text='Conexiones'),  # Usar dict para título
                            xanchor='left',
                            # titleside removido
                        ),
                        line=dict(width=2, color='white')
                    )
                )
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title=f'Grafo de Relaciones - {ente_seleccionado}',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0,l=0,r=0,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=700
                                ))
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_2824")
                st.markdown("---")
                st.markdown("#### 📊 Estadísticas del Grafo")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔢 Nodos", G.number_of_nodes())
                with col2:
                    st.metric("🔗 Conexiones", G.number_of_edges())
                with col3:
                    densidad = nx.density(G)
                    st.metric("📊 Densidad", f"{densidad:.3f}")
                with col4:
                    grado_promedio = sum(dict(G.degree()).values()) / G.number_of_nodes()
                    st.metric("📈 Grado Promedio", f"{grado_promedio:.2f}")
                st.markdown("---")
                st.markdown("#### 👥 Nodos Más Conectados")
                grados = dict(G.degree())
                top_nodos = sorted(grados.items(), key=lambda x: x[1], reverse=True)[:10]
                df_top_nodos = pd.DataFrame(top_nodos, columns=['Nombre', 'Conexiones'])
                fig = px.bar(
                    df_top_nodos,
                    x='Conexiones',
                    y='Nombre',
                    orientation='h',
                    title='Top 10 Personas Más Conectadas',
                    color='Conexiones',
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_2852")
            else:
                st.info("ℹ️ No hay suficientes datos para generar el grafo.")
        elif not NETWORKX_AVAILABLE:
            st.warning("⚠️ NetworkX no está disponible. Instale con: pip install networkx")
        else:
            st.info("ℹ️ No hay coincidencias detectadas. Ajuste el umbral en la pestaña 'Análisis de Similitud'.")
    with tab3:
        st.markdown("### 📋 Tabla de Casos Sospechosos")
        if len(df_coincidencias) > 0:
            st.markdown("""
            <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <p style='margin: 0; color: #f44336; font-weight: 600;'>
                ⚠️ Casos que requieren investigación adicional por posible nepotismo
                </p>
            </div>
            """, unsafe_allow_html=True)
            df_sospechosos = df_coincidencias.copy()
            def calcular_nivel_sospecha(row):
                puntos = 0
                if row['Similitud (%)'] >= 95:
                    puntos += 3
                elif row['Similitud (%)'] >= 90:
                    puntos += 2
                elif row['Similitud (%)'] >= 85:
                    puntos += 1
                if row['Tipo'] == 'Mismo Apellido':
                    puntos += 2
                if puntos >= 4:
                    return '🔴 CRÍTICO'
                elif puntos >= 2:
                    return '🟡 MEDIO'
                else:
                    return '🟢 BAJO'
            df_sospechosos['Nivel_Sospecha'] = df_sospechosos.apply(calcular_nivel_sospecha, axis=1)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Casos", len(df_sospechosos))
            with col2:
                criticos = len(df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🔴 CRÍTICO'])
                st.markdown(f"""
                <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;'>
                    <p style='margin: 0; font-weight: 600; color: #f44336;'>🔴 Críticos</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #f44336;'>{criticos}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                medios = len(df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🟡 MEDIO'])
                st.markdown(f"""
                <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800;'>
                    <p style='margin: 0; font-weight: 600; color: #ff9800;'>🟡 Medios</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #ff9800;'>{medios}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                bajos = len(df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🟢 BAJO'])
                st.markdown(f"""
                <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                    <p style='margin: 0; font-weight: 600; color: #4CAF50;'>🟢 Bajos</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #4CAF50;'>{bajos}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            filtro_nivel = st.selectbox(
                "Filtrar por nivel de sospecha:",
                ['Todos', '🔴 CRÍTICO', '🟡 MEDIO', '🟢 BAJO'],
                key="filtro_nivel_nepotismo"
            )
            if filtro_nivel == 'Todos':
                df_mostrar = df_sospechosos
            else:
                df_mostrar = df_sospechosos[df_sospechosos['Nivel_Sospecha'] == filtro_nivel]
            df_mostrar = df_mostrar.sort_values(['Nivel_Sospecha', 'Similitud (%)'], ascending=[True, False])
            st.markdown(f"#### 📋 Listado: {len(df_mostrar):,} casos")
            st.dataframe(df_mostrar, use_container_width=True, height=400)
            st.markdown("---")
            if st.button("💾 Exportar Casos Críticos"):
                try:
                    df_criticos = df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🔴 CRÍTICO']
                    output = BytesIO()
                    df_para_excel = preparar_para_excel(df_criticos)
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_para_excel.to_excel(writer, sheet_name='Casos Críticos', index=False)
                        resumen = pd.DataFrame({
                            'Nivel': ['🔴 CRÍTICO', '🟡 MEDIO', '🟢 BAJO'],
                            'Cantidad': [criticos, medios, bajos],
                            'Porcentaje': [
                                f"{(criticos/len(df_sospechosos)*100):.2f}%",
                                f"{(medios/len(df_sospechosos)*100):.2f}%",
                                f"{(bajos/len(df_sospechosos)*100):.2f}%"
                            ]
                        })
                        resumen.to_excel(writer, sheet_name='Resumen', index=False)
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Descargar Excel",
                        data=output,
                        file_name="casos_nepotismo_criticos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error exportando: {e}")
        else:
            st.info("ℹ️ No hay coincidencias detectadas. Ajuste el umbral en la pestaña 'Análisis de Similitud'.")
def seccion_denuncias():
    """Sistema de seguimiento de denuncias con WordCloud y clasificación IA"""
    st.markdown("### 📢 Sistema de Denuncias Ciudadanas")
    
    tab1, tab2, tab3 = st.tabs(["📝 Nueva Denuncia", "📊 Análisis de Denuncias", "☁️ WordCloud"])
    
    with tab1:
        st.markdown("#### 📝 Registrar Nueva Denuncia")
        
        with st.form("form_denuncia"):
            nombre_denunciante = st.text_input("Nombre (opcional):", key="denuncia_nombre")
            servidor_denunciado = st.text_input("Servidor público denunciado:", key="denuncia_servidor")
            dependencia = st.text_input("Dependencia/Institución:", key="denuncia_dep")
            descripcion = st.text_area("Descripción de los hechos:", height=150, key="denuncia_desc")
            evidencia = st.text_input("Enlaces a evidencia (opcional):", key="denuncia_evidencia")
            
            submitted = st.form_submit_button("📤 Enviar Denuncia")
            
            if submitted:
                if servidor_denunciado and descripcion:
                    denuncia = {
                        'fecha': datetime.now(),
                        'denunciante': nombre_denunciante if nombre_denunciante else 'Anónimo',
                        'servidor': servidor_denunciado,
                        'dependencia': dependencia,
                        'descripcion': descripcion,
                        'evidencia': evidencia,
                        'id': len(st.session_state.denuncias) + 1
                    }
                    st.session_state.denuncias.append(denuncia)
                    st.success("✅ Denuncia registrada exitosamente")
                    st.balloons()
                else:
                    st.error("⚠️ Complete los campos obligatorios")
    
    with tab2:
        st.markdown("#### 📊 Historial de Denuncias")
        
        if st.session_state.denuncias:
            df_denuncias = pd.DataFrame(st.session_state.denuncias)
            
            st.markdown(f"**Total de denuncias registradas:** {len(df_denuncias)}")
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Denuncias", len(df_denuncias))
            with col2:
                if 'dependencia' in df_denuncias.columns:
                    deps_unicas = df_denuncias['dependencia'].nunique()
                    st.metric("🏛️ Dependencias", deps_unicas)
            with col3:
                denuncias_anonimas = (df_denuncias['denunciante'] == 'Anónimo').sum()
                st.metric("👤 Anónimas", denuncias_anonimas)
            
            st.markdown("---")
            
            # Mostrar denuncias
            for _, denuncia in df_denuncias.iterrows():
                with st.expander(f"Denuncia #{denuncia['id']} - {denuncia['servidor']} ({denuncia['fecha'].strftime('%Y-%m-%d %H:%M')})"):
                    st.write(f"**Denunciante:** {denuncia['denunciante']}")
                    st.write(f"**Servidor Denunciado:** {denuncia['servidor']}")
                    st.write(f"**Dependencia:** {denuncia['dependencia']}")
                    st.write(f"**Descripción:**")
                    st.info(denuncia['descripcion'])
                    if denuncia.get('evidencia'):
                        st.write(f"**Evidencia:** {denuncia['evidencia']}")
        else:
            st.info("ℹ️ No hay denuncias registradas aún.")
    
    with tab3:
        st.markdown("#### ☁️ WordCloud de Denuncias")
        
        if st.session_state.denuncias and WORDCLOUD_AVAILABLE:
            df_denuncias = pd.DataFrame(st.session_state.denuncias)
            
            # Combinar todas las descripciones
            texto_completo = ' '.join(df_denuncias['descripcion'].astype(str))
            
            if texto_completo.strip():
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(texto_completo)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("ℹ️ No hay suficiente texto para generar WordCloud.")
        elif not WORDCLOUD_AVAILABLE:
            st.warning("⚠️ Instale wordcloud: `pip install wordcloud`")
        else:
            st.info("ℹ️ No hay denuncias registradas.")
def seccion_diccionario_datos():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 📋 Diccionario de Datos")
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #667eea; margin-top: 0;'>📖 Información Completa del Dataset</h3>
        <p>Diccionario completo de todas las columnas con estadísticas y codificación aplicada</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2 = st.tabs([
        "📊 Diccionario Completo",
        "🔢 Diccionario de Codificación"
    ])
    with tab1:
        st.markdown("### 📊 Información Detallada por Columna")
        diccionario = []
        for col in df.columns:
            tipo_dato = str(df[col].dtype)
            valores_nulos = df[col].isnull().sum()
            porcentaje_nulos = (valores_nulos / len(df)) * 100
            valores_unicos = df[col].nunique()
            muestra = df[col].dropna().head(3).tolist()
            muestra_str = ', '.join([str(x)[:50] for x in muestra])
            diccionario.append({
                'Columna': col,
                'Tipo de Dato': tipo_dato,
                'Valores Nulos': valores_nulos,
                '% Nulos': f"{porcentaje_nulos:.2f}%",
                'Valores Únicos': valores_unicos,
                'Muestra': muestra_str
            })
        df_diccionario = pd.DataFrame(diccionario)
        st.markdown(f"**Total de columnas:** {len(df_diccionario)}")
        st.dataframe(df_diccionario, use_container_width=True, height=500)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Estadísticas Generales")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("📋 Total Columnas", len(df.columns))
            with col1b:
                columnas_con_nulos = (df.isnull().sum() > 0).sum()
                st.metric("❌ Columnas con Nulos", columnas_con_nulos)
            with col1c:
                promedio_nulos = df.isnull().sum().mean()
                st.metric("📊 Promedio Nulos", f"{promedio_nulos:.2f}")
        with col2:
            st.markdown("#### 🎯 Distribución de Tipos de Datos")
            tipos_datos = df_diccionario['Tipo de Dato'].value_counts()
            fig = px.pie(
                values=tipos_datos.values,
                names=tipos_datos.index,
                title='Distribución de Tipos de Datos',
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_3016")
        st.markdown("---")
        if st.button("💾 Exportar Diccionario a Excel"):
            try:
                output = BytesIO()
                df_para_excel = preparar_para_excel(df_diccionario)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Diccionario', index=False)
                    if st.session_state.encoding_dicts:
                        dict_rows = []
                        for col, mappings in st.session_state.encoding_dicts.items():
                            for original, codigo in mappings.items():
                                dict_rows.append({
                                    'Columna': col,
                                    'Valor Original': original,
                                    'Código': codigo
                                })
                        if dict_rows:
                            df_encoding = pd.DataFrame(dict_rows)
                            df_encoding.to_excel(writer, sheet_name='Codificación', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Excel",
                    data=output,
                    file_name="diccionario_datos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exportando: {e}")
    with tab2:
        st.markdown("### 🔢 Diccionario de Codificación de Variables")
        if not st.session_state.encoding_dicts:
            st.info("ℹ️ No hay variables codificadas. Primero genere el dataset codificado en la sección de Machine Learning.")
            st.markdown("""
            ### ¿Qué es la Codificación de Variables?
            La codificación de variables es el proceso de convertir datos categóricos (texto) a valores numéricos
            para que puedan ser utilizados en algoritmos de Machine Learning.
            **Ejemplo:**
            - **Original:** Casa, Terreno, Parcela
            - **Codificado:** 1, 2, 3
            **Ventajas:**
            - ✅ Permite usar algoritmos de ML
            - ✅ Mejora el rendimiento
            - ✅ Facilita cálculos matemáticos
            - ✅ Reduce el tamaño de los datos
            """)
        else:
            st.markdown("#### 📖 Columnas Codificadas")
            st.metric("📊 Total Columnas Codificadas", len(st.session_state.encoding_dicts))
            columna_seleccionada = st.selectbox(
                "Seleccione una columna para ver su diccionario:",
                list(st.session_state.encoding_dicts.keys()),
                key="dict_col_select"
            )
            if columna_seleccionada:
                st.markdown(f"#### 🔍 Diccionario: {columna_seleccionada}")
                mapping = st.session_state.encoding_dicts[columna_seleccionada]
                df_mapping = pd.DataFrame([
                    {"Valor Original": k, "Código Numérico": v}
                    for k, v in mapping.items()
                ]).sort_values('Código Numérico')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Valores Únicos", len(mapping))
                with col2:
                    st.metric("🔢 Rango de Códigos", f"0 - {len(mapping)-1}")
                st.dataframe(df_mapping, use_container_width=True, height=400)
                st.markdown("---")
                fig = px.bar(
                    df_mapping,
                    x='Código Numérico',
                    y='Valor Original',
                    orientation='h',
                    title=f'Codificación de {columna_seleccionada}',
                    color='Código Numérico',
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_3093")
def seccion_reporte_ejecutivo():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 📄 Reporte Ejecutivo")
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #667eea; margin-top: 0;'>📊 Resumen Ejecutivo del Análisis Anticorrupción</h3>
        <p>Dashboard consolidado con los hallazgos más importantes del análisis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 📊 Métricas Generales")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄 Total Registros", f"{len(df):,}")
    with col2:
        if 'NombreCompleto' in df.columns:
            servidores_unicos = df['NombreCompleto'].nunique()
            st.metric("👥 Servidores Únicos", f"{servidores_unicos:,}")
    with col3:
        if 'anioEjercicio' in df.columns:
            años = df['anioEjercicio'].dropna().unique()
            st.metric("📅 Años Analizados", len(años))
    with col4:
        ente_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico'
        if ente_col in df.columns:
            entes = df[ente_col].nunique()
            st.metric("🏢 Entes Públicos", f"{entes:,}")
    with col5:
        memoria = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("💾 Datos Procesados", f"{memoria:.2f} MB")
    st.markdown("---")
    st.markdown("## 🚨 Hallazgos Principales")
    ingresos_cols = [col for col in df.columns if col.startswith('declaracion_situacionPatrimonial_ingresos_') 
                    and df[col].dtype in [np.float64, np.int64]]
    if ingresos_cols and 'NombreCompleto' in df.columns:
        df_temp = df.copy()
        df_temp['Total_Ingresos'] = df_temp[ingresos_cols].sum(axis=1)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💰 Top 10 Mayores Ingresos")
            top_ingresos = df_temp.groupby('NombreCompleto')['Total_Ingresos'].sum().sort_values(ascending=False).head(10)
            df_top_ingresos = pd.DataFrame({
                'Servidor Público': top_ingresos.index,
                'Ingresos Totales': [f"${x:,.2f}" for x in top_ingresos.values]
            })
            st.dataframe(df_top_ingresos, use_container_width=True)
        with col2:
            st.markdown("### 📈 Evolución de Ingresos Promedio")
            if 'anioEjercicio' in df.columns:
                evolucion = df_temp.groupby('anioEjercicio')['Total_Ingresos'].mean().reset_index()
                fig = px.line(
                    evolucion,
                    x='anioEjercicio',
                    y='Total_Ingresos',
                    title='Ingresos Promedio por Año',
                    markers=True,
                    line_shape='spline'
                )
                fig.update_traces(line_color='#667eea', marker=dict(size=10, color='#764ba2'))
                st.plotly_chart(fig, use_container_width=True, key="plotly_chart_3156")
    st.markdown("---")
    st.markdown("## 🎯 Recomendaciones")
    recomendaciones = [
        {
            'Prioridad': '🔴 ALTA',
            'Área': 'Evolución Patrimonial',
            'Recomendación': 'Investigar casos con incremento patrimonial superior al 200% sin justificación por ingresos'
        },
        {
            'Prioridad': '🟡 MEDIA',
            'Área': 'Nepotismo',
            'Recomendación': 'Revisar coincidencias de apellidos en mismo ente público'
        },
        {
            'Prioridad': '🟢 BAJA',
            'Área': 'Declaraciones',
            'Recomendación': 'Verificar completitud de declaraciones en servidores con alta rotación'
        }
    ]
    df_recomendaciones = pd.DataFrame(recomendaciones)
    st.dataframe(df_recomendaciones, use_container_width=True)
    st.markdown("---")
    # Botones de descarga para hallazgos clave
    st.markdown("## 📥 Descargar Hallazgos Clave")
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'NombreCompleto' in df.columns and ingresos_cols:
            top_ingresos = df_temp.groupby('NombreCompleto')['Total_Ingresos'].sum().sort_values(ascending=False).head(10).reset_index()
            if st.button("📊 Descargar Top 10 Ingresos"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(top_ingresos)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Top 10 Ingresos', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Top 10 Ingresos (Excel)",
                    data=output,
                    file_name="top_10_ingresos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    with col2:
        if st.session_state.df_alertas is not None and not st.session_state.df_alertas.empty:
            if st.button("🚨 Descargar Alertas Detectadas"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_alertas)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Alertas', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Alertas (Excel)",
                    data=output,
                    file_name="alertas_ejecutivo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    with col3:
        if st.session_state.df_anomalias is not None and not st.session_state.df_anomalias.empty:
            if st.button("⚠️ Descargar Anomalías Detectadas"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_anomalias)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Anomalías', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Anomalías (Excel)",
                    data=output,
                    file_name="anomalias_ejecutivo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    st.markdown("---")
    if st.button("📄 Generar Reporte Completo PDF"):
        st.info("🚧 Funcionalidad de generación de PDF en desarrollo")
def seccion_borrar_datos():
    st.markdown("# 🗑️ Borrar Datos y Sesión")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%); 
    padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(244, 67, 54, 0.3);'>
        <h3 style='color: #f44336; margin-top: 0;'>⚠️ ADVERTENCIA</h3>
        <p style='margin-bottom: 0;'>
        Esta acción eliminará TODOS los datos cargados y reiniciará la sesión.
        Esta operación NO se puede deshacer.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.datasets:
        st.markdown("### 📊 Datos Actualmente Cargados")
        for nombre, df in st.session_state.datasets.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**📁 {nombre}**")
            with col2:
                st.write(f"{len(df):,} registros")
            with col3:
                memoria = df.memory_usage(deep=True).sum() / (1024**2)
                st.write(f"{memoria:.2f} MB")
        st.markdown("---")
        confirmacion = st.text_input(
            "Para confirmar, escriba 'BORRAR' en mayúsculas:",
            key="confirmacion_borrar"
        )
        if confirmacion == "BORRAR":
            if st.button("🗑️ CONFIRMAR BORRADO", type="primary"):
                st.session_state.datasets = {}
                st.session_state.dataset_seleccionado = None
                st.session_state.encoding_dicts = {}
                st.session_state.df_encoded = None
                st.session_state.ml_models = {}
                st.session_state.df_procesado = None
                st.session_state.df_metricas_persona = None
                st.session_state.df_servidores_unicos = None
                st.session_state.df_anomalias = None
                st.session_state.df_alertas = None
                st.success("✅ Todos los datos han sido eliminados exitosamente.")
                st.balloons()
                st.rerun()
        else:
            st.info("ℹ️ Escriba 'BORRAR' para habilitar el botón de confirmación.")
    else:
        st.info("ℹ️ No hay datos cargados en la sesión actual.")
def main():
    """Función principal de la aplicación"""
    st.sidebar.markdown("""
    <div class='hero-header' style='padding: 1.5rem; margin-bottom: 1rem;'>
        <div class='brand-title' style='font-size: 2rem;'>🔍 AURORA ETHICS</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Carga de archivos CSV/Excel tradicionales
    uploaded_files = st.sidebar.file_uploader(
        "📂 Seleccione archivos CSV/Excel",
        type=['csv', 'xls', 'xlsx'],
        accept_multiple_files=True,
        help="Carga archivos sin límite de tamaño"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.datasets:
                with st.spinner(f"Procesando {uploaded_file.name}..."):
                    df = procesar_archivo_cargado(uploaded_file)
                    if df is not None:
                        st.session_state.datasets[uploaded_file.name] = df
                        st.sidebar.success(f"✅ {uploaded_file.name}")
    
    # Botón para consolidar datasets
    if len(st.session_state.datasets) > 1:
        st.sidebar.markdown("---")
        if st.sidebar.button("🔗 Consolidar Todos los Archivos", key="consolidar_btn"):
            consolidar_datasets()

    # ================================
    # 🌟 MOSTRAR SIEMPRE EL ENCABEZADO PRINCIPAL
    # ================================
    st.markdown("""
    <div class='hero-header'>
        <div class='brand-title'>🔍 AURORA ETHICS</div>
        <p class='quote-text'>"La transparencia es el mejor antídoto contra la corrupción"</p>
        <p class='author-text'>Marco Antonio Sánchez Cedillo & Saul Avila Hernández</p>
        <p class='signature-text'>Datatón Anticorrupción México 2025 - SESNA</p>
    </div>
    """, unsafe_allow_html=True)

    # ================================
    # Selector de dataset (solo si hay datasets CSV/Excel cargados)
    # ================================
    if st.session_state.datasets:
        if st.session_state.dataset_consolidado is not None:
            opciones_dataset = ['Dataset Consolidado'] + list(st.session_state.datasets.keys())
        else:
            opciones_dataset = list(st.session_state.datasets.keys())
        
        dataset_sel = st.selectbox(
            "📁 Seleccione el dataset a analizar:",
            opciones_dataset,
            key="dataset_main_selector"
        )
        
        if dataset_sel != 'Dataset Consolidado':
            st.session_state.dataset_seleccionado = dataset_sel
        
        st.markdown("---")

    # ================================
    # 🌟 MOSTRAR SIEMPRE TODOS LOS TABS (nuevos + originales)
    # ================================
    
    # Tabs de extracción (JSON/ZIP)
    extraction_tabs = st.tabs([
        "📤 Cargar Archivos",
        "📊 Vista de Datos",
        "➕ Campo Personalizado",
        "📥 Descargar Excel"
    ])
    
    with extraction_tabs[0]:
        st.markdown("## 📤 Carga de Archivos")
        
        st.markdown("""
        <div class="info-box">
        <h4>📦 Formatos Soportados</h4>
        <p>Puedes cargar archivos en dos formatos:</p>
        <ul>
            <li>✅ <strong>Archivo ZIP</strong> con múltiples JSON (recomendado para grandes volúmenes)</li>
            <li>✅ <strong>Archivos JSON individuales</strong> (uno o varios a la vez)</li>
        </ul>
        <p>El sistema filtra <strong>244 columnas específicas</strong> (archivo origen + Estado + 242 PDN) más <strong>8 métricas calculadas</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        tipo_carga = st.radio(
            "Selecciona el tipo de archivo:",
            ["📦 Archivo ZIP", "📄 Archivos JSON"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if tipo_carga == "📦 Archivo ZIP":
            st.markdown("### 📂 Selecciona tu archivo ZIP")
            
            uploaded_zip = st.file_uploader(
                "Arrastra un archivo ZIP aquí o haz clic para seleccionar",
                type=['zip'],
                accept_multiple_files=False,
                help="Archivo ZIP con múltiples JSON de declaraciones PDN",
                key="uploader_zip"
            )
            
            if uploaded_zip is not None:
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📦 Archivo", uploaded_zip.name)
                
                with col2:
                    size_mb = uploaded_zip.size / (1024 * 1024)
                    st.metric("💾 Tamaño", f"{size_mb:.2f} MB")
                
                with col3:
                    st.metric("📊 Estado", "✅ Listo")
                
                st.markdown("---")
                
                col_analizar = st.columns([1, 2, 1])
                
                with col_analizar[1]:
                    if st.button("🔍 ANALIZAR CONTENIDO DEL ZIP", type="secondary", use_container_width=True, key="btn_analizar_zip"):
                        
                        with st.spinner("Analizando contenido..."):
                            archivos_json = extraer_archivos_json_del_zip(uploaded_zip)
                            
                            if archivos_json:
                                st.success(f"✅ Se encontraron {len(archivos_json)} archivos JSON en el ZIP")
                                
                                with st.expander("📋 Ver archivos encontrados", expanded=False):
                                    for i, archivo in enumerate(archivos_json[:50], 1):
                                        st.write(f"{i}. {archivo['nombre']}")
                                    
                                    if len(archivos_json) > 50:
                                        st.info(f"... y {len(archivos_json) - 50} archivos más")
                            else:
                                st.error("❌ No se encontraron archivos JSON en el ZIP")
                
                st.markdown("---")
                
                col_procesar = st.columns([1, 2, 1])
                
                with col_procesar[1]:
                    if st.button("🚀 PROCESAR Y CONSOLIDAR DESDE ZIP", type="primary", use_container_width=True, key="btn_procesar_zip"):
                        
                        st.markdown("""
                        <div style="text-align: center; margin: 1rem 0;">
                            <span class="processing-indicator">⏳ Procesando archivos JSON...</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        tiempo_inicio = time.time()
                        
                        status_text.text("📦 Extrayendo archivos del ZIP...")
                        archivos_json = extraer_archivos_json_del_zip(uploaded_zip)
                        
                        if archivos_json:
                            df_resultado = procesar_archivos_desde_zip(
                                archivos_json,
                                progress_bar,
                                status_text
                            )
                            
                            if df_resultado is not None and not df_resultado.empty:
                                
                                st.session_state.df_consolidado = df_resultado
                                
                                tiempo_total = time.time() - tiempo_inicio
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.success("✅ ¡Procesamiento completado exitosamente!")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("📦 Archivos", st.session_state.archivos_procesados)
                                
                                with col2:
                                    st.metric("📊 Registros", f"{len(df_resultado):,}")
                                
                                with col3:
                                    st.metric("📋 Columnas", len(df_resultado.columns))
                                
                                with col4:
                                    st.metric("⏱️ Tiempo", f"{tiempo_total:.2f} seg")
                                
                                if st.session_state.archivos_con_error:
                                    st.warning(f"⚠️ {len(st.session_state.archivos_con_error)} archivos con errores")
                                    with st.expander("👁️ Ver errores"):
                                        for error in st.session_state.archivos_con_error[:10]:
                                            st.write(f"❌ {error['archivo']}: {error['error']}")
                                
                                st.markdown("### 📊 Vista Previa (primeras 20 filas)")
                                st.dataframe(df_resultado.head(20), use_container_width=True)
                                
                                st.info("👉 Ve a las siguientes pestañas para agregar campo personalizado y descargar Excel")
                                st.balloons()
                            
                            else:
                                st.error("❌ No se pudieron procesar los archivos JSON")
                        else:
                            st.error("❌ No se encontraron archivos JSON en el ZIP")
            
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.3);">
                    <h2 style="color: #667eea;">📦 Selecciona un archivo ZIP</h2>
                    <p style="color: #764ba2;">Arrastra el archivo aquí o haz clic en "Browse files"</p>
                    <p style="color: #a0aec0;">Soporta archivos de hasta +2GB con miles de JSON</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("### 📄 Selecciona tus archivos JSON")
            
            uploaded_jsons = st.file_uploader(
                "Arrastra archivos JSON aquí o haz clic para seleccionar (puedes seleccionar múltiples)",
                type=['json'],
                accept_multiple_files=True,
                help="Uno o varios archivos JSON de declaraciones PDN",
                key="uploader_json"
            )
            
            if uploaded_jsons:
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Archivos", len(uploaded_jsons))
                
                with col2:
                    total_size = sum([f.size for f in uploaded_jsons]) / (1024 * 1024)
                    st.metric("💾 Tamaño Total", f"{total_size:.2f} MB")
                
                with col3:
                    st.metric("📊 Estado", "✅ Listo")
                
                st.markdown("---")
                
                with st.expander("📋 Ver archivos cargados", expanded=False):
                    for i, f in enumerate(uploaded_jsons, 1):
                        size_mb = f.size / (1024 * 1024)
                        st.write(f"{i}. {f.name} ({size_mb:.2f} MB)")
                
                st.markdown("---")
                
                col_procesar = st.columns([1, 2, 1])
                
                with col_procesar[1]:
                    if st.button("🚀 PROCESAR Y CONSOLIDAR JSON", type="primary", use_container_width=True, key="btn_procesar_json"):
                        
                        st.markdown("""
                        <div style="text-align: center; margin: 1rem 0;">
                            <span class="processing-indicator">⏳ Procesando archivos JSON...</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        tiempo_inicio = time.time()
                        
                        df_resultado = procesar_archivos_json_individuales(
                            uploaded_jsons,
                            progress_bar,
                            status_text
                        )
                        
                        if df_resultado is not None and not df_resultado.empty:
                            
                            st.session_state.df_consolidado = df_resultado
                            
                            tiempo_total = time.time() - tiempo_inicio
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("✅ ¡Procesamiento completado exitosamente!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("📄 Archivos", st.session_state.archivos_procesados)
                            
                            with col2:
                                st.metric("📊 Registros", f"{len(df_resultado):,}")
                            
                            with col3:
                                st.metric("📋 Columnas", len(df_resultado.columns))
                            
                            with col4:
                                st.metric("⏱️ Tiempo", f"{tiempo_total:.2f} seg")
                            
                            if st.session_state.archivos_con_error:
                                st.warning(f"⚠️ {len(st.session_state.archivos_con_error)} archivos con errores")
                                with st.expander("👁️ Ver errores"):
                                    for error in st.session_state.archivos_con_error[:10]:
                                        st.write(f"❌ {error['archivo']}: {error['error']}")
                            
                            st.markdown("### 📊 Vista Previa (primeras 20 filas)")
                            st.dataframe(df_resultado.head(20), use_container_width=True)
                            
                            st.info("👉 Ve a las siguientes pestañas para agregar campo personalizado y descargar Excel")
                            st.balloons()
                        
                        else:
                            st.error("❌ No se pudieron procesar los archivos JSON")
            
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.3);">
                    <h2 style="color: #667eea;">📄 Selecciona archivos JSON</h2>
                    <p style="color: #764ba2;">Arrastra los archivos aquí o haz clic en "Browse files"</p>
                    <p style="color: #a0aec0;">Puedes seleccionar múltiples archivos JSON</p>
                </div>
                """, unsafe_allow_html=True)

    with extraction_tabs[1]:
        st.markdown("## 📊 Vista de Datos Consolidados")
        
        if st.session_state.df_consolidado is not None and not st.session_state.df_consolidado.empty:
            
            df_mostrar = st.session_state.df_consolidado.head(100)
            
            st.markdown("### 📋 Dataset Consolidado (primeras 100 filas)")
            st.dataframe(df_mostrar, use_container_width=True, height=400)
            
            with st.expander("📋 Ver información detallada de columnas"):
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
            <p>Ve a la pestaña "📤 Cargar Archivos" para procesar archivos.</p>
            </div>
            """, unsafe_allow_html=True)

    with extraction_tabs[2]:
        st.markdown("## ➕ Agregar Campo Personalizado")
        
        if st.session_state.df_consolidado is not None and not st.session_state.df_consolidado.empty:
            
            st.markdown("""
            <div class="info-box">
            <h4>🏷️ Campo Personalizado</h4>
            <p>Agrega una columna adicional con un valor constante a todos los registros.</p>
            <p><strong>Ejemplo:</strong> Agregar columna "Estado" con valor "Michoacán"</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                nombre_campo = st.text_input(
                    "Nombre del campo:",
                    value="Estado",
                    help="Nombre de la nueva columna"
                )
            
            with col2:
                valor_campo = st.text_input(
                    "Valor del campo:",
                    value="",
                    help="Valor que se asignará a todos los registros"
                )
            
            st.markdown("---")
            
            if st.button("➕ AGREGAR CAMPO", type="primary", use_container_width=True):
                
                if nombre_campo and valor_campo:
                    
                    df_con_campo = agregar_campo_personalizado(
                        st.session_state.df_consolidado,
                        nombre_campo,
                        valor_campo
                    )
                    
                    st.session_state.df_consolidado = df_con_campo
                    st.session_state.campo_agregado = True
                    st.session_state.campo_personalizado_nombre = nombre_campo
                    st.session_state.campo_personalizado_valor = valor_campo
                    
                    st.success(f"✅ Campo '{nombre_campo}' agregado exitosamente con valor '{valor_campo}'")
                    
                    st.markdown("### 📊 Vista Previa")
                    st.dataframe(df_con_campo.head(10), use_container_width=True)
                
                else:
                    st.error("❌ Debes ingresar nombre y valor del campo")
            
            st.markdown("---")
            
            if st.session_state.campo_agregado:
                st.markdown("### 📊 Campos Personalizados Agregados")
                
                st.markdown(f"""
                <div class="success-box">
                <p>✅ <strong>Columna:</strong> {st.session_state.campo_personalizado_nombre}</p>
                <p>✅ <strong>Valor:</strong> {st.session_state.campo_personalizado_valor}</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="warning-box">
            <h3>📭 No hay datos cargados</h3>
            <p>Primero debes procesar archivos en la pestaña "📤 Cargar Archivos".</p>
            </div>
            """, unsafe_allow_html=True)

    with extraction_tabs[3]:
        st.markdown("## 📥 Descargar Datos en Excel")
        
        if st.session_state.df_consolidado is not None and not st.session_state.df_consolidado.empty:
            
            df_descarga = st.session_state.df_consolidado.copy()
            
            with st.spinner("🧮 Calculando métricas patrimoniales..."):
                df_descarga = calcular_metricas_patrimoniales(df_descarga)
            
            with st.spinner("🔍 Filtrando columnas requeridas..."):
                df_descarga = filtrar_columnas_requeridas(df_descarga)
            
            st.markdown("""
            <div class="success-box">
            <h4>✅ Datos listos para descargar</h4>
            <p>Tu dataset consolidado está listo para exportarse en formato Excel (.xlsx)</p>
            <p><strong>📋 Columnas filtradas: 244 columnas específicas + 8 métricas calculadas</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Resumen del Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Registros", f"{len(df_descarga):,}")
            
            with col2:
                st.metric("📋 Columnas", len(df_descarga.columns))
            
            with col3:
                memoria_mb = df_descarga.memory_usage(deep=True).sum() / 1024**2
                st.metric("💾 Tamaño Aprox.", f"{memoria_mb:.2f} MB")
            
            with col4:
                if st.session_state.campo_agregado:
                    st.metric("🏷️ Campo Agregado", "✅ Sí")
                else:
                    st.metric("🏷️ Campo Agregado", "❌ No")
            
            st.markdown("---")
            
            if st.session_state.campo_agregado:
                st.markdown(f"### 📋 Columnas Incluidas ({len(df_descarga.columns)} totales - incluye campo personalizado '{st.session_state.campo_personalizado_nombre}')")
            else:
                st.markdown(f"### 📋 Columnas Incluidas ({len(df_descarga.columns)} totales)")
            
            with st.expander("👁️ Ver lista completa de columnas", expanded=False):
                for i, col in enumerate(df_descarga.columns, 1):
                    if col in ['Patrimonio_Total', 'Ratio_Patrimonio_Ingreso', 'Delta_Patrimonial_Anual', 
                              'Incremento_Porcentual_Patrimonio', 'Años_en_Cargo', 'Tiene_Conflicto_Interes',
                              'Numero_Bienes_Totales', 'Ratio_Deuda_Activos']:
                        st.markdown(f"**{i}. {col}** 🧮 *(Calculada)*")
                    elif col == '_nombre_archivo_origen':
                        st.markdown(f"**{i}. {col}** 📂 *(Archivo origen)*")
                    elif st.session_state.campo_agregado and col == st.session_state.campo_personalizado_nombre:
                        st.markdown(f"**{i}. {col}** 🏷️ *(Campo personalizado: {st.session_state.campo_personalizado_valor})*")
                    else:
                        st.markdown(f"{i}. {col}")
            
            st.markdown("---")
            
            st.markdown("### 📥 Generar y Descargar Excel")
            
            col_download = st.columns([1, 2, 1])
            
            with col_download[1]:
                
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_descarga.to_excel(writer, index=False, sheet_name='Datos PDN')
                
                buffer.seek(0)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                nombre_archivo = f"AURORA_PDN_FILTRADO_{timestamp}.xlsx"
                
                st.download_button(
                    label=f"⬇️ DESCARGAR EXCEL ({len(df_descarga.columns)} COLUMNAS)",
                    data=buffer,
                    file_name=nombre_archivo,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type="primary",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            st.markdown("### 📊 Vista Previa del Excel")
            st.dataframe(df_descarga.head(20), use_container_width=True)
            
            if st.session_state.campo_agregado:
                st.info(f"💡 El archivo Excel incluye: archivo origen + **{st.session_state.campo_personalizado_nombre}** (campo personalizado) + 242 columnas PDN + 8 métricas = {len(df_descarga.columns)} columnas totales")
            else:
                st.info(f"💡 El archivo Excel incluye: archivo origen + 242 columnas PDN + 8 métricas = {len(df_descarga.columns)} columnas totales")
        
        else:
            st.markdown("""
            <div class="warning-box">
            <h3>📭 No hay datos para descargar</h3>
            <p>Primero debes procesar archivos en la pestaña "📤 Cargar Archivos".</p>
            </div>
            """, unsafe_allow_html=True)

    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "🏠 Inicio",
        "🔍 EDA",
        "📈 Visualización Global",
        "🎨 Personalizado",
        "🤖 Machine Learning",
        "🌍 Geo Inteligencia",
        "💰 Evolución Patrimonial",
        "🕸 Nepotismo",
        "📢 Denuncias",
        "📋 Diccionario",
        "📄 Reporte Ejecutivo",
        "🗑️ Borrar Datos"
    ])
    
    with tab1:
        seccion_inicio()
    with tab2:
        seccion_eda()
    with tab3:
        seccion_visualizacion_global()
    with tab4:
        seccion_visualizacion_personalizada()
    with tab5:
        seccion_machine_learning()
    with tab6:
        seccion_geo_inteligencia()
    with tab7:
        seccion_evolucion_patrimonial()
    with tab8:
        seccion_nepotismo()
    with tab9:
        seccion_denuncias()
    with tab10:
        seccion_diccionario_datos()
    with tab11:
        seccion_reporte_ejecutivo()
    with tab12:
        seccion_borrar_datos()

    # ================================
    # Mensaje de bienvenida (opcional, solo si no hay nada cargado)
    # ================================
    if not st.session_state.datasets and (not hasattr(st.session_state, 'df_consolidado') or st.session_state.df_consolidado is None):
        st.markdown("---")
        st.info("👈 **Por favor, cargue archivos CSV/Excel desde el panel lateral o use la pestaña '📤 Cargar Archivos' para procesar JSON/ZIP**")
if __name__ == "__main__":
    main()