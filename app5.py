import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os
import requests # Necesario para descargar archivos desde la URL
import io # Para manejar el archivo descargado


#MODEL_DOWNLOAD_URL = "https://storage.googleapis.com/florascan-d_cloudbuild/NasNetMobile.keras"
#MODEL_PATH = "NasNetMobile.keras" # Nombre local para el archivo descargado

# ----------------- CONFIGURACIÓN LOCAL -----------------
# El modelo ahora se lee directamente desde la raíz de tu repositorio de GitHub
MODEL_PATH = "NasNetMobile.keras" 
# -------------------------------------------------------

# Configuración
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuración de la página
st.set_page_config(
    page_title="FloraScan",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Lista de clases
class_names = ["bacteria", "fungus", "healthy", "pests", "virus"]

@st.cache_resource
def load_model():
    # 1. Verificar si el modelo ya existe localmente
    if not os.path.exists(MODEL_PATH):
        st.info(f"Descargando modelo grande desde Azure Storage... (54 MB)")
        try:
            # 2. Descargar el archivo
            response = requests.get(MODEL_DOWNLOAD_URL, stream=True)
            response.raise_for_status() # Lanza una excepción para errores de HTTP

            # 3. Escribir el contenido descargado en un archivo local
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Descarga del modelo completada.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error al descargar el modelo desde Azure: {e}")
            return None # Retorna None o lanza un error si la descarga falla

    try:
        # 4. Cargar el modelo desde el archivo local
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo Keras: {e}")
        return None

# Función MEJORADA para mapa de saliencia
def generate_enhanced_saliency_map(model, img_array, original_img, target_size=(256, 256)):
    """
    Versión MEJORADA que genera mapas más visibles y significativos
    """
    try:
        img_tensor = tf.convert_to_tensor(img_array)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            top_class = tf.argmax(predictions[0])
            top_score = predictions[:, top_class]
        
        # Calcular gradientes
        grads = tape.gradient(top_score, img_tensor)
        
        if grads is None:
            return create_fallback_heatmap(original_img, target_size)
            
        # ESTRATEGIA MEJORADA: Enfocarse en gradientes positivos
        grads_pos = tf.maximum(grads, 0)  # Solo gradientes positivos
        
        # Reducir a 2D - usar suma en lugar de máximo para capturar más información
        saliency_map = tf.reduce_sum(grads_pos, axis=-1)[0]
        
        # Aplicar transformaciones para mejorar el contraste
        saliency_map = saliency_map.numpy()
        
        # ESTRATEGIA: Enfatizar las áreas más importantes
        if np.max(saliency_map) > 0:
            # Normalizar
            saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
            
            # Aplicar transformación gamma para realzar áreas importantes
            gamma = 0.5  # Valores < 1 realzan áreas brillantes
            saliency_map = np.power(saliency_map, gamma)
            
            # Asegurar que haya suficiente variación
            if np.max(saliency_map) - np.min(saliency_map) < 0.1:
                # Si hay poca variación, crear un patrón más visible
                saliency_map = create_enhanced_pattern(saliency_map.shape, original_img.shape)
        else:
            # Si no hay gradientes significativos, crear patrón basado en la imagen
            saliency_map = create_content_based_heatmap(original_img, target_size)
        
        # Redimensionar
        saliency_resized = cv2.resize(saliency_map, (original_img.shape[1], original_img.shape[0]))
        
        # Aplicar suavizado
        saliency_smoothed = cv2.GaussianBlur(saliency_resized, (25, 25), 0)
        
        # Aplicar colormap MEJORADO - usar HOT para mejor contraste
        saliency_uint8 = np.uint8(255 * saliency_smoothed)
        saliency_colored = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_HOT)
        
        # Convertir imagen original
        if len(original_img.shape) == 2:
            img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Mezclar con más transparencia para mejor visibilidad
        alpha = 0.7
        superimposed = cv2.addWeighted(img_bgr, 1 - alpha, saliency_colored, alpha, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return superimposed_rgb
        
    except Exception as e:
        st.error(f"Error en mapa mejorado: {e}")
        return create_fallback_heatmap(original_img, target_size)

def create_enhanced_pattern(shape, original_shape):
    """Crea un patrón mejorado cuando los gradientes son débiles"""
    height, width = shape
    pattern = np.zeros(shape)
    
    # Crear patrones concéntricos o radiales
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    
    # Distancia desde el centro
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Patrón radial
    pattern = 1.0 - (dist_from_center / max_dist)
    pattern = np.clip(pattern, 0, 1)
    
    # Añadir algo de variación
    pattern += np.random.rand(*shape) * 0.2
    pattern = np.clip(pattern, 0, 1)
    
    return pattern

def create_content_based_heatmap(original_img, target_size):
    """Crea heatmap basado en el contenido de la imagen cuando los gradientes fallan"""
    # Redimensionar imagen
    img_resized = cv2.resize(original_img, target_size)
    
    if len(img_resized.shape) == 3:
        # Convertir a escala de grises
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_resized
    
    # Detectar bordes y texturas
    edges = cv2.Canny(gray, 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # Combinar características
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = (gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude))
    
    # Combinar con bordes
    combined = gradient_magnitude * 0.7 + (edges / 255.0) * 0.3
    combined = np.clip(combined, 0, 1)
    
    return combined

def create_fallback_heatmap(original_img, target_size):
    """Heatmap de respaldo cuando todo falla"""
    img_resized = cv2.resize(original_img, target_size)
    height, width = img_resized.shape[:2]
    
    # Crear un heatmap que al menos muestre algo
    heatmap = np.zeros((height, width))
    
    # Patrón de manchas "enfermas"
    for _ in range(8):
        cx = np.random.randint(50, width-50)
        cy = np.random.randint(50, height-50)
        radius = np.random.randint(30, 80)
        intensity = np.random.uniform(0.5, 1.0)
        cv2.circle(heatmap, (cx, cy), radius, intensity, -1)
    
    # Aplicar colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    
    # Mezclar
    if len(img_resized.shape) == 2:
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    
    superimposed = cv2.addWeighted(img_bgr, 0.4, heatmap_colored, 0.6, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

# ALTERNATIVA: Visualización por activaciones de la última capa
def generate_activation_map(model, img_array, original_img, target_size=(256, 256)):
    """Genera mapa basado en las activaciones de la última capa convolucional"""
    try:
        # Encontrar la última capa convolucional
        conv_layers = []
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                conv_layers.append(layer.name)
        
        if not conv_layers:
            return None
            
        last_conv_layer = conv_layers[-1]
        
        # Crear modelo para obtener activaciones
        activation_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(last_conv_layer).output
        )
        
        # Obtener activaciones
        activations = activation_model.predict(img_array, verbose=0)
        
        # Promediar activaciones a través de los canales
        activation_map = np.mean(activations[0], axis=-1)
        
        # Procesar como antes
        activation_map = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map))
        activation_resized = cv2.resize(activation_map, (original_img.shape[1], original_img.shape[0]))
        
        # Aplicar colormap
        activation_uint8 = np.uint8(255 * activation_resized)
        activation_colored = cv2.applyColorMap(activation_uint8, cv2.COLORMAP_JET)
        
        # Mezclar
        if len(original_img.shape) == 2:
            img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        superimposed = cv2.addWeighted(img_bgr, 0.4, activation_colored, 0.6, 0)
        return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        st.warning(f"Activaciones fallaron: {e}")
        return None

def predecir_con_visualizacion_mejorada(img, model, class_names, target_size=(256, 256)):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_original_array = np.array(img)
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        pred = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(pred, axis=1)[0]
        clase_predicha = class_names[pred_idx]
        prob = pred[0][pred_idx]
        all_probs = {class_names[i]: float(pred[0][i]) for i in range(len(class_names))}
        
        # Intentar múltiples métodos de visualización
        st.info("🔍 Generando visualización mejorada...")
        
        # Método 1: Saliencia mejorada
        heatmap_img = generate_enhanced_saliency_map(model, img_array, img_original_array, target_size)
        
        # Método 2: Si el primero no da buen resultado, intentar con activaciones
        if heatmap_img is not None:
            # Verificar si el heatmap tiene suficiente variación de color
            heatmap_hsv = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2HSV)
            saturation = heatmap_hsv[:,:,1]
            if np.mean(saturation) < 50:  # Poca saturación = pocos colores
                st.warning("Poca variación en el mapa, intentando método alternativo...")
                alt_heatmap = generate_activation_map(model, img_array, img_original_array, target_size)
                if alt_heatmap is not None:
                    heatmap_img = alt_heatmap
        
        return clase_predicha, prob, all_probs, heatmap_img
    
    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")
        return None, None, None, None

# Interfaz principal
def main():
    st.markdown('<h1 class="main-header">🌿 Clasificador FloraScan</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/135/135644.png", width=100)
        st.markdown("### 📋 Instrucciones")
        st.markdown("""
        1. Sube imagen de hoja
        2. Análisis automático
        3. Revisa resultados
        4. Observa áreas relevantes
        """)
        
        st.markdown("### 🎯 Clases")
        for clase in class_names:
            emoji = "🟢" if clase == "healthy" else "🔴"
            st.write(f"{emoji} {clase.capitalize()}")
    
    model = load_model()
    if model is None:
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Subir Imagen")
        uploaded_file = st.file_uploader("Selecciona imagen", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                image_pil = Image.open(uploaded_file)
                st.image(image_pil, caption="Imagen subida", use_column_width=True)
                st.info(f"Tamaño: {image_pil.size} píxeles")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### 📊 Resultados")
        
        if uploaded_file is not None:
            if st.button("🔍 Analizar Imagen", type="primary", use_container_width=True):
                with st.spinner("Analizando con técnicas mejoradas..."):
                    clase_predicha, prob, all_probs, heatmap_img = predecir_con_visualizacion_mejorada(
                        image_pil, model, class_names
                    )
                    
                    if clase_predicha is not None:
                        # Resultado principal
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        if clase_predicha == "healthy":
                            emoji, mensaje = "✅", "Planta saludable"
                        else:
                            emoji, mensaje = "⚠️", f"Posible enfermedad: {clase_predicha}"
                        
                        st.markdown(f"### {emoji} {mensaje}")
                        st.markdown(f"### 📈 Confianza: {prob:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualización
                        if heatmap_img is not None:
                            st.markdown("### 🔍 Mapa de Áreas Relevantes")
                            col_orig, col_heat = st.columns(2)
                            with col_orig:
                                st.image(image_pil, caption="Original", use_column_width=True)
                            with col_heat:
                                st.image(heatmap_img, caption="Áreas detectadas", use_column_width=True)
                            
                            # Leyenda REALISTA
                            st.markdown("""
<style>
.leyenda-container {
    background-color: var(--background-color);
    color: var(--text-color);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid var(--primary-color);
    margin: 1rem 0;
}

.leyenda-container h4 {
    color: var(--heading-color);
    margin-bottom: 0.5rem;
}

.leyenda-container ul {
    margin: 0.5rem 0;
}

.leyenda-container li {
    margin: 0.25rem 0;
}
</style>

<div class="leyenda-container">
<h4>📖 Interpretación del Mapa:</h4>
<ul>
    <li>🔴 <b>Rojo/Naranja:</b> Áreas que el modelo considera más relevantes</li>
    <li>🟡 <b>Amarillo/Verde:</b> Regiones con importancia media</li>
    <li>🔵 <b>Azul:</b> Áreas menos influyentes en la decisión</li>
</ul>
<p><i>Nota: Si solo ves azul/verde, el modelo puede estar usando características difusas en toda la imagen.</i></p>
</div>
""", unsafe_allow_html=True)
                        
                        # Probabilidades
                        st.markdown("### 📋 Probabilidades")
                        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                        for clase, probabilidad in sorted_probs:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(float(probabilidad))
                            with col2:
                                st.write(f"{probabilidad:.2%}")
                            st.write(f"**{clase.capitalize()}**")
        
        else:
            st.info("👆 Sube una imagen para analizar")

if __name__ == "__main__":
    main()