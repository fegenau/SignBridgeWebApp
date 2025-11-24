"""
Página de Diccionario de Señas - Visualización de todas las señas disponibles
"""

import streamlit as st
import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_classes, get_sign_type

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

# Ruta al logo
logo_path = Path(__file__).parent.parent / "assets" / "Imagenes" / "Logo" / "IconSignBridge.png"

st.set_page_config(
    page_title="SignBridge - Diccionario",
    page_icon=str(logo_path) if logo_path.exists() else "📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ESTILOS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    .sign-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .sign-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .sign-label {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 1rem 0;
    }
    
    .sign-type {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    .type-static {
        background: #4CAF50;
        color: white;
    }
    
    .type-dynamic {
        background: #FF9800;
        color: white;
    }
    
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
    }
    
    .placeholder-image {
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        color: white;
        font-size: 4rem;
        object-fit: contain;
    }
    
    .sign-image {
        width: 200px;
        height: 200px;
        object-fit: contain;
        border-radius: 10px;
        margin: 0 auto;
        display: block;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================

def main():
    # Botón de volver al inicio
    if st.button("⬅️ Volver al Inicio"):
        st.switch_page("app.py")
    
    # Título
    st.title("📚 Diccionario de Señas")
    st.markdown("Explora todas las señas disponibles en el sistema")
    
    # Cargar clases
    try:
        classes = load_classes()
        
        # Organizar por categorías
        numbers = sorted([c for c in classes if c.isdigit() and c != '0'])  # Excluir 0, mostrar 1-9
        # Agregar 10 si existe en las clases
        if '10' in classes:
            numbers.append('10')
        letters = sorted([c for c in classes if len(c) == 1 and c.isalpha()])
        phrases = sorted([c for c in classes if c not in numbers and c not in letters and c != '0'])
        
        # Estadísticas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Señas", len(classes))
        with col2:
            st.metric("Números", len(numbers))
        with col3:
            st.metric("Letras", len(letters))
        with col4:
            st.metric("Frases", len(phrases))
        
        st.markdown("---")
        
        # Filtros
        st.markdown("### 🔍 Filtros")
        col1, col2 = st.columns(2)
        
        with col1:
            category_filter = st.selectbox(
                "Categoría",
                ["Todas", "Números", "Letras", "Frases"],
                index=0
            )
        
        with col2:
            type_filter = st.selectbox(
                "Tipo de Seña",
                ["Todas", "Estáticas", "Dinámicas"],
                index=0
            )
        
        # Buscador
        search = st.text_input("🔎 Buscar seña", placeholder="Escribe para buscar...")
        
        st.markdown("---")
        
        # 🆕 TRADUCTOR DE TEXTO A SEÑAS
        st.markdown("## 🔤 Traductor de Texto a Señas")
        st.markdown("Escribe una palabra o frase y ve su traducción en lenguaje de señas")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            text_input = st.text_input(
                "Escribe tu mensaje",
                placeholder="Ej: HOLA",
                max_chars=50,
                key="translator_input"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            translate_btn = st.button("🔄 Traducir", type="primary", use_container_width=True)
        
        # Mostrar traducción
        if translate_btn and text_input:
            # Convertir a mayúsculas y limpiar
            text_clean = text_input.upper().strip()
            
            # Obtener ruta de imágenes
            assets_dir = Path(__file__).parent.parent / "assets" / "Imagenes" / "Diccionario"
            
            # Filtrar solo letras y números que tengan imagen
            valid_chars = []
            missing_chars = []
            
            for char in text_clean:
                if char == ' ':
                    valid_chars.append('SPACE')
                elif char.isalnum():  # Letras y números
                    # Buscar imagen
                    image_found = False
                    for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                        img_path = assets_dir / f"{char}{ext}"
                        if img_path.exists():
                            valid_chars.append((char, str(img_path)))
                            image_found = True
                            break
                    if not image_found:
                        missing_chars.append(char)
            
            # Mostrar traducción
            if valid_chars:
                st.markdown("### 👉 Traducción en Lenguaje de Señas:")
                
                # Usar columnas de Streamlit para mostrar imágenes
                cols = st.columns(len(valid_chars))
                for idx, item in enumerate(valid_chars):
                    with cols[idx]:
                        if item == 'SPACE':
                            st.markdown('<div style="text-align: center; font-size: 24px; color: #667eea; padding: 20px;">-</div>', unsafe_allow_html=True)
                        else:
                            char, img_path = item
                            # Responsive: 40px móvil, 50px tablet, 60px desktop
                            st.image(img_path, use_container_width=True)
                
                # Mostrar caracteres faltantes
                if missing_chars:
                    st.warning(f"⚠️ No se encontraron imágenes para: {', '.join(missing_chars)}")
            else:
                st.info("👉 Escribe letras o números para ver su traducción")
        
        st.markdown("---")
        
        # Función para mostrar señas
        def display_signs(signs, title):
            if not signs:
                return
            
            st.markdown(f'<div class="category-header"><h2>{title}</h2></div>', unsafe_allow_html=True)
            
            # Mostrar en grid de 4 columnas
            cols_per_row = 4
            for i in range(0, len(signs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(signs):
                        sign = signs[i + j]
                        sign_type = get_sign_type(sign)
                        type_class = "type-static" if sign_type == "ESTÁTICA" else "type-dynamic"
                        
                        with col:
                            # Buscar imagen de la seña
                            image_path = None
                            assets_dir = Path(__file__).parent.parent / "assets" / "Imagenes" / "Diccionario"
                            
                            # Buscar imagen con diferentes extensiones
                            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                                potential_path = assets_dir / f"{sign}{ext}"
                                if potential_path.exists():
                                    image_path = potential_path
                                    break
                            
                            # Mostrar card con imagen o placeholder
                            if image_path:
                                # Mostrar imagen real
                                st.markdown(f"""
                                <div class="sign-card">
                                    <div class="sign-label">{sign}</div>
                                    <span class="sign-type {type_class}">{sign_type}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                # Centrar imagen usando columnas
                                _, img_col, _ = st.columns([1, 2, 1])
                                with img_col:
                                    st.image(str(image_path), use_container_width=True)
                            else:
                                # Mostrar placeholder
                                st.markdown(f"""
                                <div class="sign-card">
                                    <div class="placeholder-image">
                                        {sign[0] if len(sign) == 1 else "✋"}
                                    </div>
                                    <div class="sign-label">{sign}</div>
                                    <span class="sign-type {type_class}">{sign_type}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
        
        # Filtrar señas
        def filter_signs(signs_list):
            filtered = signs_list.copy()
            
            # Filtrar por búsqueda
            if search:
                filtered = [s for s in filtered if search.lower() in s.lower()]
            
            # Filtrar por tipo
            if type_filter == "Estáticas":
                filtered = [s for s in filtered if get_sign_type(s) == "ESTÁTICA"]
            elif type_filter == "Dinámicas":
                filtered = [s for s in filtered if get_sign_type(s) == "MOVIMIENTO"]
            
            return filtered
        
        # Mostrar según categoría seleccionada
        if category_filter == "Todas" or category_filter == "Números":
            filtered_numbers = filter_signs(numbers)
            if filtered_numbers:
                display_signs(filtered_numbers, f"🔢 Números ({len(filtered_numbers)})")
        
        if category_filter == "Todas" or category_filter == "Letras":
            filtered_letters = filter_signs(letters)
            if filtered_letters:
                display_signs(filtered_letters, f"🔤 Letras ({len(filtered_letters)})")
        
        if category_filter == "Todas" or category_filter == "Frases":
            filtered_phrases = filter_signs(phrases)
            if filtered_phrases:
                display_signs(filtered_phrases, f"💬 Frases Comunes ({len(filtered_phrases)})")
        
        # Nota sobre imágenes
        st.markdown("---")
        st.info("""
        **📝 Instrucciones para agregar imágenes**: 
        
        1. Coloca las imágenes en: `WebApp/assets/Imagenes/Diccionario/`
        2. Nombra cada imagen exactamente como la seña (ej: `A.png`, `5.jpg`, `Hola.png`)
        3. Formatos soportados: PNG, JPG, JPEG, GIF, WEBP
        4. Las imágenes se mostrarán automáticamente al recargar la página
        """)
        
    except Exception as e:
        st.error(f"❌ Error al cargar las clases: {e}")

if __name__ == "__main__":
    main()
