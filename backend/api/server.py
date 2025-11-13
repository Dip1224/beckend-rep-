"""
API REST para el sistema de asistencia facial
Expone endpoints para frontend
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import cv2
import numpy as np
import base64
import os
import sys

# Cargar variables de entorno (.env opcional)
load_dotenv()

# Agregar path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.face_recognition_service import FaceRecognitionService
from services.asistencia_service import AsistenciaService
from services.supabase_service import SupabaseService

app = Flask(__name__, 
            static_folder='../../frontend/static',
            template_folder='../../frontend/templates')
CORS(app)

# Inicializar servicios (Supabase es opcional)
supabase_service = SupabaseService()
face_service = FaceRecognitionService(supabase_service)
asistencia_service = AsistenciaService(supabase_service)


def base64_to_image(base64_string):
    """Convierte base64 a imagen OpenCV"""
    # Remover prefijo si existe
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(image):
    """Convierte imagen OpenCV a base64"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


# ============================================================================
# ENDPOINTS - RECONOCIMIENTO FACIAL
# ============================================================================

@app.route('/api/detectar', methods=['POST'])
def detectar_rostros():
    """Detecta y reconoce rostros en una imagen"""
    try:
        data = request.json
        imagen_base64 = data.get('imagen')
        
        if not imagen_base64:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Convertir a imagen
        imagen = base64_to_image(imagen_base64)
        
        # Detectar rostros
        rostros = face_service.detectar_rostros(imagen)
        
        # Dibujar rectángulos
        for rostro in rostros:
            bbox = rostro['bbox']
            nombre = rostro.get('nombre', 'Desconocido')
            similitud = rostro.get('similitud', 0)
            
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            
            cv2.rectangle(imagen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            texto = f"{nombre} ({similitud:.0%})" if nombre != "Desconocido" else nombre
            cv2.putText(imagen, texto, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convertir imagen procesada a base64
        imagen_procesada = image_to_base64(imagen)
        
        # Limpiar embeddings (no enviar al frontend)
        for rostro in rostros:
            if 'embedding' in rostro:
                del rostro['embedding']
        
        return jsonify({
            "exito": True,
            "rostros": rostros,
            "total_rostros": len(rostros),
            "imagen_procesada": imagen_procesada
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/registrar', methods=['POST'])
def registrar_persona():
    """Registra una nueva persona"""
    try:
        data = request.json
        nombre = data.get('nombre')
        imagen_base64 = data.get('imagen')
        
        if not nombre or not imagen_base64:
            return jsonify({"error": "Faltan datos (nombre o imagen)"}), 400
        
        # Convertir a imagen
        imagen = base64_to_image(imagen_base64)
        
        # Registrar
        resultado = face_service.registrar_persona(nombre, imagen)
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/personas', methods=['GET'])
def listar_personas():
    """Lista todas las personas registradas"""
    try:
        personas = face_service.listar_personas()
        return jsonify({
            "exito": True,
            "personas": personas,
            "total": len(personas)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/personas/<nombre>', methods=['DELETE'])
def eliminar_persona(nombre):
    """Elimina una persona del sistema"""
    try:
        resultado = face_service.eliminar_persona(nombre)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENDPOINTS - ASISTENCIA
# ============================================================================

@app.route('/api/asistencia/registrar', methods=['POST'])
def registrar_asistencia():
    """Registra asistencia automáticamente desde detección"""
    try:
        data = request.json
        imagen_base64 = data.get('imagen')
        
        if not imagen_base64:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Convertir a imagen
        imagen = base64_to_image(imagen_base64)
        
        # Detectar rostros
        rostros = face_service.detectar_rostros(imagen)
        
        registros = []
        
        for rostro in rostros:
            nombre = rostro.get('nombre')
            
            if nombre and nombre != "Desconocido":
                # Verificar si puede registrar
                puede, tipo = asistencia_service.verificar_puede_registrar(nombre)
                
                if puede:
                    # Registrar asistencia
                    resultado = asistencia_service.registrar_asistencia(nombre, tipo)
                    registros.append(resultado)
        
        return jsonify({
            "exito": True,
            "registros": registros,
            "total_registrados": len(registros)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/asistencia/hoy', methods=['GET'])
def obtener_asistencia_hoy():
    """Obtiene la asistencia del día"""
    try:
        asistencia = asistencia_service.obtener_asistencia_hoy()
        estadisticas = asistencia_service.obtener_estadisticas_hoy()
        
        return jsonify({
            "exito": True,
            "asistencia": asistencia,
            "estadisticas": estadisticas
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/asistencia/fecha/<fecha>', methods=['GET'])
def obtener_asistencia_fecha(fecha):
    """Obtiene asistencia de una fecha específica"""
    try:
        asistencia = asistencia_service.obtener_asistencia_fecha(fecha)
        
        return jsonify({
            "exito": True,
            "fecha": fecha,
            "asistencia": asistencia
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/asistencia/historial', methods=['GET'])
def obtener_historial():
    """Obtiene todo el historial de asistencia"""
    try:
        historial = asistencia_service.obtener_historial_completo()
        
        return jsonify({
            "exito": True,
            "historial": historial
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENDPOINTS - FRONTEND
# ============================================================================

@app.route('/')
def index():
    """Página principal"""
    return send_from_directory('../../frontend/templates', 'index.html')


@app.route('/asistencia')
def asistencia():
    """Página de toma de asistencia"""
    return send_from_directory('../../frontend/templates', 'asistencia.html')


@app.route('/reportes')
def reportes():
    """Página de reportes"""
    return send_from_directory('../../frontend/templates', 'reportes.html')


# ============================================================================
# SERVIDOR
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 SERVIDOR API - SISTEMA DE ASISTENCIA FACIAL")
    print("=" * 60)
    print(f"✓ Personas registradas: {len(face_service.personas_registradas)}")
    print(f"✓ Servidor corriendo en: http://localhost:5000")
    print("=" * 60)
    print("\n📱 URLs disponibles:")
    print("   - http://localhost:5000/           (Dashboard principal)")
    print("   - http://localhost:5000/asistencia (Tomar asistencia)")
    print("   - http://localhost:5000/reportes   (Ver reportes)")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
