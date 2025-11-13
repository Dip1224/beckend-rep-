"""Servicio de reconocimiento facial con soporte para Supabase."""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from services.supabase_service import SupabaseService


class FaceRecognitionService:
    def __init__(self, supabase_service: Optional[SupabaseService] = None) -> None:
        print("Cargando modelo de reconocimiento facial...")
        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        self.supabase = supabase_service

        self.personas_db_path = "backend/data/personas_registradas.pkl"
        self.personas_registradas = self._sincronizar_personas()
        print(f"Modelo cargado. Personas registradas: {len(self.personas_registradas)}")

    # ------------------------------------------------------------------ Persistencia
    def _sincronizar_personas(self) -> Dict[str, Dict]:
        if self.supabase and self.supabase.enabled:
            return self.supabase.cargar_personas_embeddings()
        return self._cargar_personas_local()

    def _cargar_personas_local(self) -> Dict[str, Dict]:
        if os.path.exists(self.personas_db_path):
            with open(self.personas_db_path, "rb") as handler:
                return pickle.load(handler)
        return {}

    def _guardar_personas_local(self) -> None:
        os.makedirs(os.path.dirname(self.personas_db_path), exist_ok=True)
        with open(self.personas_db_path, "wb") as handler:
            pickle.dump(self.personas_registradas, handler)

    # ------------------------------------------------------------------ API publica
    def detectar_rostros(self, imagen):
        faces = self.app.get(imagen)
        resultados = []

        for face in faces:
            bbox = face.bbox.astype(int)
            genero_etiqueta = None
            if hasattr(face, "sex"):
                genero_etiqueta = "Masculino" if face.sex == "M" else "Femenino"

            resultado = {
                "bbox": bbox.tolist(),
                "edad": int(face.age) if hasattr(face, "age") else None,
                "genero": genero_etiqueta,
                "confianza": float(face.det_score) if hasattr(face, "det_score") else None,
                "embedding": face.normed_embedding,
            }

            nombre, similitud = self._reconocer_persona(face.normed_embedding)
            resultado["nombre"] = nombre or "Desconocido"
            resultado["similitud"] = float(similitud)
            resultados.append(resultado)

        return resultados

    def _reconocer_persona(self, embedding, umbral: float = 0.4):
        mejor_match = None
        mejor_similitud = 0.0

        for nombre, datos in self.personas_registradas.items():
            registrado = datos.get("embedding")
            if registrado is None or len(registrado) == 0:
                continue

            similitud = float(np.dot(embedding, registrado))
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_match = nombre

        if mejor_similitud >= umbral:
            return mejor_match, mejor_similitud
        return None, 0.0

    def registrar_persona(self, nombre: str, imagen):
        faces = self.app.get(imagen)

        if len(faces) == 0:
            return {"exito": False, "mensaje": "No se detecto ningun rostro en la imagen"}
        if len(faces) > 1:
            return {"exito": False, "mensaje": "Se detectaron multiples rostros. Usa una sola persona"}

        face = faces[0]
        persona_info = {
            "embedding": face.normed_embedding,
            "fecha_registro": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "edad": int(face.age) if hasattr(face, "age") else None,
            "genero": "M" if getattr(face, "sex", "M") == "M" else "F",
        }
        self.personas_registradas[nombre] = persona_info

        if self.supabase and self.supabase.enabled:
            self.supabase.guardar_persona(
                nombre=nombre,
                embedding=face.normed_embedding,
                edad=persona_info["edad"],
                genero=persona_info["genero"],
                fecha_registro=persona_info["fecha_registro"],
                imagen_bgr=imagen,
            )
        else:
            self._guardar_personas_local()
            self._guardar_foto_local(nombre, imagen)

        return {
            "exito": True,
            "mensaje": f"{nombre} registrado correctamente",
            "total_registrados": len(self.personas_registradas),
        }

    def listar_personas(self):
        if self.supabase and self.supabase.enabled:
            return self.supabase.listar_personas()

        personas = []
        for nombre, datos in self.personas_registradas.items():
            genero = datos.get("genero")
            personas.append(
                {
                    "nombre": nombre,
                    "fecha_registro": datos.get("fecha_registro"),
                    "edad": datos.get("edad"),
                    "genero": "Masculino" if genero == "M" else "Femenino",
                }
            )
        return personas

    def eliminar_persona(self, nombre: str):
        if nombre not in self.personas_registradas:
            return {"exito": False, "mensaje": f"{nombre} no encontrado"}

        del self.personas_registradas[nombre]
        if self.supabase and self.supabase.enabled:
            self.supabase.eliminar_persona(nombre)
        else:
            self._guardar_personas_local()
            self._eliminar_foto_local(nombre)

        return {"exito": True, "mensaje": f"{nombre} eliminado correctamente"}

    # ------------------------------------------------------------------ Utilidades locales
    def _guardar_foto_local(self, nombre: str, imagen) -> None:
        foto_path = f"backend/data/registros/{nombre.replace(' ', '_')}.jpg"
        os.makedirs(os.path.dirname(foto_path), exist_ok=True)
        cv2.imwrite(foto_path, imagen)

    def _eliminar_foto_local(self, nombre: str) -> None:
        foto_path = f"backend/data/registros/{nombre.replace(' ', '_')}.jpg"
        if os.path.exists(foto_path):
            os.remove(foto_path)
