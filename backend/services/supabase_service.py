"""Helper para centralizar toda la comunicacion con Supabase."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from supabase import Client, create_client


class SupabaseService:
    """Encapsula operaciones de lectura/escritura con Supabase."""

    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL")
        self.api_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "rostros")
        self.storage_prefix = os.getenv("SUPABASE_STORAGE_FOLDER", "personas")
        self.client: Optional[Client] = None

        if self.url and self.api_key:
            try:
                self.client = create_client(self.url, self.api_key)
                print("[Supabase] Conexion inicializada.")
            except Exception as e:
                print(f"[Supabase] Error al conectar: {e}")
                print("[Supabase] Usando modo local por error de conexion.")
                self.client = None
        else:
            print("[Supabase] Credenciales no encontradas. Se usara disco local.")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ Personas
    def cargar_personas_embeddings(self) -> Dict[str, Dict]:
        """Obtiene todas las personas registradas para uso en memoria."""
        if not self.enabled:
            return {}

        try:
            response = self.client.table("personas").select("*").execute()
            personas: Dict[str, Dict] = {}
            for row in response.data or []:
                vector = np.array(row.get("embedding") or [], dtype=np.float32)
                personas[row["nombre"]] = {
                    "embedding": vector,
                    "fecha_registro": row.get("fecha_registro"),
                    "edad": row.get("edad"),
                    "genero": row.get("genero"),
                    "foto_url": row.get("foto_url"),
                }
            return personas
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error al cargar personas: {exc}")
            return {}

    def guardar_persona(
        self,
        nombre: str,
        embedding: np.ndarray,
        edad: Optional[int],
        genero: Optional[str],
        fecha_registro: str,
        imagen_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Crea/actualiza el registro de una persona y sube la foto al bucket."""
        if not self.enabled:
            return None

        foto_url = None
        if imagen_bgr is not None:
            foto_url = self._subir_imagen(nombre, imagen_bgr)

        payload = {
            "nombre": nombre,
            "embedding": self._serialize_embedding(embedding),
            "edad": edad,
            "genero": genero,
            "fecha_registro": fecha_registro or datetime.utcnow().isoformat(),
        }
        if foto_url:
            payload["foto_url"] = foto_url

        try:
            self.client.table("personas").upsert(payload, on_conflict="nombre").execute()
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error guardando persona {nombre}: {exc}")

        return foto_url

    def listar_personas(self) -> List[Dict]:
        """Devuelve todas las personas listas para la API."""
        if not self.enabled:
            return []

        try:
            response = (
                self.client.table("personas")
                .select("nombre,fecha_registro,edad,genero,foto_url")
                .order("nombre")
                .execute()
            )
            personas = []
            for row in response.data or []:
                genero = row.get("genero")
                genero_texto = "Masculino" if genero == "M" else "Femenino" if genero == "F" else "N/A"
                personas.append(
                    {
                        "nombre": row["nombre"],
                        "fecha_registro": row.get("fecha_registro"),
                        "edad": row.get("edad"),
                        "genero": genero_texto,
                        "foto_url": row.get("foto_url"),
                    }
                )
            return personas
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error listando personas: {exc}")
            return []

    def eliminar_persona(self, nombre: str) -> bool:
        """Elimina una persona y su foto asociada."""
        if not self.enabled:
            return False

        try:
            self.client.table("personas").delete().eq("nombre", nombre).execute()
            self._eliminar_imagen(nombre)
            return True
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error eliminando persona {nombre}: {exc}")
            return False

    # ------------------------------------------------------------------ Asistencia
    def guardar_asistencia(self, nombre: str, tipo: str, fecha: str, hora: str) -> Dict:
        """Inserta un registro de asistencia."""
        if not self.enabled:
            return {}

        payload = {
            "persona_nombre": nombre,
            "tipo": tipo,
            "fecha": fecha,
            "hora": hora,
        }

        try:
            self.client.table("asistencias").insert(payload).execute()
            return payload
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error guardando asistencia de {nombre}: {exc}")
            return {}

    def obtener_asistencia_por_fecha(self, fecha: str) -> Dict[str, List[Dict]]:
        """Obtiene registro agrupado por persona para una fecha."""
        if not self.enabled:
            return {}

        try:
            response = (
                self.client.table("asistencias")
                .select("*")
                .eq("fecha", fecha)
                .order("hora", desc=False)
                .execute()
            )
            return self._group_by_person(response.data or [])
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error obteniendo asistencia del {fecha}: {exc}")
            return {}

    def obtener_historial(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Devuelve todo el historial agrupado por fecha y persona."""
        if not self.enabled:
            return {}

        try:
            response = self.client.table("asistencias").select("*").order("fecha").order("hora").execute()
            historial: Dict[str, Dict[str, List[Dict]]] = {}
            for row in response.data or []:
                fecha = row["fecha"]
                persona = row["persona_nombre"]
                historial.setdefault(fecha, {})
                historial[fecha].setdefault(persona, [])
                historial[fecha][persona].append({"tipo": row["tipo"], "hora": row["hora"]})
            return historial
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error obteniendo historial: {exc}")
            return {}

    def obtener_ultimo_registro(self, nombre: str, fecha: str) -> Optional[Dict]:
        """Devuelve el ultimo registro de una persona en una fecha."""
        if not self.enabled:
            return None

        try:
            response = (
                self.client.table("asistencias")
                .select("*")
                .eq("persona_nombre", nombre)
                .eq("fecha", fecha)
                .order("hora", desc=True)
                .limit(1)
                .execute()
            )
            data = response.data or []
            return data[0] if data else None
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error obteniendo ultimo registro de {nombre}: {exc}")
            return None

    # ------------------------------------------------------------------ Helpers
    def _serialize_embedding(self, embedding: np.ndarray) -> List[float]:
        vector = embedding.astype(float).tolist()
        return vector

    def _slugify(self, nombre: str) -> str:
        return nombre.strip().lower().replace(" ", "_")

    def _subir_imagen(self, nombre: str, imagen_bgr: np.ndarray) -> Optional[str]:
        if not self.enabled:
            return None

        success, buffer = cv2.imencode(".jpg", imagen_bgr)
        if not success:
            return None

        file_path = f"{self.storage_prefix}/{self._slugify(nombre)}.jpg"
        try:
            self.client.storage.from_(self.bucket).upload(
                file_path,
                buffer.tobytes(),
                {"content-type": "image/jpeg", "cache-control": "3600", "upsert": True},
            )
            public_url = self.client.storage.from_(self.bucket).get_public_url(file_path)
            return public_url
        except Exception as exc:  # pragma: no cover - logging
            print(f"[Supabase] Error subiendo imagen de {nombre}: {exc}")
            return None

    def _eliminar_imagen(self, nombre: str) -> None:
        if not self.enabled:
            return

        file_path = f"{self.storage_prefix}/{self._slugify(nombre)}.jpg"
        try:
            self.client.storage.from_(self.bucket).remove(file_path if isinstance(file_path, list) else [file_path])
        except Exception:
            # Ignorar si no existe
            pass

    def _group_by_person(self, rows: List[Dict]) -> Dict[str, List[Dict]]:
        agrupado: Dict[str, List[Dict]] = {}
        for row in rows:
            persona = row["persona_nombre"]
            agrupado.setdefault(persona, [])
            agrupado[persona].append({"tipo": row["tipo"], "hora": row["hora"]})
        return agrupado
