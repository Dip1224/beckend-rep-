"""Servicio de asistencia con soporte para Supabase o almacenamiento local."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

from services.supabase_service import SupabaseService


class AsistenciaService:
    def __init__(self, supabase_service: Optional[SupabaseService] = None) -> None:
        self.supabase = supabase_service
        self.archivo_asistencia = "backend/data/asistencia.json"
        self.usa_supabase = bool(self.supabase and self.supabase.enabled)
        self.asistencia_cache = {} if self.usa_supabase else self._cargar_asistencia_local()

    # ------------------------------------------------------------------ Local helpers
    def _cargar_asistencia_local(self) -> Dict:
        if os.path.exists(self.archivo_asistencia):
            with open(self.archivo_asistencia, "r", encoding="utf-8") as handler:
                return json.load(handler)
        return {}

    def _guardar_asistencia_local(self) -> None:
        os.makedirs(os.path.dirname(self.archivo_asistencia), exist_ok=True)
        with open(self.archivo_asistencia, "w", encoding="utf-8") as handler:
            json.dump(self.asistencia_cache, handler, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ API publica
    def registrar_asistencia(self, nombre: str, tipo: str = "entrada"):
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        hora_actual = datetime.now().strftime("%H:%M:%S")

        if self.usa_supabase:
            self.supabase.guardar_asistencia(nombre, tipo, fecha_hoy, hora_actual)
        else:
            self._registrar_local(nombre, tipo, fecha_hoy, hora_actual)

        return {
            "exito": True,
            "nombre": nombre,
            "tipo": tipo,
            "hora": hora_actual,
            "fecha": fecha_hoy,
        }

    def _registrar_local(self, nombre: str, tipo: str, fecha: str, hora: str) -> None:
        if fecha not in self.asistencia_cache:
            self.asistencia_cache[fecha] = {}
        if nombre not in self.asistencia_cache[fecha]:
            self.asistencia_cache[fecha][nombre] = []
        self.asistencia_cache[fecha][nombre].append({"tipo": tipo, "hora": hora})
        self._guardar_asistencia_local()

    def obtener_asistencia_hoy(self):
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        if self.usa_supabase:
            return self.supabase.obtener_asistencia_por_fecha(fecha_hoy)
        return self.asistencia_cache.get(fecha_hoy, {})

    def obtener_asistencia_fecha(self, fecha: str):
        if self.usa_supabase:
            return self.supabase.obtener_asistencia_por_fecha(fecha)
        return self.asistencia_cache.get(fecha, {})

    def obtener_asistencia_persona(self, nombre: str, fecha: Optional[str] = None):
        fecha = fecha or datetime.now().strftime("%Y-%m-%d")
        registros = self.obtener_asistencia_fecha(fecha)
        return registros.get(nombre, [])

    def obtener_estadisticas_hoy(self):
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        registros = self.obtener_asistencia_hoy()
        estadisticas = self._calcular_estadisticas(registros)
        estadisticas["fecha"] = fecha_hoy
        return estadisticas

    def obtener_historial_completo(self):
        if self.usa_supabase:
            return self.supabase.obtener_historial()
        return self.asistencia_cache

    def verificar_puede_registrar(self, nombre: str, tiempo_espera: int = 10) -> Tuple[bool, Optional[str]]:
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")

        if self.usa_supabase:
            ultimo = self.supabase.obtener_ultimo_registro(nombre, fecha_hoy)
            return self._evaluar_registro(nombre, ultimo, tiempo_espera)

        registros = self.obtener_asistencia_hoy()
        historial = registros.get(nombre, [])
        ultimo = historial[-1] if historial else None
        return self._evaluar_registro(nombre, ultimo, tiempo_espera)

    # ------------------------------------------------------------------ Utilidades
    def _calcular_estadisticas(self, registros: Dict) -> Dict:
        total_personas = len(registros)
        con_entrada = 0
        con_salida = 0
        presentes = 0

        for eventos in registros.values():
            tiene_entrada = any(r["tipo"] == "entrada" for r in eventos)
            tiene_salida = any(r["tipo"] == "salida" for r in eventos)
            if tiene_entrada:
                con_entrada += 1
            if tiene_salida:
                con_salida += 1
            if tiene_entrada and not tiene_salida:
                presentes += 1

        return {
            "total_personas": total_personas,
            "con_entrada": con_entrada,
            "con_salida": con_salida,
            "presentes": presentes,
        }

    def _evaluar_registro(self, nombre: str, ultimo, tiempo_espera: int) -> Tuple[bool, Optional[str]]:
        if not ultimo:
            return True, "entrada"

        hora_actual = datetime.now()
        hora_ultimo = datetime.strptime(ultimo["hora"], "%H:%M:%S").time()
        ultimo_dt = datetime.combine(datetime.today(), hora_ultimo)
        diff_segundos = (hora_actual - ultimo_dt).seconds

        if diff_segundos < tiempo_espera:
            return False, None

        tipo_siguiente = "salida" if ultimo["tipo"] == "entrada" else "entrada"
        return True, tipo_siguiente
