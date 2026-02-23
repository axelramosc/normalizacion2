"""
MedNorm — API de Normalización de Medicamentos
FastAPI backend for Vercel Serverless
"""

import os
import re
import io
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from thefuzz import fuzz, process

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://abaazvcjwzvmwkgrtwee.supabase.co").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()

SCORE_THRESHOLD = 85  # >= 85 → mapeado_seguro, < 85 → revision_manual
BATCH_SIZE = 500      # Insert batch size for Vercel timeout safety

app = FastAPI(
    title="MedNorm API",
    description="Normalización de catálogos de medicamentos con CNIS",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("mednorm")

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
def get_sb() -> Client:
    if not SUPABASE_KEY:
        raise HTTPException(500, "SUPABASE_SERVICE_KEY no configurada")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Helpers — Limpieza de texto
# ---------------------------------------------------------------------------
_RE_CLEAN = re.compile(r"[^a-záéíóúñü0-9\s/\.\-\,]", re.IGNORECASE)
_RE_SPACES = re.compile(r"\s+")
_UNIT_MAP = {
    "mgs": "mg", "grs": "g", "gms": "g", "mcgs": "mcg",
    "mls": "ml", "lts": "l", "ui": "ui", "uis": "ui",
    "tabletas": "tab", "tableta": "tab", "tabs": "tab",
    "capsulas": "cap", "capsula": "cap", "caps": "cap",
    "ampolletas": "amp", "ampolleta": "amp", "amps": "amp",
    "frasco": "fco", "frascos": "fco",
    "solucion": "sol", "solución": "sol",
    "suspension": "susp", "suspensión": "susp",
    "inyectable": "iny", "inyección": "iny", "inyeccion": "iny",
}


def limpiar_descripcion(texto: str) -> str:
    """Normaliza una descripción de medicamento para matching."""
    if not texto:
        return ""
    t = texto.lower().strip()
    t = _RE_CLEAN.sub(" ", t)
    # Normalizar unidades
    for original, reemplazo in _UNIT_MAP.items():
        t = re.sub(rf"\b{original}\b", reemplazo, t)
    t = _RE_SPACES.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Gemini fallback (optional)
# ---------------------------------------------------------------------------
async def gemini_match(descripcion: str, candidatos: list[str]) -> Optional[dict]:
    """Uses Google Gemini as fallback for fuzzy matching when thefuzz score is low."""
    if not GOOGLE_API_KEY:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=GOOGLE_API_KEY)
        candidatos_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidatos[:20]))
        prompt = (
            f"Eres un experto farmacéutico. Dado el medicamento:\n"
            f"  \"{descripcion}\"\n\n"
            f"¿Cuál de estos candidatos del CNIS es el mismo medicamento?\n"
            f"{candidatos_text}\n\n"
            f"Responde SOLO con el número del candidato correcto (1-{min(len(candidatos), 20)}), "
            f"o '0' si ninguno coincide. Solo el número."
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = response.text.strip()
        match = re.search(r"\d+", answer)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(candidatos):
                return {"match": candidatos[idx], "score": 80, "method": "gemini"}
    except Exception as e:
        logger.warning(f"Gemini fallback error: {e}")
    return None


# ---------------------------------------------------------------------------
# POST /api/cargar_cnis
# ---------------------------------------------------------------------------
@app.post("/api/cargar_cnis")
async def cargar_cnis(file: UploadFile = File(...)):
    """Recibe Excel del CNIS e inserta masivamente en cnis_catalogo."""
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "Solo se aceptan archivos .xlsx, .xls o .csv")

    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Error al procesar el archivo: {str(e)}")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map common column name variants
    col_map = {}
    for col in df.columns:
        if "clave" in col:
            col_map[col] = "clave_cnis"
        elif "descripcion" in col or "presentacion" in col or "descrip" in col:
            if "clave_cnis" not in col_map.values():
                pass  # skip
            if "descripcion_cnis_limpia" not in col_map.values():
                col_map[col] = "descripcion_cnis_limpia"
        elif "grupo" in col or "terapeutico" in col:
            col_map[col] = "grupo_terapeutico"
        elif "indicacion" in col:
            col_map[col] = "indicaciones"
        elif "contraindicacion" in col:
            col_map[col] = "contraindicaciones"

    df = df.rename(columns=col_map)

    # Ensure required columns
    if "clave_cnis" not in df.columns:
        raise HTTPException(400, f"No se encontró columna 'clave'. Columnas: {list(df.columns)}")

    # Build description from available text columns if not directly mapped
    if "descripcion_cnis_limpia" not in df.columns:
        text_cols = [c for c in df.columns if c not in ("clave_cnis", "grupo_terapeutico", "indicaciones", "contraindicaciones")]
        if text_cols:
            df["descripcion_cnis_limpia"] = df[text_cols[0]].astype(str)
        else:
            raise HTTPException(400, "No se encontró columna de descripción")

    # Clean descriptions
    df["descripcion_cnis_limpia"] = df["descripcion_cnis_limpia"].apply(limpiar_descripcion)

    # Fill NaN
    for col in ["grupo_terapeutico", "indicaciones", "contraindicaciones"]:
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].where(df[col].notna(), None)

    # Deduplicate by clave_cnis
    df = df.drop_duplicates(subset=["clave_cnis"], keep="first")

    # Insert in batches
    sb = get_sb()

    # Clear existing data
    sb.table("cnis_catalogo").delete().neq("id_cnis", 0).execute()

    target_cols = ["clave_cnis", "descripcion_cnis_limpia", "grupo_terapeutico", "indicaciones", "contraindicaciones"]
    records = df[[c for c in target_cols if c in df.columns]].to_dict("records")

    # Clean None/NaN values
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v) if isinstance(v, float) else False:
                rec[k] = None

    inserted = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        sb.table("cnis_catalogo").upsert(batch, on_conflict="clave_cnis").execute()
        inserted += len(batch)

    return {
        "status": "ok",
        "total_registros": len(records),
        "insertados": inserted,
        "columnas_detectadas": col_map,
    }


# ---------------------------------------------------------------------------
# POST /api/procesar_catalogo
# ---------------------------------------------------------------------------
@app.post("/api/procesar_catalogo")
async def procesar_catalogo(file: UploadFile = File(...)):
    """Recibe Excel local. Limpia, hace fuzzy match contra CNIS."""
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "Solo se aceptan archivos .xlsx, .xls o .csv")

    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Error al procesar el archivo local: {str(e)}")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect columns
    sal_col = next((c for c in df.columns if "sal" in c and "id" in c), None)
    desc_col = next((c for c in df.columns if "descripcion" in c or "descrip" in c or "medicamento" in c), None)
    precio_col = next((c for c in df.columns if "precio" in c or "costo" in c or "monto" in c), None)
    recetados_col = next((c for c in df.columns if "recetado" in c or "cantidad" in c or "receta" in c), None)

    if not desc_col:
        raise HTTPException(400, f"No se encontró columna de descripción. Columnas: {list(df.columns)}")

    sb = get_sb()

    # Fetch CNIS catalog for matching
    cnis_resp = sb.table("cnis_catalogo").select("id_cnis, clave_cnis, descripcion_cnis_limpia").execute()
    cnis_data = cnis_resp.data
    if not cnis_data:
        raise HTTPException(400, "El catálogo CNIS está vacío. Carga primero el CNIS.")

    cnis_lookup = {r["descripcion_cnis_limpia"]: r for r in cnis_data}
    cnis_descriptions = list(cnis_lookup.keys())

    # Clear previous local data
    sb.table("local_catalogo").delete().neq("id_local", 0).execute()

    results = {"mapeado_seguro": 0, "revision_manual": 0, "total": len(df)}
    records_to_insert = []

    for _, row in df.iterrows():
        desc_sucia = str(row.get(desc_col, ""))
        desc_limpia = limpiar_descripcion(desc_sucia)
        sal_id = str(row.get(sal_col, "")) if sal_col else None
        precio = float(row[precio_col]) if precio_col and pd.notna(row.get(precio_col)) else None
        recetados = int(row[recetados_col]) if recetados_col and pd.notna(row.get(recetados_col)) else None

        # Fuzzy match with thefuzz
        best_match = process.extractOne(
            desc_limpia,
            cnis_descriptions,
            scorer=fuzz.token_sort_ratio,
        )

        id_cnis = None
        score = 0
        estado = "pendiente"

        if best_match:
            matched_desc, score, _ = best_match
            if score >= SCORE_THRESHOLD:
                id_cnis = cnis_lookup[matched_desc]["id_cnis"]
                estado = "mapeado_seguro"
                results["mapeado_seguro"] += 1
            else:
                # Try Gemini fallback for low scores
                gemini_result = await gemini_match(desc_limpia, cnis_descriptions)
                if gemini_result and gemini_result["match"] in cnis_lookup:
                    id_cnis = cnis_lookup[gemini_result["match"]]["id_cnis"]
                    score = gemini_result["score"]
                    estado = "revision_manual"  # Still flag for review since it's AI
                    results["revision_manual"] += 1
                else:
                    estado = "revision_manual"
                    results["revision_manual"] += 1

        record = {
            "sal_id": sal_id,
            "descripcion_sucia": desc_sucia,
            "descripcion_limpia": desc_limpia,
            "precio": precio,
            "recetados": recetados,
            "id_cnis": id_cnis,
            "similitud_score": score,
            "estado_mapeo": estado,
        }
        records_to_insert.append(record)

    # Batch insert
    for i in range(0, len(records_to_insert), BATCH_SIZE):
        batch = records_to_insert[i : i + BATCH_SIZE]
        sb.table("local_catalogo").insert(batch).execute()

    return {
        "status": "ok",
        "total": results["total"],
        "mapeado_seguro": results["mapeado_seguro"],
        "revision_manual": results["revision_manual"],
    }


# ---------------------------------------------------------------------------
# GET /api/tabla_maestra
# ---------------------------------------------------------------------------
@app.get("/api/tabla_maestra")
async def tabla_maestra(
    agrupar_por: Optional[str] = Query(None, description="grupo_terapeutico o indicaciones"),
    estado: Optional[str] = Query(None, description="mapeado_seguro, revision_manual, pendiente"),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
):
    """Devuelve JOIN de local_catalogo + cnis_catalogo con todas las columnas."""
    sb = get_sb()

    # Use the SQL view for the join
    query = sb.table("v_tabla_maestra").select("*")

    if estado:
        query = query.eq("estado_mapeo", estado)

    # Pagination
    offset = (page - 1) * limit
    query = query.range(offset, offset + limit - 1)

    # Order
    if agrupar_por and agrupar_por in ("grupo_terapeutico", "indicaciones"):
        query = query.order(agrupar_por)
    else:
        query = query.order("id_local")

    resp = query.execute()

    # Get total count
    count_query = sb.table("v_tabla_maestra").select("id_local", count="exact")
    if estado:
        count_query = count_query.eq("estado_mapeo", estado)
    count_resp = count_query.execute()

    return {
        "data": resp.data,
        "total": count_resp.count if count_resp.count else len(resp.data),
        "page": page,
        "limit": limit,
        "agrupar_por": agrupar_por,
    }


# ---------------------------------------------------------------------------
# GET /api/revision_manual
# ---------------------------------------------------------------------------
@app.get("/api/revision_manual")
async def get_revision_manual(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
):
    """Devuelve medicamentos pendientes de revisión manual (score < 85)."""
    sb = get_sb()
    offset = (page - 1) * limit

    resp = (
        sb.table("v_tabla_maestra")
        .select("*")
        .eq("estado_mapeo", "revision_manual")
        .order("similitud_score", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    count_resp = (
        sb.table("v_tabla_maestra")
        .select("id_local", count="exact")
        .eq("estado_mapeo", "revision_manual")
        .execute()
    )

    return {
        "data": resp.data,
        "total": count_resp.count if count_resp.count else len(resp.data),
        "page": page,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# PATCH /api/revision_manual/{id_local}
# ---------------------------------------------------------------------------
@app.patch("/api/revision_manual/{id_local}")
async def patch_revision_manual(id_local: int, clave_cnis: str = Query(...)):
    """Asigna manualmente una clave CNIS a un medicamento."""
    sb = get_sb()

    # Find the CNIS record
    cnis_resp = sb.table("cnis_catalogo").select("id_cnis").eq("clave_cnis", clave_cnis).execute()
    if not cnis_resp.data:
        raise HTTPException(404, f"Clave CNIS '{clave_cnis}' no encontrada")

    id_cnis = cnis_resp.data[0]["id_cnis"]

    # Update local record
    sb.table("local_catalogo").update({
        "id_cnis": id_cnis,
        "estado_mapeo": "mapeado_seguro",
        "similitud_score": 100,
    }).eq("id_local", id_local).execute()

    return {"status": "ok", "id_local": id_local, "clave_cnis": clave_cnis}


# ---------------------------------------------------------------------------
# GET /api/buscar_cnis
# ---------------------------------------------------------------------------
@app.get("/api/buscar_cnis")
async def buscar_cnis(q: str = Query(..., min_length=2)):
    """Busca en el CNIS por texto parcial (para autocompletar en revisión manual)."""
    sb = get_sb()
    resp = (
        sb.table("cnis_catalogo")
        .select("id_cnis, clave_cnis, descripcion_cnis_limpia, grupo_terapeutico")
        .ilike("descripcion_cnis_limpia", f"%{q}%")
        .limit(20)
        .execute()
    )
    return {"results": resp.data}


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------
@app.get("/api/stats")
async def stats():
    """Estadísticas generales del sistema."""
    sb = get_sb()

    cnis_count = sb.table("cnis_catalogo").select("id_cnis", count="exact").execute()
    local_count = sb.table("local_catalogo").select("id_local", count="exact").execute()
    mapeados = sb.table("local_catalogo").select("id_local", count="exact").eq("estado_mapeo", "mapeado_seguro").execute()
    revision = sb.table("local_catalogo").select("id_local", count="exact").eq("estado_mapeo", "revision_manual").execute()
    pendientes = sb.table("local_catalogo").select("id_local", count="exact").eq("estado_mapeo", "pendiente").execute()

    return {
        "cnis_total": cnis_count.count or 0,
        "local_total": local_count.count or 0,
        "mapeado_seguro": mapeados.count or 0,
        "revision_manual": revision.count or 0,
        "pendientes": pendientes.count or 0,
    }
