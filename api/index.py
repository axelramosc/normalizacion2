"""
MedNorm — API de Normalización de Medicamentos
FastAPI backend for Vercel Serverless
Architecture: Supabase Storage + Batch Processing
"""

import os
import re
import io
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from thefuzz import fuzz, process

# ---------------------------------------------------------------------------
# Config — sanitize env vars aggressively (remove ALL non-printable chars)
# ---------------------------------------------------------------------------
_CLEAN_ENV = re.compile(r'[^\x20-\x7E]')  # keep only printable ASCII
SUPABASE_URL = _CLEAN_ENV.sub('', os.environ.get("SUPABASE_URL", "https://abaazvcjwzvmwkgrtwee.supabase.co"))
SUPABASE_KEY = _CLEAN_ENV.sub('', os.environ.get("SUPABASE_SERVICE_KEY", ""))
GOOGLE_API_KEY = _CLEAN_ENV.sub('', os.environ.get("GOOGLE_API_KEY", ""))

SCORE_THRESHOLD = 85  # >= 85 → mapeado_seguro, < 85 → revision_manual
STORAGE_BUCKET = "catalogos"

app = FastAPI(
    title="MedNorm API",
    description="Normalización de catálogos de medicamentos con CNIS",
    version="3.0.0",
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
    # Unificar todas las abreviaturas a su forma base SNOMED o estándar
    "mgs": "mg", "grs": "g", "gms": "g", "mcgs": "mcg",
    "mls": "ml", "lts": "l", "uis": "ui", 
    "tab": "tableta", "tabs": "tableta", "tabletas": "tableta",
    "cap": "capsula", "caps": "capsula", "capsulas": "capsula",
    "amp": "ampolleta", "amps": "ampolleta", "ampolletas": "ampolleta",
    "fco": "frasco", "frascos": "frasco", "fcos": "frasco",
    "sol": "solucion", "solución": "solucion", "soln": "solucion",
    "susp": "suspension", "suspensión": "suspension",
    "iny": "inyectable", "inyección": "inyectable", "inyeccion": "inyectable",
    "cra": "crema", "crm": "crema",
    "ung": "unguento", "ungüento": "unguento",
    "pvo": "polvo", "polv": "polvo",
    "jbe": "jarabe",
    "gts": "gotas", "got": "gotas",
    "grg": "gragea", "grageas": "gragea",
    "pca": "parche",
    "sup": "supositorio", "supositorios": "supositorio",
    "ped": "pediatrico", "pediatrica": "pediatrico",
    "inf": "infantil",
    "ad": "adulto", "adultos": "adulto",
}


def limpiar_descripcion(texto: str) -> str:
    """Normaliza una descripción quitando puntos, comas, y expandiendo abreviaturas."""
    if pd.isna(texto) or not str(texto).strip():
        return ""
    t = str(texto).lower().strip()
    # Remover comas, puntos y caracteres especiales que no aportan valor
    t = _RE_CLEAN.sub(" ", t)
    
    # Reemplazar abreviaturas por palabras completas basadas en el diccionario
    for original, reemplazo in _UNIT_MAP.items():
        # \b asegura que solo se modifiquen palabras completas (no "estableta" por tener "tab")
        t = re.sub(rf"\b{original}\b", reemplazo, t)
        
    t = _RE_SPACES.sub(" ", t).strip()
    return t


def normalize_medication(description: str, substance_hint: str = None) -> dict:
    """Parse a medication description into SNOMED CT components by systematic removal."""
    result = {
        "sustancia": None,
        "forma_farmaceutica": None,
        "concentracion": None,
        "presentacion": None,
    }
    if not description or pd.isna(description):
        return result
        
    # Texto limpio expandido sin puntuación
    text = limpiar_descripcion(description)
    original_text = text  # Conservamos copia por si queda vacío
    
    # 1. Extraer y remover concentración (ej. 150 mg, 5 g/100 ml)
    conc_match = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|l|ui|%)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|l|ui|%))?)", text, re.IGNORECASE)
    if conc_match:
        c_str = conc_match.group(1)
        result["concentracion"] = c_str.replace(" ", "")
        text = text.replace(c_str, "")

    # 2. Extraer y remover forma farmacéutica
    formas_conocidas = ["tableta", "capsula", "ampolleta", "frasco", "solucion", "suspension", 
                       "inyectable", "crema", "unguento", "polvo", "jarabe", "gotas", "gragea", 
                       "parche", "implante", "gel", "locion", "supositorio", "ovulo", "aerosol", "spray", "pomada"]
    
    forma_encontrada = None
    # Buscamos de forma independiente cada palabra
    for forma in formas_conocidas:
        if re.search(rf"\b{forma}\b", text):
            forma_encontrada = forma
            text = re.sub(rf"\b{forma}s?\b", "", text) # Retiramos plurales por si acaso
            break
            
    if forma_encontrada:
        result["forma_farmaceutica"] = forma_encontrada

    # 3. Extraer y remover presentación (ej. envase con 30 tabletas)
    pres_match = re.search(r"(envase con\s+\d+\s+[a-z]+|\b(?:con|caja con)\s+\d+(?:\s+[a-z]+)?\b)", text, re.IGNORECASE)
    if pres_match:
        result["presentacion"] = pres_match.group(1).strip()
        text = text.replace(pres_match.group(1), "")

    # 4. Limpieza final de la sustancia (Lo sobrante es la sustancia activa)
    # Removemos preposiciones huérfanas o basura residual
    text = re.sub(r"\b(caja|envase|con|para)\b", "", text)
    # Remove loose numbers, slashes, and periods like "1.5/", " . " which are leftovers from concentration
    text = re.sub(r"\b\d+(?:\.\d+)?(?:/|\\s*/\\s*)\b", "", text)
    text = re.sub(r"[,\.\-/]+(?=\s|$)", " ", text) # remove punctuation at end of words
    text = re.sub(r"^\s*[,\.\-/]+", " ", text) # remove punctuation at start
    text = _RE_SPACES.sub(" ", text).strip()

    # Si nos dieron la sustancia_hint explícita, priorizamos limpiarla
    if substance_hint and not pd.isna(substance_hint):
        result["sustancia"] = re.sub(r"[,\.\-]+(?=\s|$)", " ", limpiar_descripcion(substance_hint)).strip()
    else:
        # Si no hay hint, usamos el contenido depurado
        if text:
            # Quitamos prefijos como "contiene "
            text = re.sub(r"^contiene\b", "", text).strip()
            # Quitamos comas y puntos finales perdidos
            text = text.replace(" .", "").replace(" ,", "").replace(".", "").replace(",", "").strip()
            result["sustancia"] = text
            
    return result


def read_file_from_bytes(contents: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or Excel from bytes with encoding fallback."""
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contents), encoding="latin-1")
    else:
        return pd.read_excel(io.BytesIO(contents))


def clean_nan(records: list[dict]) -> list[dict]:
    """Replace NaN/float values with None for JSON serialization."""
    for rec in records:
        for k, v in list(rec.items()):
            try:
                if v is None or pd.isna(v):
                    rec[k] = None
            except (TypeError, ValueError):
                pass
    return records


# ---------------------------------------------------------------------------
# Gemini fallback (optional)
# ---------------------------------------------------------------------------
async def gemini_match(descripcion: str, candidatos: list[str]) -> Optional[dict]:
    """Uses Google Gemini as fallback for fuzzy matching."""
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
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
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
# POST /api/upload_file — Upload CSV/Excel to Supabase Storage
# ---------------------------------------------------------------------------
@app.post("/api/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    type: str = Form("cnis"),  # "cnis" or "local"
):
    """Upload file to Supabase Storage and return metadata."""
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "Solo se aceptan archivos .xlsx, .xls o .csv")

    contents = await file.read()

    # Parse to count rows and detect columns
    try:
        df = read_file_from_bytes(contents, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Error al leer el archivo: {str(e)}")

    total_rows = len(df)
    columns = list(df.columns)

    # Upload to Supabase Storage
    sb = get_sb()
    file_path = f"{type}/{file.filename}"

    try:
        # Remove existing file if any
        try:
            sb.storage.from_(STORAGE_BUCKET).remove([file_path])
        except Exception:
            pass
        # Upload new file
        sb.storage.from_(STORAGE_BUCKET).upload(
            file_path,
            contents,
            {"content-type": "application/octet-stream"},
        )
    except Exception as e:
        raise HTTPException(500, f"Error al subir archivo a Storage: {str(e)}")

    return {
        "status": "ok",
        "file_path": file_path,
        "total_rows": total_rows,
        "columns": columns,
        "filename": file.filename,
    }


# ---------------------------------------------------------------------------
# POST /api/procesar_lote — Process a batch of rows from Storage
# ---------------------------------------------------------------------------
@app.post("/api/procesar_lote")
async def procesar_lote(
    file_path: str = Form(...),
    type: str = Form("cnis"),     # "cnis" or "local"
    offset: int = Form(0),
    batch_size: int = Form(50),
    clear_data: str = Form("false"),  # "true" on first call to clear old data
):
    """Download file from Storage, process rows [offset:offset+batch_size], insert into DB."""
    should_clear = clear_data.lower() in ("true", "1", "yes")

    try:
        sb = get_sb()

        # Download file from Storage
        try:
            file_bytes = sb.storage.from_(STORAGE_BUCKET).download(file_path)
        except Exception as e:
            raise HTTPException(400, f"Error al descargar archivo de Storage: {str(e)}")

        # Determine filename from path
        filename = file_path.split("/")[-1]

        try:
            df = read_file_from_bytes(file_bytes, filename)
        except Exception as e:
            raise HTTPException(400, f"Error al procesar archivo: {str(e)}")

        total_rows = len(df)

        if type == "cnis":
            return await _process_cnis_batch(sb, df, offset, batch_size, total_rows, should_clear)
        elif type == "local":
            return await _process_local_batch(sb, df, offset, batch_size, total_rows, should_clear)
        else:
            raise HTTPException(400, f"Tipo no válido: {type}. Use 'cnis' o 'local'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"procesar_lote error: {e}", exc_info=True)
        raise HTTPException(500, f"Error interno: {str(e)}")


async def _process_cnis_batch(sb, df, offset, batch_size, total_rows, clear_data):
    """Process a batch of CNIS catalog rows."""
    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map column names
    col_map = {}
    for col in df.columns:
        if "clave" in col:
            col_map[col] = "clave_cnis"
        elif ("descripcion" in col or "descrip" in col) and "descripcion_cnis_limpia" not in col_map.values():
            col_map[col] = "descripcion_cnis_limpia"
        elif "grupo" in col or "terapeutico" in col:
            col_map[col] = "grupo_terapeutico"
        elif "indicacion" in col:
            col_map[col] = "indicaciones"
        elif "contraindicacion" in col:
            col_map[col] = "contraindicaciones"
        elif "insumo" in col:
            col_map[col] = "sustancia_hint"

    df = df.rename(columns=col_map)

    # If we have both 'insumo' and 'descripcion' mapped, combine them
    # (handled by taking first mapped one)

    if "clave_cnis" not in df.columns:
        raise HTTPException(400, f"No se encontró columna 'clave'. Columnas: {list(df.columns)}")

    if "descripcion_cnis_limpia" not in df.columns:
        text_cols = [c for c in df.columns if c not in ("clave_cnis", "grupo_terapeutico", "indicaciones", "contraindicaciones")]
        if text_cols:
            df["descripcion_cnis_limpia"] = df[text_cols[0]].astype(str)
        else:
            raise HTTPException(400, "No se encontró columna de descripción")

    # Deduplicate
    df = df.drop_duplicates(subset=["clave_cnis"], keep="first")
    total_unique = len(df)

    # Filter out rows with null clave
    df = df[df["clave_cnis"].notna() & (df["clave_cnis"].astype(str).str.strip() != "")]

    # Get the batch
    batch_df = df.iloc[offset:offset + batch_size]
    if batch_df.empty:
        return {"status": "complete", "processed": 0, "total": total_unique, "remaining": 0}

    # Clean descriptions and parse components
    batch_df = batch_df.copy()
    
    parsed_records = []
    for _, row in batch_df.iterrows():
        desc = str(row.get("descripcion_cnis_limpia", "")).strip()
        hint = row.get("sustancia_hint")
        parsed = normalize_medication(desc, substance_hint=str(hint) if pd.notna(hint) else None)
        
        # Guardar como normalizada aplicando la misma limpieza inicial (minúsculas, sin acentos/comas, etc)
        desc_limpia = limpiar_descripcion(desc)
        
        parsed_records.append({
            "descripcion_cnis_limpia": desc_limpia,
            "sustancia": parsed["sustancia"],
            "forma_farmaceutica": parsed["forma_farmaceutica"],
            "concentracion": parsed["concentracion"],
            "presentacion": parsed["presentacion"]
        })
        
    for k in ["descripcion_cnis_limpia", "sustancia", "forma_farmaceutica", "concentracion", "presentacion"]:
        batch_df[k] = [r[k] for r in parsed_records]

    # Fill NaN
    for col in ["grupo_terapeutico", "indicaciones", "contraindicaciones"]:
        if col not in batch_df.columns:
            batch_df[col] = None
        else:
            batch_df[col] = batch_df[col].where(batch_df[col].notna(), None)

    target_cols = [
        "clave_cnis", "descripcion_cnis_limpia", "grupo_terapeutico", 
        "indicaciones", "contraindicaciones", "sustancia", "forma_farmaceutica", 
        "concentracion", "presentacion"
    ]
    records = batch_df[[c for c in target_cols if c in batch_df.columns]].to_dict("records")
    records = clean_nan(records)

    # On first batch, clear existing data
    if clear_data:
        sb.table("cnis_catalogo").delete().neq("id_cnis", 0).execute()

    # Upsert
    sb.table("cnis_catalogo").upsert(records, on_conflict="clave_cnis").execute()

    processed = offset + len(records)
    remaining = max(0, total_unique - processed)

    return {
        "status": "complete" if remaining == 0 else "processing",
        "processed": processed,
        "batch_inserted": len(records),
        "total": total_unique,
        "remaining": remaining,
    }


async def _process_local_batch(sb, df, offset, batch_size, total_rows, clear_data):
    """Process a batch of local catalog rows with fuzzy matching."""
    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect columns
    sal_col = next((c for c in df.columns if "sal" in c and "id" in c), None)
    desc_col = next((c for c in df.columns if "descripcion" in c or "descrip" in c or "medicamento" in c), None)
    precio_col = next((c for c in df.columns if "precio" in c or "costo" in c or "monto" in c), None)
    recetados_col = next((c for c in df.columns if "recetado" in c or "cantidad" in c or "receta" in c), None)
    presentacion_col = next((c for c in df.columns if "presentaci" in c or "envase" in c), None)

    if not desc_col:
        raise HTTPException(400, f"No se encontró columna de descripción. Columnas: {list(df.columns)}")

    # On first batch, clear existing data and fetch CNIS catalog
    if clear_data:
        sb.table("local_catalogo").delete().neq("id_local", 0).execute()

    # Fetch ALL CNIS catalog for matching via pagination
    cnis_data = []
    page_size = 1000
    start = 0
    while True:
        resp = sb.table("cnis_catalogo").select(
            "id_cnis, clave_cnis, descripcion_cnis_limpia, sustancia, forma_farmaceutica, concentracion, presentacion"
        ).range(start, start + page_size - 1).execute()
        
        if not resp.data:
            break
            
        cnis_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
            
        start += page_size

    if not cnis_data:
        raise HTTPException(400, "El catálogo CNIS está vacío. Carga primero el CNIS.")

    cnis_lookup = {r["descripcion_cnis_limpia"]: r for r in cnis_data}
    cnis_descriptions = list(cnis_lookup.keys())
    cnis_substances_list = list(set([str(r["sustancia"]) for r in cnis_data if r.get("sustancia")]))

    # Get the batch
    batch_df = df.iloc[offset:offset + batch_size]
    if batch_df.empty:
        return {"status": "complete", "processed": 0, "total": total_rows, "remaining": 0}

    records_to_insert = []
    mapeado_seguro = 0
    revision_manual = 0

    for _, row in batch_df.iterrows():
        desc_sucia = str(row.get(desc_col, ""))
        desc_limpia = limpiar_descripcion(desc_sucia)
        sal_id = str(row.get(sal_col, "")) if sal_col else None
        
        # Parse local description
        parsed_local = normalize_medication(desc_sucia)
        if presentacion_col and pd.notna(row.get(presentacion_col)):
            explicit_pres = str(row.get(presentacion_col, "")).strip()
            if explicit_pres:
                parsed_local["presentacion"] = explicit_pres

        sal_id = str(row.get(sal_col, "")) if sal_col else None
        precio = float(row[precio_col]) if precio_col and pd.notna(row.get(precio_col)) else None
        recetados = int(row[recetados_col]) if recetados_col and pd.notna(row.get(recetados_col)) else None

        id_cnis = None
        score = 0
        estado = "pendiente"

        local_substance = parsed_local["sustancia"]
        matched_cnis_substance = None
        substance_fuzzy_score = 0

        if local_substance:
            # 1. First find the best substance match (Exact or Fuzzy on Substance Only)
            if local_substance in cnis_substances_list:
                matched_cnis_substance = local_substance
                substance_fuzzy_score = 100
            else:
                # Fuzzy match just the substance
                best_sub = process.extractOne(local_substance, cnis_substances_list, scorer=fuzz.token_sort_ratio)
                if best_sub and best_sub[1] >= 80:
                    matched_cnis_substance = best_sub[0]
                    substance_fuzzy_score = best_sub[1]

        if matched_cnis_substance:
            # We found a matching substance! Let's find the best specific medication row for this substance
            candidate_rows = [r for r in cnis_data if r.get("sustancia") == matched_cnis_substance]
            
            best_candidate = None
            best_score = -1
            
            for c_row in candidate_rows:
                c_score = 0
                
                # Compare Concentration (Weight: 40 points)
                if parsed_local["concentracion"] and c_row["concentracion"] == parsed_local["concentracion"]:
                    c_score += 40
                
                # Compare Forma Farmaceutica (Weight: 40 points)
                if parsed_local["forma_farmaceutica"] and c_row["forma_farmaceutica"] == parsed_local["forma_farmaceutica"]:
                    c_score += 40
                    
                # Compare Presentacion (Weight: 20 points)
                if parsed_local["presentacion"] and c_row["presentacion"] == parsed_local["presentacion"]:
                    c_score += 20
                
                # Calculate final candidate score: 
                # Combine base substance score (e.g. 100) and component bonus (0-100)
                # If we have exact substance = 100. Component match gives bonuses.
                # Example: Exact substance (100) + Exact Conc (40) + Exact Form (40) = Perfect match
                # Let's average them out or use a weighted system
                # Substance is 60% of the weight, components are 40%
                total_match_score = (substance_fuzzy_score * 0.6) + (c_score * 0.4)
                
                if total_match_score > best_score:
                    best_score = total_match_score
                    best_candidate = c_row

            # Apply thresholds based on best candidate score
            id_cnis = best_candidate["id_cnis"]
            score = int(best_score)
            
            if score >= 85:
                estado = "mapeado_seguro"
                mapeado_seguro += 1
            elif score >= 50:
                estado = "revision_manual"
                revision_manual += 1
            else:
                estado = "revision_manual"  # Even if score < 50, we matched substance, so flag for manual
                revision_manual += 1
                
        else:
            # 2. No substance matched. Do a last resort fuzzy match on full description, but penalize heavily
            best_fallback = process.extractOne(desc_limpia, cnis_descriptions, scorer=fuzz.token_sort_ratio)
            if best_fallback and best_fallback[1] >= 85:
                # We need extremely high certainty on fallback because they bypass substance normalization
                matched_desc, s_score = best_fallback[0], best_fallback[1]
                id_cnis = cnis_lookup[matched_desc]["id_cnis"]
                score = s_score
                estado = "revision_manual"  # Always set fallback matches to manual review to prevent silent bad matches
                revision_manual += 1
            else:
                estado = "pendiente"

        records_to_insert.append({
            "sal_id": sal_id,
            "descripcion_sucia": desc_sucia,
            "descripcion_limpia": desc_limpia,
            "precio": precio,
            "recetados": recetados,
            "sustancia": parsed_local["sustancia"],
            "forma_farmaceutica": parsed_local["forma_farmaceutica"],
            "concentracion": parsed_local["concentracion"],
            "presentacion": parsed_local["presentacion"],
            "id_cnis": id_cnis,
            "similitud_score": score,
            "estado_mapeo": estado,
        })

    # Insert batch
    if records_to_insert:
        sb.table("local_catalogo").insert(records_to_insert).execute()

    processed = offset + len(records_to_insert)
    remaining = max(0, total_rows - processed)

    return {
        "status": "complete" if remaining == 0 else "processing",
        "processed": processed,
        "batch_inserted": len(records_to_insert),
        "total": total_rows,
        "remaining": remaining,
        "mapeado_seguro": mapeado_seguro,
        "revision_manual": revision_manual,
    }


# ---------------------------------------------------------------------------
# GET /api/storage_files — List files in Supabase Storage
# ---------------------------------------------------------------------------
@app.get("/api/storage_files")
async def storage_files(type: str = Query("cnis", description="cnis or local")):
    """List files in the storage bucket for a given type."""
    sb = get_sb()
    try:
        files = sb.storage.from_(STORAGE_BUCKET).list(type)
        result = []
        for f in files:
            if f.get("name") and not f["name"].startswith("."):
                result.append({
                    "name": f["name"],
                    "size": f.get("metadata", {}).get("size", 0),
                    "created_at": f.get("created_at", ""),
                    "updated_at": f.get("updated_at", ""),
                    "path": f"{type}/{f['name']}",
                })
        return {"files": result, "type": type}
    except Exception as e:
        logger.warning(f"Error listing storage files: {e}")
        return {"files": [], "type": type}


# ---------------------------------------------------------------------------
# DELETE /api/limpiar_datos — Clear database tables and optionally storage
# ---------------------------------------------------------------------------
@app.delete("/api/limpiar_datos")
async def limpiar_datos(
    type: str = Query("all", description="cnis, local, or all"),
    clear_storage: bool = Query(False, description="Also delete files from storage"),
):
    """Clear database tables and optionally storage files."""
    sb = get_sb()
    deleted = {"cnis": 0, "local": 0, "storage_cnis": 0, "storage_local": 0}

    try:
        if type in ("cnis", "all"):
            resp = sb.table("cnis_catalogo").delete().neq("id_cnis", 0).execute()
            deleted["cnis"] = len(resp.data) if resp.data else 0
            if clear_storage:
                try:
                    files = sb.storage.from_(STORAGE_BUCKET).list("cnis")
                    paths = [f"cnis/{f['name']}" for f in files if f.get("name") and not f["name"].startswith(".")]
                    if paths:
                        sb.storage.from_(STORAGE_BUCKET).remove(paths)
                        deleted["storage_cnis"] = len(paths)
                except Exception as e:
                    logger.warning(f"Error clearing CNIS storage: {e}")

        if type in ("local", "all"):
            resp = sb.table("local_catalogo").delete().neq("id_local", 0).execute()
            deleted["local"] = len(resp.data) if resp.data else 0
            if clear_storage:
                try:
                    files = sb.storage.from_(STORAGE_BUCKET).list("local")
                    paths = [f"local/{f['name']}" for f in files if f.get("name") and not f["name"].startswith(".")]
                    if paths:
                        sb.storage.from_(STORAGE_BUCKET).remove(paths)
                        deleted["storage_local"] = len(paths)
                except Exception as e:
                    logger.warning(f"Error clearing local storage: {e}")

        return {"status": "ok", "deleted": deleted, "type": type}
    except Exception as e:
        raise HTTPException(500, f"Error al limpiar datos: {str(e)}")


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
    query = sb.table("v_tabla_maestra").select("*")

    if estado:
        query = query.eq("estado_mapeo", estado)

    offset = (page - 1) * limit
    query = query.range(offset, offset + limit - 1)

    if agrupar_por and agrupar_por in ("grupo_terapeutico", "indicaciones"):
        query = query.order(agrupar_por)
    else:
        query = query.order("id_local")

    resp = query.execute()

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
    """Devuelve medicamentos pendientes de revisión manual."""
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

    cnis_resp = sb.table("cnis_catalogo").select("id_cnis").eq("clave_cnis", clave_cnis).execute()
    if not cnis_resp.data:
        raise HTTPException(404, f"Clave CNIS '{clave_cnis}' no encontrada")

    id_cnis = cnis_resp.data[0]["id_cnis"]

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
    """Busca en el CNIS por texto parcial."""
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
