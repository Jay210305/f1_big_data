# Phase 9.5 — Cierre de los 25 defectos auditados (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework
- Deliverable for: la revisión de integridad del proyecto (25 defectos reproducidos
  y registrados como `xfail(strict=True)`)

## Files created

| File | Purpose |
|------|---------|
| `src/models/calibrate.py` | Calibración del simulador contra los tiempos realizados |
| `tests/test_model_properties.py` | 10 invariantes de modelado (M01–M10), antes `xfail` |
| `tests/test_mapreduce_properties.py` | 4 invariantes del runtime MR (MR01–MR04), antes `xfail` |
| `tests/test_mongo_integration_properties.py` | 5 invariantes de MongoDB (I01, I02, M01m–M03m), antes `xfail` |
| `tests/test_optimizer.py` | 2 tests de regresión nuevos (ritmo fresco, vuelta de vuelo) |
| `src/mapreduce/runner.py` | `_rename()` con reintento ante el bloqueo de archivos de Windows |
| `src/nosql/ingest.py` | Poda de documentos huérfanos en la ingesta completa |

## Steps to run (in order)

```powershell
# 1. Suite completa (antes: 129 passed / 25 xfailed; después: 157 passed / 0 xfailed)
python -m pytest -q --basetemp=.tools/pytest-final

# 2. Regenerar los artefactos que dependen de los modelos corregidos
python -m src.models.train_style
python -m src.models.export_labeled --force
python -m src.models.fit_degradation
python -m src.models.train_rul --epochs 40
python -m src.models.eval_rul

# 3. Re-ejecutar las dos capas de Phase 9 con los artefactos nuevos
python -m src.mapreduce.cli all --force
python -m src.nosql.cli ingest --scope all --refresh-clusters
python -m src.mapreduce.evidence --silver --mongo

# 4. Calibración del simulador
python -m src.models.calibrate --apply

# 5. Backtest y figura
python -m src.models.backtest --sc-hazard
python -m src.eval.plots
```

## What's going on

El proyecto arrastraba 25 defectos reproducidos, registrados como tests
`xfail(strict=True)`: el marcador documentaba la invariante deseada y a la vez
impedía que el test pasara hasta que el defecto se corrigiera. Se han cerrado
los 25 y se han retirado los 25 marcadores, de modo que ahora son aserciones
ordinarias que impiden la regresión.

El defecto de mayor impacto no estaba en la capa de datos sino en el
**simulador de estrategias**: dos errores estructurales (la degradación se
cobraba dos veces, y la vuelta de entrada a boxes se usaba como ritmo base)
que se anulaban en la media y por eso eran invisibles para el backtest. El
detalle completo está en `phase_10_simulator_calibration.md`.

## After you run it

Comprobar `data/models/calibration.json`, `data/mapreduce/parity.json`,
`data/mongo/quality_report.json` y los manifiestos de ingesta. El informe
compacto para el report está en `data/models/phase9_tables.json`.

## Results (filled after running)

### Suite de tests

| Momento | Resultado |
|---|---|
| Antes | 129 passed, **25 xfailed** |
| Después | **157 passed, 0 failed, 0 xfailed** |
| Estabilidad | 12 ejecuciones consecutivas de la suite completa sin fallos |

### Defectos de modelado (M01–M10)

| ID | Defecto | Corrección |
|---|---|---|
| M01 | El descuento de pit por SC/VSC se aplicaba al final de la vuelta siguiente, y la meta de carrera se contaba como parada | `stop_laps = [strategy[i][1] for i < len-1]`, sin filtro de vuelta |
| M02 | Candidato sin paradas (un solo compuesto) y compuestos húmedos en contexto seco | `_allowed_compounds()` + regla de dos compuestos de la F1 |
| M03 | La vuelta de=onicio se sacaba del conjunto de riesgo: laazard podía valer 2.0 | La vuelta de onsets permanece en el riesgo; toda celda ∈ [0,1] |
| M04 | El relleno de las vueltas bajo SC arrastraba el ritmo verde **futuro** hacia atrás | Solo `forward_fill`; las vueltas previas al primer verde quedan nulas |
| M05 | La bandera de lluvia de toda la sesión lavaba la goma antes de que lloviera | El lavado se resuelve por vuelta desde la primera vuelta mojada |
| M06 | El estado en vivo solo se reiniciaba al cambiar de compuesto | El número de stint del feed es la autoridad |
| M07 | El camino en vivo usaba edad 1-based y temperatura de media de sesión | Se adoptan las convenciones del camino por lotes |
| M08 | El modelo guardado no tenía parámetros de compuesto húmedo | Regenerado: β, σ, cliff y offsets para los 5 compuestos (4,628 stints) |
| M09 | El informe de tiempos realizados parseaba la vuelta final como compuesto | Se usa la estrategia estructurada del optimizador |
| M10 | Un β negativo empujaba el límite superior por debajo del inferior físico | El límite superior usa la parte no negativa de la media ajustada |

### Defectos de MapReduce (MR01–MR04)

| ID | Defecto | Corrección |
|---|---|---|
| MR01 | El verificador no comprobaba que el `task` de una fila coincidiera con su partición | `verify.py` lanza `ValueError` ante partición no entera, fuera de rango o `task != partition` |
| MR02 | El protocolo de publicación dejaba el path publicado **ausente** entre dos renombrados | La ruta publicada nunca se renombra: los ficheros se promocionan uno a uno con reversión |
| MR03 | Un particionador que devolvía un valor fraccionario pasaba la comprobación de rango | `_bucket()` exige un `int` real; `bool` y fracciones lanzan `TypeError` |
| MR04 | La referencia Polars ordenaba solo por `life` y el reductor por `(life, lap)` | Ambos usan el mismo desempate |

### Defectos de MongoDB / integración (I01, I02, M01m–M03m)

| ID | Defecto | Corrección |
|---|---|---|
| I01 | Un compuesto `null` canónico rompía la ordenación de `get_state` | Los compuestos desconocidos se filtran del conjunto |
| I02 | Re-ejecutar `style` dejaba obsoletos el labeled frame, las cachés MR y los clusters | Invalidación en `run_all` + detección de deriva en `export_labeled` |
| M01m | El cargador sintético aceptaba `batch_size = 0` y reportaba `n loaded` sin insertar | `ValueError` antes de insertar |
| M02m | La caché del benchmark validaba solo el número de documentos | La caché se valida por hash SHA-256 del contenido |
| M03m | Una ingesta con ámbito de sesión no-Race arrastraba decisiones de carrera | Las decisiones son de nivel carrera y se excluyen del ámbito |

### Dos defectos adicionales hallados al cerrar los anteriores

| Defecto | Corrección |
|---|---|
| Una ingesta completa dejaba documentos obsoletos cuando la fuente cambiaba (200 decisiones `sc_hazard` junto a 346 `clean_air`) | La ingesta completa **poda** los documentos que la fuente ya no produce; una ingesta con ámbito nunca borra |
| `PermissionError` de Windows (WinError 5) en la limpieza del árbol temporal del runner de MR | `_remove_tree()` reintenta y luego cede en silencio; un árbol temporal con nombre único nunca se vuelve a leer |

### Números finales

| Métrica | Valor |
|---|---|
| Tests | **157 passed, 0 failed, 0 xfailed** |
| Beta MR vs Polars | 3.55e-15 (4,628 stints) |
| Beta Mongo vs Polars | 1.78e-15 (4,628 stints) |
| Silver 2025 | 69,459 filas, 0 diferentes |
| Traza | 107,606 → 106,735 → 4,981 claves → 4 particiones, verificada |
| Decisiones en Mongo | 313 (antes 546 con 233 obsoletas) |
| Ingesta idempotente | 0 modified, 0 upserted, 0 pruned |
| Violaciones de esquema | 0 en las 5 colecciones |

- Next step: `phase_10_simulator_calibration.md` (calibración del simulador) y
  la redacción de `MAPREDUCE_Y_MONGODB.md`.
