# MapReduce y MongoDB — Arquitectura Big Data del Motor de Estrategia F1

**Documento de referencia técnico** · Versión 1.0 · 2026-10-21

Este documento describe en detalle las dos capas de arquitectura Big Data
implementadas sobre el **motor de estrategia prescriptiva de neumáticos y
stints de Fórmula 1**: un runtime de **MapReduce propio** y una capa
**MongoDB** de modelo documental.

La característica que distingue a este trabajo no es que las dos tecnologías
"estén instaladas", sino que **se integran sobre el pipeline real de
telemetría** y que **demuestran su equivalencia numérica** con la implementación
columnar de referencia. El motor de estrategia produce recomendaciones de
parada; MapReduce reproduce sus estimadores de forma independiente; MongoDB los
sirve en tiempo de ejecución. Los tres motores coinciden hasta el orden de
`10⁻¹⁵`.

---

## Índice

1. [Contexto: dónde encaja esta capa](#1-contexto-dónde-encaja-esta-capa)
2. [Parte I — El runtime de MapReduce](#2-parte-i--el-runtime-de-mapreduce)
3. [Parte II — La capa MongoDB](#3-parte-ii--la-capa-mongodb)
4. [Parte III — La integración](#4-parte-iii--la-integración)
5. [Parte IV — Reproducibilidad](#5-parte-iv--reproducibilidad)
6. [Glosario](#6-glosario)

---

## 1. Contexto: dónde encaja esta capa

### 1.1 El sistema completo

```
  2021/ … 2025/                        57 GB de JSON crudo de telemetría
            │
            ▼  src/ingest/etl.py  (8 workers, memoria acotada)
  BRONZE   325 541 022 filas · 16,69 GB ZSTD
            │
            ▼  src/ingest/etl.py::_silver_exprs
  SILVER   317 264 vueltas (stint_features) + 94 793 filas (race_events)
            │
            ▼  src/features/build_gold.py
  GOLD     lap_features 317 264×61 · labeled_frame 107 606×85
            │  stint_degradation 44 298×19 · driver_embeddings 172×21
            │
            ├──────────────┬──────────────────┬─────────────────────┐
            ▼              ▼                  ▼                     │
      Modelo 1 RUL   Modelo 2 Estilo    Modelo 3 Optimizador         │
      xgb/lgb/lstm    AE + HDBSCAN       MC + enumeración           │
            │              │                  │                     │
            └──────────────┴──────────────────┴─────────┐           │
                                                        ▼           ▼
                                          backtest (313 decisiones)  ┌────────────┐
                                          77,6 % · +18,0 s mediana  │ PARTE II   │
                                                        │           │ MongoDB    │
                                          document/main.tex (ES)    │ operativa  │
                                          strategy_engine_report    └────────────┘
```

### 1.2 Por qué dos capas Big Data y no una

El pipeline columnar (Parquet + DuckDB + Polars) es excelente para **analítica
masiva sobre datos tabulares**, y es el motor de referencia del proyecto. Pero
presenta dos carencias que las dos capas nuevas cubren:

| Carencia del pipeline columnar | Capa que la cubre | Cómo |
|---|---|---|
| La telemetría cruda es **JSON semiestructurado** (`rcm.json`, `weather.json`, `corners.json`); Parquet lo aplana a columnas rígidas y pierde la forma original | **MongoDB** | Modelo documental con sub-documentos embebidos (`race_sessions.weather`, `.corners`, `.rcm`) |
| El paralelismo del ETL **nunca se expresa en el paradigma clave/valor**: no hay *map*, ni *shuffle*, ni *reduce* explícitos que razonar sobre ellos | **MapReduce** | Runtime propio con esas cuatro etapas tipadas y auditables |
| Reejecutar el ETL **reescribe los ficheros Parquet enteros**: no hay actualización a nivel de fila | **MongoDB** | `updateOne(..., upsert=True)` sobre una clave natural determinista |
| No hay una capa de **medición de coste de consulta**: DuckDB no tiene un `explain()` con `totalDocsExamined` | **MongoDB** | `explain("executionStats")` con seis consultas antes/después |
| No hay **tolerancia a fallos** por tarea: un worker que muere tumba el proceso | **MapReduce** | Reintento por tarea y publicación atómica con reversión |
| No hay **traza de auditoría** por etapa del pipeline | **MapReduce** | Traza JSONL con un registro por par clave/valor en cada etapa |

### 1.3 Principio rector

> **Ninguna cifra de este documento es una afirmación: todas son medidas.**

Cada número de las secciones siguientes procede de un artefacto en `data/` o de
una ejecución real reproducible con los comandos del apartado 5.

---

## 2. Parte I — El runtime de MapReduce

### 2.1 Por qué un MapReduce propio y no Hadoop

El requisito de la práctica es la expresión del **paradigma** MapReduce, no de un
producto concreto. Implementarlo sobre `pyspark` o `mrjob` habría exigido un
clúster, un gestor de recursos y un despliegue que no encajan en el entorno
del proyecto (portátil de 32 GB, sin Hadoop, sin WSL, todo ejecutable desde el
`.venv`).

La decisión fue escribir un runtime **propio, sin dependencias externas**: solo
la biblioteca estándar de Python más `polars`/`pyarrow`, que el proyecto ya
usa. El resultado son **1 050 líneas** en `src/mapreduce/` que se ejecutan en un
portátil y que exponen, de forma explícita y auditable, las cuatro etapas
clásicas.

Lo que **no** se afirma: esto no es un sistema distribuido multi-nodo. Es un
runtime de MapReduce **local** con un pool de procesos. El paralelismo de
particiones es real; el intercambio de red entre máquinas no existe.

### 2.2 Arquitectura del runtime

| Módulo | Líneas | Responsabilidad |
|---|---:|---|
| `types.py` | 44 | Contratos: `KV`, `MapContext`, protocolos `Mapper`/`Reducer`/`Combiner`/`Partitioner`, dataclass `JobSpec` |
| `partitioner.py` | 42 | `HashPartitioner` (CRC32), `RangePartitioner` (bisect), `IdentityPartitioner` (demostraciones de *skew*) |
| `runner.py` | 300 | `LocalJobRunner`: troceado, pool de procesos, combiner, shuffle a disco, reductor, publicación atómica, reintentos |
| `trace.py` | 27 | Escritor de traza JSONL: un registro por par clave/valor emitido en cada etapa |
| `io.py` | 60 | Troceadores de Parquet (por *row-group* o por fichero), lector de telemetría, sumidero JSONL canónico |
| `verify.py` | 68 | Verificador independiente de la traza: conservación, propiedad de claves, referencia externa |
| `evidence.py` | 212 | Prueba de paridad entre motores: MR ↔ Polars ↔ Mongo ↔ modelo guardado |
| `cli.py` | 76 | Interfaz de línea de comandos: `run` y `all` |
| `jobs/` | 215 | Los tres trabajos reales sobre datos de producción |

### 2.3 El modelo de ejecución

```
   entrada (Parquet / JSONL)
        │
        │  input_splitter:  una tarea por row-group de Parquet
        ▼
   ┌──────────────┐   el mapper recibe UN registro y devuelve 0..N pares (k,v)
   │     MAP      │   la clave se particiona; el valor se serializa canónicamente
   └──────┬───────┘
          │
          │  combiner (opcional): agregación local asociativa, por clave
          ▼
   ┌──────────────┐   las etapas map/combine escriben JSONL: [partición, clave, valor]
   │   COMBINE    │
   └──────┬───────┘
          │
          │  SHUFFLE + SORT: volcado a SQLite, índice (partition, key, seq)
          ▼
   ┌──────────────┐   cada reductor abre la BD en solo lectura y recibe
   │    SHUFFLE   │   los valores de sus claves, ordenados por clave
   └──────┬───────┘
          ▼
   ┌──────────────┐   el reductor agrupa por clave y emite 0..N pares de salida
   │    REDUCE    │
   └──────┬───────┘
          ▼
   ┌──────────────┐   publicación atómica: nunca se renombra la ruta publicada
   │    COMMIT    │   si falla, se restauran los ficheros anteriores
   └──────────────┘
```

**Detalles de implementación que importan:**

- **Pool de procesos con `spawn`.** Se usa `multiprocessing.get_context("spawn")`
  en lugar de `fork` porque en Windows el `fork` con un pool de hilos de Polars
  puede colgarse. Antes de crear el pool se fija `POLARS_MAX_THREADS=1` para que
  cada worker no cree su propio pool de hilos, y se restaura el valor original
  al terminar.
- **Shuffle a disco, nunca en RAM.** El intermediario es una base SQLite:

  ```sql
  CREATE TABLE kv(seq INTEGER PRIMARY KEY, partition INTEGER, key TEXT, value TEXT);
  CREATE INDEX ordered ON kv(partition, key, seq);
  ```

  Los reductores la abren en solo lectura (`file:...?mode=ro`) y la consultan con
  `WHERE partition=? ORDER BY key, seq`, de modo que cada clave llega a su
  reductor ya agrupada y ordenada. `seq` es el orden global de inserción y
  sirve de desempate determinista dentro de una clave.
- **Serialización canónica.** Todo par clave/valor se serializa con
  `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)`. El
  `allow_nan=False` hace que un `NaN` o un `Inf` raise en lugar de producir
  bytes no reproducibles.

### 2.4 Los tipos y el contrato de un trabajo

```python
KV = tuple[str, Any]

class Mapper(Protocol):
    def __call__(self, record: dict, ctx: MapContext) -> list[KV]: ...
class Reducer(Protocol):
    def __call__(self, key: str, values: list[Any]) -> list[KV]: ...
class Combiner(Reducer, Protocol): ...      # asociativa, mismo tipo de valor
class Partitioner(Protocol):
    def __call__(self, key: str, num_partitions: int) -> int: ...

@dataclass
class JobSpec:
    name: str
    mapper: Mapper
    reducer: Reducer
    combiner: Combiner | None = None
    partitioner: Partitioner = field(default_factory=HashPartitioner)
    num_partitions: int = 4
    input_splitter: Callable | None = None
    output_sink: Callable | None = None
    max_attempts: int = 3
```

Dos invariantes del contrato que el verificador respeta:

- Un mapper puede emitir **cero, uno o muchos** pares: los registros filtrados
  legítimamente no producen nada. Por eso el número de registros de entrada y
  el número de pares emitidos son **contadores distintos**.
- Un combiner debe ser **asociativo** y conservar el tipo de valor: si no, el
  resultado depende del particionado.

### 2.5 Los particionadores

| Particionador | Criterio | Uso |
|---|---|---|
| `HashPartitioner` | `crc32(clave) % n` | El de por defecto. **CRC32 y no `hash()`**: el hash de Python está aleatorizado por proceso (`PYTHONHASHSEED`), de modo que dos ejecuciones repartirían las claves de forma distinta. Con CRC32 la asignación es idéntica entre ejecuciones y entre máquinas. |
| `RangePartitioner` | `bisect_right` sobre límites ordenados | Reparto por rangos de clave; exige exactamente `n − 1` límites |
| `IdentityPartitioner` | Todo a la partición 0 | Demostración de *skew* y de que el runtime detecta un desequilibrio |

Medición del desequilibrio (trabajo `stint_stats`, 4 981 claves):

| Particionador | Particiones | Claves por partición | Tiempo |
|---|---|---|---|
| `identity` | 8 | 4 981 (todas → 0) | 5,81 s |
| `hash` | 8 | 603 – 652 | 5,53 s |

Estas cifras locales son **ilustrativas**, no una afirmación de escalado multi-host.

### 2.6 El combiner: la fusión de Chan, Golub y LeVeque

Solo uno de los tres trabajos usa combiner, y es el que procesa los 325 M de
muestras de telemetría. El valor que viaja por el shuffle no es la lista de
muestras sino un **conjunto de estadísticas suficientes** por canal:

```python
{"n": ..., "sum": ..., "mean": ..., "m2": ..., "min": ..., "max": ...}
```

donde `m2 = var(ddof=0) · n` es la **suma de desviaciones cuadradas
centradas**, que es la magnitud que hace posible fusionar varianzas de forma
exacta. La fusión es el algoritmo de Chan, Golub y LeVeque:

```
n   = n_a + n_b
δ   = media_b − media_a
m2  = m2_a + m2_b + δ² · n_a · n_b / n
media = (suma_a + suma_b) / n
```

que es **exacta y asociativa**: el resultado es independiente del orden y del
particionado. Esto es lo que permite al reductor reconstruir `speed_std` con la
misma desviación muestral (`ddof=1`) que calcula Polars, sin haber transportado
ni una sola muestra cruda por el shuffle.

**El combiner se aplica incrementalmente, dos valores a la vez**
(`combiner(clave, [valor_ya_combinado, valor_nuevo])`), de modo que la memoria
del mapper queda acotada por el número de claves distintas de la tarea, no por
el número de pares emitidos.

### 2.7 El shuffle: qué se mide y por qué importa

El experimento central de la práctica es el **efecto del combiner sobre el
volumen de datos que cruza la red** (aquí, el disco). Se ejecuta el mismo
trabajo con y sin combiner y se comparan los contadores **completos**, no
muestreados:

| Métrica | Sin combiner | Con combiner | Variación |
|---|---:|---:|---|
| Pares clave/valor en el shuffle | 103 244 | **69 459** | **−32,8 %** |
| Bytes serializados del shuffle | 96 823 560 | **65 750 633** | **−32,09 %** |
| Filas de salida | 69 459 | 69 459 | 0 (idéntico) |

El resultado clave: **el combiner reduce un tercio del tráfico de shuffle
sin cambiar una sola fila de la salida**. Esa es la propiedad que define a un
combiner útil: preagregar localmente es seguro exactamente cuando la
operación es asociativa.

### 2.8 La traza: el instrumento de auditoría

`trace.py` escribe un JSONL por etapa (`map.jsonl`, `combine.jsonl`,
`shuffle.jsonl`, `reduce.jsonl`) con un registro por par emitido:

```json
{"stage":"shuffle","task":3,"partition":2,"key":"2025|Bahrain Grand Prix|Race|VER|12","value":{...}}
```

Cada intento de tarea escribe en un **directorio privado**, de modo que los
intentos fallidos nunca entran en la traza publicada. Para el trabajo
`stint_stats` la traza completa ocupa:

| Fichero | `stint_stats` (sin combiner) | `lap_aggregate` (con combiner) |
|---|---:|---:|
| `map.jsonl` | 29,2 MB | 97,6 MB |
| `combine.jsonl` | 0 B | **66,5 MB** |
| `shuffle.jsonl` | 29,6 MB | 66,4 MB |
| `reduce.jsonl` | 1,3 MB | 45,2 MB |

La columna de la derecha muestra el coste de la etapa de combinación: 66,5 MB
de traza de preagregación, que es exactamente el trabajo que el combiner
evita repetir en cada reductor.

### 2.9 El verificador independiente (`verify.py`)

El verificador no se fía del runtime: reconstruye la conservación **desde la
traza**. Sus aserciones son:

| # | Aserción | Por qué importa |
|---|---|---|
| 1 | `len(map) == manifest.kv_before_combine` y `len(shuffle) == manifest.kv_after_combine` | Conservación: ningún par se pierde ni se duplica en el camino |
| 2 | Multiconjunto `map` ≡ `combine` (usando un combiner de referencia independiente) y `combine` ≡ `shuffle` | El combiner no pierde ni duplica valores |
| 3 | **Cada clave llega exactamente a un reductor** | Unicidad de la propiedad de la clave |
| 4 | Una **reducción de referencia** independiente, calculada con Polars y no con el reductor de producción, coincide con la etapa `reduce` | Equivalencia algorítmica, no solo consistencia interna |

> **Dos decisiones de diseño explícitamente contrarias al enunciado de la
> práctica, y documentadas como tales:**
>
> - El enunciado pedía «registros de entrada == suma de KV emitidos». Se
>   **invierte** porque un mapper puede emitir legítimamente cero pares (registros
>   filtrados), de modo que la igualdad literal sería falsa. Se comprueba la
>   **conservación** en su lugar.
> - El enunciado pedía «claves distintas del shuffle == número de particiones
>   reductoras no vacías». Se **no comprueba** porque son magnitudes distintas:
>   el caso real es **4 981 claves en 4 particiones**. Ambas cifras se reportan.
>
> Además, el verificador **rechaza** una traza muestreada: una traza con muestreo
> no puede demostrar cobertura completa, y en su lugar se declara explícitamente
> como ilustración.

### 2.10 Tolerancia a fallos y publicación atómica

**Reintento por tarea.** Solo las tareas fallidas se reponen; una tarea que
ya tuvo éxito **nunca** se vuelve a ejecutar. La identidad de la tarea es
estable entre intentos, pero su *ruta* cambia (`map-3-1`, `map-3-2`), de modo que
los artefactos de un intento fallido no colisionan con los del bueno. Al
agotar los intentos se lanza un `RuntimeError` con la carga útil canónica del
fallo.

**Demostración con fallo inyectado:**

```powershell
python -m src.mapreduce.cli run --job stint_stats --trace --inject-fault map 3 1 `
       --output data/mapreduce/output/stint_stats_retry --force
```

El fallo `RuntimeError: injected fault map-3 attempt 1` se dispara en la tarea
3 de map en su primer intento; el runner reintenta solo esa tarea y el
informe registra `"retries": 1`. El resultado es que **los cuatro ficheros
`part-*.jsonl` finales son byte a byte idénticos** a los de la ejecución sin
fallo:

| Parte | SHA-256 (prefijo) | Tamaño |
|---|---|---:|
| `part-00000` | `b898583f2f94c83a` | 286 432 B |
| `part-00001` | `b1eb8d3535f58ddf` | 267 292 B |
| `part-00002` | `7c907a08d1b26368` | 279 404 B |
| `part-00003` | `c2b807f7a1c4c350` | 275 789 B |

**Publicación sin ventana de invisibilidad.** El protocolo anterior renombraba
la ruta publicada a `.previous-<uuid>` y luego renombraba la nueva encima,
dejando la ruta **ausente** entre ambos renombrados: un lector concurrente
podía ver un directorio vacío. El protocolo actual **nunca renombra la ruta
publicada**: los ficheros preparados se promocionan de uno en uno (cada
renombrado es atómico), los ficheros anteriores se aparcan en el área de
preparación y, si la promoción falla a mitad, los ficheros ya promocionados se
retiran y los anteriores se restauran.

*Limitación honesta:* en Windows `MoveFileEx` no puede reemplazar un directorio,
de modo que la promoción es **por fichero**, no un único renombrado de
directorio. Durante la promoción un lector puede ver una mezcla de ficheros
viejos y nuevos (cada fichero completo). En POSIX sí es posible el intercambio
único.

### 2.11 Determinismo

Con combiner, el mapper escribe las claves **en orden lexicográfico**, de modo
que la salida es byte a byte idéntica con independencia del número de workers
o del troceado. Es una propiedad comprobada por test
(`test_integer_reduction_is_byte_stable_across_workers_splits_and_combining`).

### 2.12 Los tres trabajos reales

| Trabajo | Entrada | Clave | Salida |
|---|---|---|---|
| `lap_aggregate` | Telemetría Bronze de 2025 | `season\|gp\|session\|driver\|lap` | **69 459** filas Silver — puerto directo de `etl.py::_silver_exprs` |
| `stint_stats` | `labeled_frame.parquet` | `season\|gp\|session\|driver\|stint` | **4 628** stints + el modelo β/σ del optimizador |
| `driver_profile` | `labeled_frame.parquet` | `driver\|session_type` | **70** perfiles por piloto y tipo de sesión |

**`stint_stats` es el más importante**, porque su salida **es** el modelo de
degradación que gobierna el optimizador: reproduce exactamente el estimador de
`fit_degradation.py`, incluidos el descarte de las tres primeras vueltas del stint
(el calentamiento y el tráfico de salida sesgan la pendiente a la baja), el
excluido de las vueltas bajo Safety Car, la fórmula `cov(life, fuel_corr) / var(life)`
con la corrección de tamaños muestrales, el mínimo de cuatro vueltas, el IQR
robusto con el factor 1,349 y el recorte de σ al intervalo [0,02; 0,5].

```
beta[compuesto]  = mediana de las pendientes por stint
sigma[compuesto] = (Q3 − Q1) / 1,349        (cuantil "nearest", como Polars)
```

### 2.13 La prueba de equivalencia: tres motores, un número

Este es el resultado central de la parte I. El mismo estimador se calcula en
tres motores independientes y se comparan stint a stint:

| Comparación | Stints comparados | Diferencia máxima |
|---|---:|---:|
| MapReduce ↔ Polars (referencia independiente) | 4 628 | **3,55 × 10⁻¹⁵** |
| MongoDB `$group` ↔ Polars | 4 628 | **1,78 × 10⁻¹⁵** |
| Modelo guardado ↔ MapReduce (β y σ, compuestos secos) | 4 274 | **0,0** (σ: 1,39 × 10⁻¹⁷) |
| Bronze 2025 → Silver (reproducción fila a fila) | 69 459 | **0 filas diferentes** (máx. 2,70 × 10⁻¹³) |
| Perfil por piloto ↔ Polars | 70 grupos | 2,84 × 10⁻¹⁴ |
| Traza: 107 606 entradas → 106 735 pares → 4 981 claves → 4 reductores | — | **verificada** |

La diferencia media es de **1,98 × 10⁻¹⁷** segundos por vuelta: ruido de
punto flotante puro. Los tres motores calculan **el mismo número**.

> **Una comparación que se declara explícitamente NO equivalente.** La tabla
> `stint_degradation.parquet` de la capa Gold contiene 44 298 pendientes
> *crudas*, calculadas sobre el `laptime` sin corrección de combustible, con
> otro filtro de population y otro objetivo. Frente a las 4 628 pendientes
> elegibles equivalentes, la diferencia media es **3,71** y la máxima **363,08**.
> Se reporta como **comparación diagnóstica**, no como afirmación de equivalencia.

### 2.14 Limitaciones honestas de la parte I

- Es un runtime **local**, no un clúster. El paralelismo de particiones es real;
  el intercambio de red entre máquinas no existe y no se mide.
- La publicación es **por fichero** en Windows (ver §2.10).
- El trabajo `lap_aggregate` se ha ejecutado sobre **una temporada** (2025), no
  sobre las cinco: **48,38 s** sin combiner y sin traza, **56,02 s** con
  combiner y sin traza, y **118,81 s** con la traza completa (la primera
  medición, con el disco bajo carga, fue de 89,70 s: el tiempo de la traza
  depende mucho de la carga de disco). La reproducción Silver covers 69 459
  vueltas.
- El combiner se aplica **incrementalmente** (dos valores a la vez), lo que
  acota la memoria del mapper pero es más lento que agrupar y fusionar de una
  vez.

---

## 3. Parte II — La capa MongoDB

### 3.1 Por qué un modelo documental

La telemetría de FastF1 incluye JSON **semiestructurado** que Parquet aplana:
`rcm.json` (mensajes de control de carrera: banderas, SC, VSC), `weather.json`
(series temporales de temperatura, lluvia, viento) y `corners.json` (geometría
del circuito). Al pasarlos a columnas rígidas se pierde:

- la **forma original** del mensaje,
- la **granularidad temporal** (una serie de observaciones por vuelta),
- la **naturaleza condicional** de la señal (el viento circular, por ejemplo).

MongoDB, con su modelo documental, conserva esa forma y además ofrece
actualización por fila, índices medibles y agregaciones en el motor.

### 3.2 El modelo de documentos

Cinco colecciones, todas con **clave natural determinista** (por eso la ingesta
es idempotente):

| Colección | Fuente | Documentos | `_id` natural |
|---|---|---:|---|
| `race_sessions` | Silver `race_events` + Gold `environmental` | **576** | `season\|gp\|session` |
| `stints` | Gold `stint_degradation` | **44 298** | `season\|gp\|session\|driver\|stint` |
| `laps` | Gold `labeled_frame` | **107 606** | `season\|gp\|session\|driver\|lap` |
| `strategy_decisions` | `phase7_backtest.json::decisions` | **313** | `season\|gp\|driver\|scenario` |
| `bench_laps` | `src/nosql/synthetic.py` | **100 000** | `bench-<i>` |

**`race_sessions` es el documento richest** y el que justifica el modelo
documental: embide el clima, la geometría de las curvas y el histórico de
control de carrera dentro del mismo documento.

```json
{
  "_id": "2025|Bahrain Grand Prix|Race",
  "season": "2025", "gp": "Bahrain Grand Prix", "session": "Race",
  "session_type": "Race",
  "weather": { "track_temp_mean": 31.9, "ambient_temp_mean": 27.4, "rain_flag": 0, ... },
  "weather_samples": [ { "lap": 1, "track_temp": 30.1, ... }, ... ],
  "corners":   [ { "number": 1, "x": 120.4, "y": -80.2, ... }, ... ],
  "rcm":       [ { "lap": 12, "cat": "SafetyCar", "msg": "SAFETY CAR DEPLOYED" }, ... ],
  "sc_laps":   [ 12, 13, 14 ],
  "sc_laps_source": "rul_data.load_sc_laps: derived SC/VSC interval expansion",
  "n_laps": 57, "n_drivers": 20
}
```

Lo que se gana de verdad es `weather_samples`: la serie temporal por vuelta del
clima, que en el formato columnar se habría tenga que reconstruir con
unpivot y un `join` por lap.

### 3.3 Validadores estrictos (`$jsonSchema`)

Cada colección se crea (o se modifica con `collMod`) con un validador
`$jsonSchema` y `validationLevel: "strict"`, `validationAction: "error"`: un
documento que no encaja **es rechazado en la escritura**, no filtrado en
consulta.

```json
{
  "bsonType": "object",
  "required": ["_id", "season", "gp", "session", "driver", "lap", "stint", "life", "compound"],
  "properties": {
    "_id":      { "bsonType": "string" },
    "season":   { "bsonType": "string" },
    "lap":      { "bsonType": ["int", "long"] },
    "laptime":  { "bsonType": ["double", "int", "long", "decimal", "null"] },
    "compound": { "bsonType": ["string", "null"] }
  }
}
```

**Medición:** **0** documentos que violan el validador en las cinco
colecciones, y **0** `laptime` no numéricos en `laps`.

### 3.4 Una sola capa de normalización

El error más caro de un proyecto con dos capas de ingesta es tener **dos**
normalizadores que divergen. Aquí hay **uno solo**: `src/nosql/cleaning.py`
reutiliza `to_float`, `to_int` y `to_str` de `src/ingest/schema.py`, incluido
el conjunto de tokens nulos del dataset original:

```python
NULL_TOKENS = frozenset({"", "None", "none", "null", "N/A", "nan"})
```

Reglas adicionales propias de BSON:

| Regla | Motivo |
|---|---|
| `float` no finito → `null` (+ contador `nonfinite_to_null`) | MongoDB *puede* representar NaN/Infinity en BSON, pero un regressor no debe verlos: se rompen las métricas y la serialización deja de ser reproducible |
| `bool` → `int` en los campos enteros | El tipo de Python `bool` es subclase de `int` y colaría como 0/1 silenciosamente |
| Cadenas → `to_str` | Hereda el tratamiento de `"None"` del pipeline columnar |

Cada conversión se contabiliza en un **libro de coerciones** que se guarda en
los manifiestos de ingesta: por ejemplo, en la ingesta completa de `laps`,
`bool_to_int` 107 606 y `str_to_NoneType` 871.

### 3.5 La ingesta: idempotencia y reconciliación

**Upsert por clave natural.** Los documentos se escriben en lotes de 5 000 con
`bulk_write` de `UpdateOne({"_id": ...}, {"$set": ...}, upsert=True)` y
`ordered=False` (un documento problemático no aborta el lote).

**Idempotencia demostrada.** Una segunda ingesta completa de los mismos datos:

| Colección | procesados | modificados | insertados | podados | total |
|---|---:|---:|---:|---:|---:|
| `race_sessions` | 576 | **0** | **0** | 0 | 576 |
| `stints` | 44 298 | **0** | **0** | 0 | 44 298 |
| `laps` | 107 606 | **0** | **0** | 0 | 107 606 |
| `strategy_decisions` | 313 | **0** | **0** | 0 | 313 |

**Reconciliación con la fuente (defecto corregido).** Antes, un cambio en la
fuente dejaba documentos huérfanos: al cambiar el escenario del backtest
quedaron **200 decisiones `sc_hazard` huérfanas** junto a 346 nuevas `clean_air`,
y la colección dejó de ser un espejo fiel del dataset. Ahora una ingesta
**completa** poda los documentos que la fuente ya no produce (233 podados en
esa ocasión), mientras que una ingesta **con ámbito** nunca borra, porque solo
cubre una parte de la colección.

### 3.6 Los índices y la evidencia de `explain()`

| Colección | Índices |
|---|---|
| `laps` | `{season, gp, driver, lap}` · `{compound, life}` · `{laptime}` |
| `strategy_decisions` | `{season, gp, driver}` |
| `stints` | `{season, gp, driver}` |

Seis consultas representativas, medidas con
`explain("executionStats")` **sobre los mismos datos**, con la ruta de acceso
forzada de forma controlada (`$natural` frente al índice con *hint* explícito):

| # | Consulta | Docs examinados antes | después | Tiempo antes → después |
|---|---|---:|---:|---|
| 1 | Historial de un piloto `{season, gp, driver}` | 107 606 | **30** | 55 ms → **0 ms** |
| 2 | Temporada `{season:"2021"}` | 107 606 | 21 179 | 100 ms → 11 ms |
| 3 | Edad del neumático `{compound:"SOFT", life≥15}` | 107 606 | 5 016 | 49 ms → 9 ms |
| 4 | Neumático duro `{compound:"HARD"}` | 107 606 | 42 763 | 48 ms → **46 ms** |
| 5 | Vueltas lentas `{laptime>120}` | 107 606 | 4 601 | 75 ms → 13 ms |
| 6 | Vueltas rápidas `{laptime<80}` | 107 606 | 26 659 | 54 ms → 33 ms |

La consulta 4 es una **no mejora genuinamente medida** (48 ms → 46 ms) y se
reporta como tal, sin esconderla.

> **Qué no se afirma.** La comparación usa `$natural` y *hints* de índice
> nombrados sobre los mismos datos: mide rutas de acceso controladas **sin
> eliminar los índices del usuario**. No demuestra que un planificador sin
> *hint* elija siempre `IXSCAN`.

### 3.7 El coste de los índices, medido

Con 100 000 documentos sintéticos:

| Métrica | Solo `_id` | Con 4 índices | Ratio |
|---|---:|---:|---:|
| Rendimiento de inserción | **66 366 docs/s** | 43 053 docs/s | **1,5415×** |
| Tiempo de inserción | 1,507 s | 2,323 s | 1,54× |
| Tamaño lógico del documento | 21 111 910 B | 21 111 910 B | 1,00× |
| Tamaño total de índices | 610 304 B | **2 301 952 B** | 3,77× |
| Número de índices | 1 | 4 | — |

La lectura honesta: **los índices cuestan ~1,54× en escritura y 3,77× en
espacio**, a cambio de evitar un recorrido completo de 107 606 documentos en
las consultas de servicio.

### 3.8 Streaming con memoria acotada

El cursor con `batch_size` mantiene acotada la memoria del cliente:

| Modo | Asignación máxima de Python | Presupuesto de 16 MiB |
|---|---:|---|
| Cursor en streaming (`batch_size=1000`) | **1 649 106 B** | ✅ dentro |
| Materialización completa en una lista | **141 560 662 B** | ❌ supera |

**Ratio: 85,84×.** Y el crecimiento está acotado: de 1 000 a 10 000 y a 100 000
documentos, la asignación máxima se mueve de 1 632 402 B a 1 646 231 B y a
1 650 682 B.

> **Qué mide y qué no mide.** Es `tracemalloc` —asignaciones de Python— y
> **no** el RSS total del proceso. Mide la memoria que el código de
> agregación retiene, que es la parte que el programador controla; no mide la
> memoria del motor de MongoDB, que es del servidor.

### 3.9 Las agregaciones: `$group` es el *reduce* del motor

El motor de MongoDB ejecuta `$group` de forma nativa con una semántica de
clave/valor equivalente a la etapa *reduce* del runtime MapReduce, pero dentro
del motor de almacenamiento en vez de en un clúster de cómputo. Las
implementaciones de `src/nosql/aggregation.py`:

| Pipeline | Reproduce |
|---|---|
| `compound_pace` | Ritmo medio por compuesto y temporada (`$match`/`$group`/`$avg`/`$stdDevSamp`/`$sort`) |
| `stint_slopes` | La pendiente de degradación por stint, calculada con el **mismo estimador** que MapReduce y Polars |
| `safety_car_share` | Proporción de vueltas bajo Safety Car |
| `race_control_counts` | Recuento de mensajes de control de carrera (`$unwind`) |
| `pace_facet` | Ritmo segmentado por compuesto y piloto (`$facet`) |

La paridad de `stint_slopes` con MapReduce y Polars es la prueba formal de
que la equivalencia semántica se sostiene: **4 628 stints, diferencia máxima
1,78 × 10⁻¹⁵**.

### 3.10 MongoDB como fuente operativa del optimizador

El optimizador puede leer su estado de la base de datos en lugar del Parquet:

```powershell
python -m src.models.optimizer --source mongo --season 2025 `
       --gp "Bahrain Grand Prix" --driver VER --lap 12 --n-sims 500
```

El estado que necesita (`fuel_corr`, `pace_above_best`, `compound`, `stint`,
`life`, `track_temp_mean`, `cluster`) **está en el documento** porque la ingesta
exportó el labeled frame entero; y la longitud de la carrera se toma de
`race_sessions` (sin filtrar), que es el número de vueltas real y no el de los
stints que sobrevivieron al filtrado.

**El comportamiento ante fallos es explícito:**

- **Error de conexión** → se registra y se cae a Parquet, y el proceso sigue.
- **Conectado pero sin datos** → es un **error de ingesta**, no un motivo para
  caer a Parquet: un silencio de datos sería peor que un fallo visible.

### 3.11 La pestaña MongoDB del dashboard

`app/streamlit_app.py` se reestructuró en dos pestañas: **«Vivo»** (el
recomendador en carrera, que no necesita MongoDB) y **«Historial MongoDB»**.
La segunda funciona **sin subir ningún fichero de replay**:

- Filtros de temporada, GP, piloto y compuesto.
- Tabla paginada de decisiones (25 por página) con salto de página.
- **Histograma del delta calculado en el servidor**, con un *pipeline* de
  agregación en `repository.py` — no en Python.
- Panel de `collStats` con el tamaño de la colección y de los índices.
- Banner de degradación elegante cuando el servidor no está disponible.

### 3.12 Limitaciones honestas de la parte II

- **Un solo nodo local.** No hay conjunto de réplicas ni *sharding*. La
  escalabilidad horizontal se **discute**, no se demuestra.
- El **coste de escritura** de los índices es real y medido (1,54×); con una
  carga de solo lectura el balance es distinto.
- El modelo de `strategy_decisions` guarda el historial de decisiones
  (313 documentos), no el espacio de estados completo que el optimizador
  exploraría.
- El `explain()` se midió con rutas de acceso forzadas; no es una medición del
  planificador real.

---

## 4. Parte III — La integración

### 4.1 La misma recomendación desde dos motores

La prueba de que las dos capas no son un adorno: el optimizador produce **la
misma recomendación** leyendo de Parquet o de MongoDB.

| Comprobación | Resultado |
|---|---|
| Estados de carrera comparados (Mongo vs Parquet) | **5 / 5** coinciden campo a campo |
| Recomendaciones resultantes | **5 / 5** coinciden |
| Salida de la CLI (Bahrain / VER / vuelta 12) | **byte a byte idéntica** entre fuentes |
| La fuente elegida se registra en la salida | ✅ `[source] mongo (labeled lap history + unfiltered session length)` |

### 4.2 La cadena de evidencia completa

```powershell
# 1. Materializar la entrada compartida (labeled frame)
python -m src.models.export_labeled

# 2. Ejecutar los tres trabajos de MapReduce y su evidencia
python -m src.mapreduce.cli all

# 3. Ingesta en MongoDB (dos veces: idempotencia)
python -m src.nosql.cli ingest --scope all --output data/mongo/ingest_first.json
python -m src.nosql.cli ingest --scope all --output data/mongo/ingest_second.json

# 4. La prueba de paridad a tres motores
python -m src.mapreduce.evidence --silver --mongo

# 5. Las tres figuras de evidencia
python -m src.eval.plots --only mapreduce_shuffle
python -m src.eval.plots --only mongo_indexes
python -m src.eval.plots --only three_way_parity
```

### 4.3 Las figuras

| Figura | Qué demuestra |
|---|---|
| `eval_plots/mapreduce_shuffle.png` | Pares y claves por partición reductora, con y sin combiner, con la reducción de bytes anotada |
| `eval_plots/mongo_indexes.png` | `totalDocsExamined` y `totalKeysExamined` antes/después, más el coste de almacenamiento de los índices |
| `eval_plots/three_way_parity.png` | La pendiente por stint calculada por MapReduce, por Polars y por Mongo `$group` |

Las tres están en el informe `document/main.tex` (§5, «Arquitecturas Big Data:
MapReduce y MongoDB») y las dos primeras en el informe en inglés.

### 4.4 Lo que la paridad prueba y lo que no

> **La paridad demuestra equivalencia computacional, no validez causal.**
> Que tres motores independientes produzcan el mismo número prueba que la
> implementación de MapReduce y la de MongoDB son correctas **con respecto al
> estimador**. No dice nada sobre si el estimador es una buena modelo de la
> física del neumático.

Esa distinción se respeta en el informe: la sección §5.8 «Integración y
limitaciones» la enuncia de forma explícita.

---

## 5. Parte IV — Reproducibilidad

### 5.1 Requisitos

- Python ≥ 3.11 en el `.venv` del proyecto (`python -m pip install -r requirements.txt`).
- MongoDB 7.0.45 en `localhost:27017`:

  ```powershell
  docker run -d --name f1-mongo -p 27017:27017 -v f1mongo:/data/db mongo:7
  # ejecuciones posteriores:  docker start f1-mongo
  ```

  Variables de entorno opcionales: `MONGO_URI`, `MONGO_DB` (por defecto
  `mongodb://localhost:27017` y `f1_tyre`).
- El lakehouse construido (`data/bronze`, `data/silver`, `data/gold`): 16,7 GB.

### 5.2 Comprobación de estado

```powershell
python -m src.nosql.cli health        # -> OK MongoDB 7.0.45 database=f1_tyre
python -m src.mapreduce.cli --help
python -m src.mapreduce.cli run --job stint_stats --trace
```

### 5.3 Las pruebas

```powershell
# Pruebas puras: no necesitan ni el dataset ni MongoDB
python -m pytest tests/test_mapreduce.py tests/test_nosql.py `
                 tests/test_phase9_parity.py tests/test_phase9_integration.py -q

# Las pruebas marcadas con @pytest.mark.mongo se saltan si el servidor no está
```

Suite completa del proyecto: **157 pruebas, 0 fallos**.

### 5.4 Artefactos que produce

| Ruta | Contenido |
|---|---|
| `data/mapreduce/output/<trabajo>/` | `part-*.jsonl`, `metrics.json`, `trace/` |
| `data/mapreduce/parity.json` | Evidencia completa de paridad a tres motores |
| `data/models/phase9_tables.json` | Instantánea compacta para el informe |
| `data/mongo/ingest_{first,second}.json` | Manifiestos de ingesta (idempotencia) |
| `data/mongo/explain.json` | `explain("executionStats")` de las seis consultas |
| `data/mongo/benchmark.json` | Rendimiento, `collStats` y memoria del cursor |
| `data/mongo/quality_report.json` | Cumplimiento de esquema y campos ausentes |
| `eval_plots/*.png` | Las figuras de evidencia |

---

## 6. Glosario

| Término | Significado en este proyecto |
|---|---|
| **KV** | Par clave/valor, la unidad que viaja por el shuffle |
| **Combiner** | Preagregación local asociativa; reduce el tráfico de shuffle sin cambiar la salida |
| **Shuffle** | Redistribución de las claves desde los mappers a los reductores; aquí, volcada a SQLite |
| **Reduce** | Agregación final por clave; en MongoDB, su equivalente nativo es `$group` |
| **Particionador** | Función clave → partición reductora; aquí CRC32, por ser estable entre procesos |
| **Traza** | Registro JSONL de cada par emitido en cada etapa, base de la verificación independiente |
| **Clave natural** | `_id` derivado de los atributos del propio dato, que hace idempotente la ingesta |
| **Upsert** | `UpdateOne` con `upsert=True`: inserta si no existe, actualiza si existe |
| **`$jsonSchema`** | Validador de documentos de MongoDB, aplicado en la escritura con nivel estricto |
| **`explain("executionStats")`** | Plan de ejecución con el coste real de una consulta: documentos y claves de índice examinados |
| **`collStats`** | Estadísticas de colección: tamaño, tamaño de índices y número de índices |
| **Estadística suficiente** | Par mínimo de números (n, suma, media, m2) que permite fusionar varianzas de forma exacta |
| **Paridad** | Acuerdo numérico entre implementaciones independientes del mismo estimador |

---

**Documentos relacionados:** [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) (el proyecto
completo) · [`RUNBOOK.md`](RUNBOOK.md) (procedimiento completo) ·
[`WORK_LOG.md`](WORK_LOG.md) (registro de trabajo) ·
[`document/main.tex`](document/main.tex) §5 (el mismo contenido en el informe) ·
[`document/phase9_traceability.md`](document/phase9_traceability.md) (matriz de
trazabilidad requisito → implementación → evidencia).
