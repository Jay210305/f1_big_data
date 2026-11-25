# Hadoop Streaming para `stint_stats`

Docker Compose levanta Apache Hadoop 3.5.0 en una sola máquina con un NameNode,
un DataNode, un ResourceManager y un NodeManager. HDFS conserva la entrada y la
salida en volúmenes de Docker. YARN ejecuta el mapper y el reducer de Python.
El pipeline habitual, el MapReduce local y MongoDB pueden usarse por separado.

## Ejecutar una demostración

Desde la raíz del repositorio, con Docker y Compose disponibles para tu
usuario y el entorno Python del proyecto activado:

```bash
docker compose -f compose.hadoop.yaml up --build -d --wait
python -m src.hadoop.cli --demo
```

La primera compilación descarga la imagen de Apache Hadoop e instala Python
dentro de ella. La demo genera 18 vueltas para dos pilotos. Tras excluir las
tres vueltas iniciales y una vuelta de Safety Car de cada stint, el resultado
debe tener dos stints de cinco vueltas válidas, con pendientes cercanas a 0.2
para VER y 0.3 para NOR.

HDFS está disponible en [localhost:9870](http://localhost:9870) y las
aplicaciones de YARN en [localhost:8088](http://localhost:8088).

## Procesar datos reales

La muestra de Bahréin 2025 incluye las 57 vueltas de Verstappen. Puedes
reproducirla desde una copia limpia con el entorno Python del proyecto:

```bash
python -m src.live.demo
python -m src.hadoop.cli --input data/demo/bahrain_2025_ver.parquet
```

Si los datos locales de la capa Gold están disponibles, genera el Parquet
etiquetado con el exportador habitual y ejecuta Hadoop para un Gran Premio:

```bash
python -m src.models.export_labeled
python -m src.hadoop.cli --season 2025 --gp "Bahrain Grand Prix"
```

Omite los filtros para procesar todo el archivo, usa `--input RUTA.parquet`
para leer otro Parquet etiquetado y `--reducers N` para ajustar el número de
reducers. Los Parquet y los conjuntos de datos originales se generan o se
suministran localmente.

## Revisar resultados

Al finalizar, el CLI imprime la ruta en HDFS y guarda `stints.jsonl` y
`parity.json` en `data/hadoop/<id>/`. El reporte registra el tamaño de la
entrada y el error absoluto máximo de la pendiente `beta` frente a Polars y
MapReduce local. La ejecución devuelve error si faltan o sobran stints, si
difieren el compuesto o el número de vueltas válidas, o si el error de
`beta` alcanza `1e-6`. El resultado local se publica después de verificarlo.
Cada ejecución conserva su directorio independiente en HDFS.

Para detener los servicios y conservar los datos de HDFS:

```bash
docker compose -f compose.hadoop.yaml down
```
