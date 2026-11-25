# Bahréin 2025

`bahrain_2025_ver.csv` contiene 57 registros de vuelta del Gran Premio de
Bahréin del 13 de abril de 2025. La muestra conserva `lap`, `compound`,
`stint` y `life`, y presenta `time`, `status` y `wTT` como `laptime`,
`track_status` y `track_temp_mean`, respectivamente.

- [Archivo de vueltas de VER en TracingInsights](https://raw.githubusercontent.com/TracingInsights-Archive/2025/main/Bahrain%20Grand%20Prix/Race/VER/laptimes.json)
- [Archivo de datos por temporada de TracingInsights](https://tracinginsights.com/data/)

La muestra contiene tiempos de vuelta y temperatura de pista. Para la demo,
el código suma la corrección de combustible del proyecto a cada tiempo y
deriva el indicador de Safety Car del estado de pista. El identificador de
clúster queda en `-1` y los campos sin medición quedan nulos. Los parámetros
secos de referencia provienen de las tablas del informe del proyecto. Las
estrategias recomendadas y sus diferencias de tiempo son simulaciones.
