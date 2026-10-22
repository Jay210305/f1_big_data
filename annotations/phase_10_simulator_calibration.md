# Phase 10 — Calibración del simulador de estrategias (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework
- Deliverable for: el sesgo de calibración detectado en el backtest de Phase 7/9

## Files created

| File | Purpose |
|------|---------|
| `src/models/calibrate.py` | Mide el residual del simulador contra los tiempos realizados y ajusta `pace_drift` |
| `src/models/optimizer.py` | Ritmo base por vuelta desde la referencia *fresco*; `get_state` usa la última vuelta de vuelo |
| `src/models/backtest_realized.py` | Etiqueta explícitamente la circularidad de su propia métrica |
| `tests/test_optimizer.py` | 2 tests de regresión que fijan las dos correcciones |
| `data/models/calibration.json` | Informe de calibración (JSON) |

## Steps to run (in order)

```powershell
python -m src.models.calibrate --apply
python -m src.models.backtest --sc-hazard
python -m src.models.backtest_realized
```

## What's going on

El backtest puntúa al recomendador y al evaluador **con el mismo simulador**.
Eso hace que un sesgo sistemático del simulador sea invisible para el
propio backtest: si el simulador premia sistemáticamente a un tipo de
estrategia, el backtest confirma esa preferencia en lugar de detectarla.

`calibrate.py` cierra el bucle. Evalúa el `optimizer.simulate()` **de
producción** contra las estrategias que **realmente se ejecutaron** en las
carreras de 2025:

```
residuo = simulado − realizado
```

donde "realizado" es la suma de los `laptime` reales (corregidos por
combustible) de las mismas vueltas.

## After you run it

Revisar `data/models/calibration.json` y las tablas por longitud de stint, por
compuesto y por posición en la carrera que imprime el script.

## Results (filled after running)

### El residuo agregado engañaba

La primera medición dio un sesgo agregado de **+179 s**. Era un error de la
propia sonda: se comparaba contra el labeled frame (107,606 filas, con
stints descartados) en vez de contra la tabla Gold de vueltas (317,264
filas). Corregida la población, el sesgo agregado resulta ser de **+0.28 %**:
pequeño para esconder un error estructural grande.

### La señal solo aparece por stint

| Métrica (387 stints secos) | Antes | Después |
|---|---|---|
| Residuo medio | **+38.4 s** | **+0.6 s** |
| Residuo mediano | **+39.7 s** | **−1.7 s** |
| MAD | 114.5 s | 22.7 s |

El residuo además crecía con el número de paradas y con la longitud del stint,
y por compuesto: HARD +41 s, MEDIUM +57 s, SOFT −12 s.

### Los dos errores estructurales

**1. La degradación se cobraba dos veces.** El ritmo base que heredaba una
vuelta era el ritmo *terminal* de la vuelta anterior, de modo que la
degradación de esa vuelta se cobraba una vez dentro de la base heredada y otra
vez con el término de degradación. El error crecía con el número de paradas y
con la longitud de los stints.

Corrección: cada vuelta parte del ritmo **fresco** del coche
(`base_pace − β·k₀`, con `k₀` las vueltas ya recorridas por el neumático
actual), y la degradación ya sufrida por ese neumático se traslada
explícitamente a sus vueltas restantes.

**2. La vuelta de entrada a boxes se usaba como ritmo base.** La decisión se
toma normalmente en la vuelta en la que el coche entra en boxes, que corre
entre 5 y 20 s más lenta que el ritmo de vuelo. `backtest.make_state` lo
corregía localmente, pero **la CLI del optimizador no**: todas las vueltas
siguientes heredaban ese ritmo inflado. `get_state()` ahora usa siempre la
última vuelta de vuelo, para cualquier llamador.

### La causa raíz única

El modelo usa una corrección de combustible fija `fuel_corr = laptime + 0.03·lap`.
El coeficiente de 0.03 s/vuelta subestima la recuperación real de ritmo por
el vaciado de combustible. Como la degradación se ajusta sobre la velocidad
*corregida*, ese mismo error se propaga al otro lado: la β ajustada queda
baja en la misma magnitud.

El coeficiente residual que absorbería el efecto se midió (**−0.04 s/vuelta**,
es decir una tasa efectiva de vaciado de ≈0.07 s/vuelta) pero **no se adoptó**:
desplaza por igual a todos los candidatos y por tanto se cancela en el
ranking que decide la recomendación, mientras que aleja el nivel global de los
datos realizados. La corrección estructural es lo que elimina el sesgo.

### Efecto en el backtest

| Métrica | Antes | Después |
|---|---|---|
| Tasa de victoria | 77.3 % | **77.6 %** |
| Δ mediana | +13.8 s | **+18.0 s** (+31 %) |
| Δ media | +1.1 s | **+7.6 s** |
| p10 / p90 | −72.6 / +37.0 s | **−63.7 / +44.8 s** |
| Paradas recomendadas vs reales | 1.51 vs 1.66 | **1.73 vs 1.66** |
| Estrato coincidente | 223 · 80.3 % · +11.8 s | 185 · 77.3 % · +14.4 s |
| Estrato divergente | 90 · 70.0 % · +21.0 s | 128 · **78.1 %** · **+27.6 s** |

El sesgo sistemático de **sub-paradas desaparece**: antes el optimizador
recomendaba menos paradas que el campo; ahora recomienda 1.73 frente a 1.66
reales. Y los dos estratos convergen: el optimizador ya no se limita a
imitar a los equipos, y donde discrepa gana incluso algo más.

### La métrica de tiempos realizados era circular

`backtest_realized.py` compara la estrategia real con la recomendada usando
los tiempos de vuelta realmente recorridos, pero su delta es
algebraicamente

```
−(pace_offset[recomendado] − pace_offset[real]) × vueltas post-parada
```

es decir **repite los offsets de ritmo del propio modelo** y por tanto no
puede refutarlos. Es una comprobación de consistencia, no una validación
independiente. Antes de este trabajo se tomaba por una validación y se
reportó como si lo fuera; ahora el código, el informe y el runbook lo dicen
explícitamente, y se añade la estadística informativa: el optimizador
recomienda **el mismo compuesto en el 47.1 %** de las 327 decisiones.

La validación independiente es la calibración de este documento.

### Lo que queda

- El residuo todavía depende de la **posición en la carrera**: los stints
  tempranos se sobreestiman y los tardíos se subestiman, porque el simulador
  no tiene término de temperatura de pista ni de tráfico.
- El **offset de ritmo por compuesto sigue sin validarse de forma
  independiente**: la evidencia dentro de una carrera es demasiado ruidosa
  (±2.9 s por par de stints) para resolver un efecto de 0.1–0.3 s, y la
  comprobación de tiempos realizados es circular.

- Decisions made: adoptar la corrección estructural (ritmo fresco + vuelta
  de vuelo) y **no** adoptar el término `pace_drift`; mantener el término en
  el código con valor por defecto 0 y documentarlo como punto de extensión.
- Next-phase risks: el sesgo residual por posición en la carrera; la
  validación del offset por compuesto.
