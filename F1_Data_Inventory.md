# F1 Data Inventory

This document describes the type, structure, and size of the Formula 1 data stored in this project.

## Project Overview

This repository is a local collection of the **TracingInsights public F1 telemetry datasets** for the 2021–2025 Formula 1 seasons. Each season is organized as a self-contained folder that mirrors the upstream TracingInsights GitHub repository for that year.

The data is primarily stored as **JSON files** and is arranged by:

- **Season** (e.g., `2024/`)
- **Grand Prix event** (e.g., `Bahrain Grand Prix/`)
- **Session** (e.g., `Race/`, `Qualifying/`, `Practice 1/`)
- **Driver** (e.g., `VER/`, `HAM/`)
- **Lap** (e.g., `1_tel.json`)

The collection also includes Python extraction scripts, FastF1 caches, and Git metadata.

---

## Storage Summary

| Year | Grand Prix Folders | Total Files | JSON Files | Approx. JSON Data Size | `.git` History Size | Total Folder Size |
|------|-------------------|------------:|-----------:|-----------------------:|--------------------:|------------------:|
| 2021 | 22 | ~63,044 | 62,996 | 11.27 GB | 19.61 GB | ~30.88 GB |
| 2022 | 22 (+ Pre-Season Test) | ~63,048 | 62,999 | 10.21 GB | 17.54 GB | ~27.75 GB |
| 2023 | 22 (+ Pre-Season Testing) | ~62,659 | 62,608 | 10.18 GB | 17.24 GB | ~27.42 GB |
| 2024 | 23 (+ Pre-Season Testing) | ~70,729 | 70,668 | 11.65 GB | 17.81 GB | ~29.46 GB |
| 2025 | 23 (+ Pre-Season Testing) | ~106,711 | 106,663 | 13.74 GB | 14.71 GB | ~28.45 GB |

> **Note:** The `.git/` directories contain the full Git object history (mostly compressed pack files) for each season repository. They are not part of the usable F1 dataset but account for a large portion of the on-disk footprint.

---

## Folder Structure

```text
fomrula_1_data/
├── 2021/
│   ├── Abu Dhabi Grand Prix/
│   │   ├── Practice 1/
│   │   ├── Practice 2/
│   │   ├── Practice 3/
│   │   ├── Qualifying/
│   │   └── Race/
│   ├── Austrian Grand Prix/
│   ├── ... (20 more events)
│   ├── cache/
│   ├── *.py              # extraction/analysis scripts
│   ├── README.md
│   └── requirements.txt
├── 2022/                 # same structure
├── 2023/                 # same structure
├── 2024/                 # same structure
└── 2025/                 # same structure + extra cache dirs
```

Each `{Year}/{Grand Prix}/{Session}/` directory follows the same layout.

---

## Data Types

### 1. Per-Lap Telemetry (`{lap_number}_tel.json`)

Stored inside each driver folder (e.g., `2024/Bahrain Grand Prix/Race/VER/1_tel.json`).

Each file contains high-frequency telemetry samples for a single lap. Fields include:

| Field | Description |
|-------|-------------|
| `time` | Elapsed time on lap (seconds) |
| `speed` | Car speed (km/h) |
| `throttle` | Throttle input percentage |
| `brake` | Brake application |
| `gear` | Selected gear |
| `rpm` | Engine RPM |
| `drs` | DRS activation status/zone |
| `acc_x` | Lateral acceleration (G) |
| `acc_y` | Longitudinal acceleration (G) |
| `acc_z` | Vertical acceleration (G) |
| `x`, `y`, `z` | Track position coordinates |
| `distance` | Distance into lap (meters) |
| `rel_distance` | Relative distance on lap |
| `DistanceToDriverAhead` | Gap to car ahead (meters) |
| `DriverAhead` | Driver code of car ahead |
| `dataKey` | Internal identifier |

### 2. Driver Lap Times (`laptimes.json`)

Stored inside each driver folder.

Contains summarized lap data for the whole session:

| Field | Description |
|-------|-------------|
| `time` | Lap time in seconds |
| `lap` | Lap number |
| `compound` | Tyre compound used (SOFT, MEDIUM, HARD, etc.) |
| `stint` | Stint number |
| `s1`, `s2`, `s3` | Sector 1, 2, and 3 times |

### 3. Session-Level Lap Times (`session_laptimes.json`)

Stored at the session root. Combines all per-driver `laptimes.json` arrays into a single file.

### 4. Driver Metadata (`drivers.json`)

Stored at the session root. Contains the list of participating drivers with:

- Driver code, first name, last name
- Team name
- Driver number
- Team color hex code
- F1 media image URL

### 5. Weather Data (`weather.json`)

Stored at the session root. Contains track weather samples such as:

- `wT`: timestamps
- `wAT`: ambient/air temperature
- (Additional weather channels may vary by session)

### 6. Corner Data (`corners.json`)

Stored at the session root. Contains circuit corner information:

| Field | Description |
|-------|-------------|
| `CornerNumber` | Corner index |
| `X`, `Y` | Corner coordinates |
| `Angle` | Corner angle |
| `Distance` | Distance into lap (meters) |
| `Rotation` | Track rotation value |

### 7. Race Control Messages (`rcm.json`)

Stored at the session root. Contains timestamped race control messages during the session.

### 8. Standings (`standings.json`) — 2025 only

Located at `2025/standings.json`. Contains driver championship standings with fields such as position, points, wins, podiums, team, nationality, car, engine, and various performance metrics.

> **Note:** The inspected `standings.json` file currently reports `"season": "2023"`, so it may be stale or placeholder data.

---

## Grand Prix Inventory by Season

### 2021 (22 events)

- Abu Dhabi Grand Prix
- Austrian Grand Prix
- Azerbaijan Grand Prix
- Bahrain Grand Prix
- Belgian Grand Prix
- British Grand Prix
- Dutch Grand Prix
- Emilia Romagna Grand Prix
- French Grand Prix
- Hungarian Grand Prix
- Italian Grand Prix
- Mexico City Grand Prix
- Monaco Grand Prix
- Portuguese Grand Prix
- Qatar Grand Prix
- Russian Grand Prix
- São Paulo Grand Prix
- Saudi Arabian Grand Prix
- Spanish Grand Prix
- Styrian Grand Prix
- Turkish Grand Prix
- United States Grand Prix

### 2022 (22 events + Pre-Season Test)

- Abu Dhabi Grand Prix
- Australian Grand Prix
- Austrian Grand Prix
- Azerbaijan Grand Prix
- Bahrain Grand Prix
- Belgian Grand Prix
- British Grand Prix
- Canadian Grand Prix
- Dutch Grand Prix
- Emilia Romagna Grand Prix
- French Grand Prix
- Hungarian Grand Prix
- Italian Grand Prix
- Japanese Grand Prix
- Mexico City Grand Prix
- Miami Grand Prix
- Monaco Grand Prix
- Pre-Season Test
- São Paulo Grand Prix
- Saudi Arabian Grand Prix
- Singapore Grand Prix
- Spanish Grand Prix
- United States Grand Prix

### 2023 (22 events + Pre-Season Testing)

- Abu Dhabi Grand Prix
- Australian Grand Prix
- Austrian Grand Prix
- Azerbaijan Grand Prix
- Bahrain Grand Prix
- Belgian Grand Prix
- British Grand Prix
- Canadian Grand Prix
- Dutch Grand Prix
- Hungarian Grand Prix
- Italian Grand Prix
- Japanese Grand Prix
- Las Vegas Grand Prix
- Mexico City Grand Prix
- Miami Grand Prix
- Monaco Grand Prix
- Pre-Season Testing
- Qatar Grand Prix
- Saudi Arabian Grand Prix
- Singapore Grand Prix
- Spanish Grand Prix
- São Paulo Grand Prix
- United States Grand Prix

### 2024 (23 events + Pre-Season Testing)

- Abu Dhabi Grand Prix
- Australian Grand Prix
- Austrian Grand Prix
- Azerbaijan Grand Prix
- Bahrain Grand Prix
- Belgian Grand Prix
- British Grand Prix
- Canadian Grand Prix
- Chinese Grand Prix
- Dutch Grand Prix
- Emilia Romagna Grand Prix
- Hungarian Grand Prix
- Italian Grand Prix
- Japanese Grand Prix
- Las Vegas Grand Prix
- Mexico City Grand Prix
- Miami Grand Prix
- Monaco Grand Prix
- Pre-Season Testing
- Qatar Grand Prix
- Saudi Arabian Grand Prix
- Singapore Grand Prix
- Spanish Grand Prix
- São Paulo Grand Prix
- United States Grand Prix

### 2025 (23 events + Pre-Season Testing)

- Abu Dhabi Grand Prix
- Australian Grand Prix
- Austrian Grand Prix
- Azerbaijan Grand Prix
- Bahrain Grand Prix
- Belgian Grand Prix
- British Grand Prix
- Canadian Grand Prix
- Chinese Grand Prix
- Dutch Grand Prix
- Emilia Romagna Grand Prix
- Hungarian Grand Prix
- Italian Grand Prix
- Japanese Grand Prix
- Las Vegas Grand Prix
- Mexico City Grand Prix
- Miami Grand Prix
- Monaco Grand Prix
- Pre-Season Testing
- Qatar Grand Prix
- Saudi Arabian Grand Prix
- Singapore Grand Prix
- Spanish Grand Prix
- São Paulo Grand Prix
- United States Grand Prix

> **Note:** Sprint sessions (Sprint, Sprint Qualifying) and alternative practice schedules are represented where applicable, typically as additional subfolders under each event.

---

## Sessions Available per Event

Most events contain up to five session folders:

- `Practice 1/`
- `Practice 2/`
- `Practice 3/`
- `Qualifying/`
- `Race/`

Sprint-format weekends may also include:

- `Sprint/` or `Sprint Race/`
- `Sprint Qualifying/`

Pre-season testing folders may use alternate session names.

---

## Python Scripts and Utilities

Each season folder includes Python scripts used to extract and process the data. Common scripts across seasons include:

| Script | Purpose |
|--------|---------|
| `tel.py` | General telemetry extraction |
| `telFP1.py`, `telFP2.py`, `telFP3.py` | Telemetry extraction for specific practice sessions |
| `telQ.py` | Telemetry extraction for Qualifying |
| `telR.py` | Telemetry extraction for Race |
| `telSQ.py`, `telSR.py` | Telemetry extraction for Sprint Qualifying / Sprint Race |
| `LapTimes.py` | Extracts per-driver lap times |
| `laptimesR.py` | Race-specific lap times extraction |
| `SessionLapTimes.py` | Merges per-driver lap times into `session_laptimes.json` |
| `corners.py` / `new_corners.py` | Corner data extraction |
| `drivers_corners.py` | Driver-specific corner analysis |
| `average_race_pace.py` | Computes average race pace |
| `fastest_lap.py` | Fastest lap analysis |
| `strategy.py` | Strategy analysis |
| `overtakes.py` | Overtake detection |
| `top_speeds.py` | Top speed extraction |
| `lap_position.py` | Lap position analysis |
| `ergast_client.py` | Client for Ergast/Jolpica F1 API |
| `utils.py` | Helper utilities (event/session lists, team colors) |
| `bot.ipynb` / `tel.ipynb` | Jupyter notebooks for exploration |
| `preseason.py` | Pre-season test data extraction |

Some years include additional notebooks (e.g., `accelerations.ipynb`, `corners.ipynb`) and year-specific variants of the corner scripts.

---

## Cache and Generated Files

| Folder / File | Description |
|---------------|-------------|
| `cache/` | FastF1 cache directory (small metadata) |
| `cache_preseason/` | Cache for pre-season sessions (2022) |
| `cache_joblib/` | 2025 only; contains ~34k additional JSON files (~1.91 GB) |
| `fastf1_cache/` | 2025 only; FastF1 persistent cache including `fastf1_http_cache.sqlite` |
| `requirements.txt` | Python dependencies |
| `README.md` | Repository readme (2025 contains detailed documentation) |
| `.git/` | Full Git history for the season repository |

---

## File Type Breakdown

Across all five seasons, the usable data is overwhelmingly JSON:

| Extension | Purpose | Approximate Volume |
|-----------|---------|-------------------:|
| `.json` | Telemetry, lap times, weather, corners, race control, standings | ~367k files, ~57 GB |
| `.py` | Extraction and analysis scripts | ~100 files |
| `.ipynb` | Jupyter notebooks | ~7 files |
| `.sqlite` | FastF1 HTTP cache database (2025) | 1 file |
| `.ff1pkl` | FastF1 pickle cache files (2025) | 10 files |
| `.log` | Extraction logs | 4 files |
| `.yml` / `.yaml` | GitHub Actions / DevContainer configs | ~100 files |
| `.md` | README files | 5 files |
| `.txt` | Requirements / license text | ~15 files |
| (none / `.git` internals) | Git pack files and metadata | ~151 GB total on disk |

---

## Data Source and Licensing

The Formula 1 datasets in this project were extracted from **TracingInsights**:

- **Data source:** <https://tracinginsights.com/data/>

According to the 2025 `README.md`, data is collected from:

- **FastF1** — official F1 timing and telemetry feeds
- **Ergast / Jolpica API** — historical results and statistics
- **OpenF1** — additional timing segments
- **F1 Live Timing feeds** — session metadata

License: **Apache-2.0** (see `LICENSE` / `LICENSE.md` in each season folder).

> This project is unofficial and not affiliated with Formula 1, FIA, or any F1 team.

---

## Notes and Observations

- The data is dense: a single race session for one driver can contain 50+ lap telemetry files, and each telemetry file can contain thousands of high-frequency samples.
- The 2025 season folder contains the most files because it includes large `cache_joblib/` and `fastf1_cache/` directories in addition to the usual JSON data.
- Some year folders contain stale or test files (e.g., `2025/standings.json` referencing the 2023 season).
- Because each season is an independent Git clone, the `.git/` folders dominate disk usage. Removing or shallow-cloning them in the future would reduce storage by roughly half.

---

*Generated on 2026-08-24.*
