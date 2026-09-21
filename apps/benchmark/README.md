# DiskANN Benchmark Suite (Windows)

Diese Dokumentation beschreibt, wie das Benchmark-Tool `diskann_build_and_test` unter Windows gebaut und ausgeführt wird.

---

## Voraussetzungen

1. **Visual Studio 2022** (mit C++ Desktop Development Workload).
2. **CMake** (in PATH verfügbar).
3. **NuGet CLI** (`nuget.exe`):
   DiskANN lädt unter Windows automatisch Abhängigkeiten wie **Boost**, **Intel OpenMP** und **Intel MKL** via NuGet herunter.
   Falls `nuget.exe` nicht im System-PATH ist, kann es direkt in das DiskANN-Wurzelverzeichnis (`C:\Lang\cpp\DiskANN\nuget.exe`) gelegt werden:
   ```powershell
   Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile "C:\Lang\cpp\DiskANN\nuget.exe"
   ```

---

## Bauen (Kompilieren)

Führe folgende Schritte im Wurzelverzeichnis von DiskANN (`C:\Lang\cpp\DiskANN`) in einer PowerShell oder im **x64 Native Tools Command Prompt für VS 2022** aus:

### 1. CMake konfigurieren
```powershell
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DNUGET_EXE=C:\Lang\cpp\DiskANN\nuget.exe
```
> **Hinweis:** Beim ersten Ausführen werden automatisch die NuGet-Pakete (Boost, Intel MKL, OpenMP) in `build/packages` heruntergeladen.

### 2. Binary im Release-Modus bauen
```powershell
cmake --build build --config Release --target diskann_build_and_test -j 8
```
Die fertigen Binaries und abhängigen DLLs (`diskann.dll`, `libiomp5md.dll`, `libtcmalloc_minimal.dll`, `diskann_build_and_test.exe`) befinden sich nach dem Build in:
```
x64\Release\
```

---

## Ausführen des Benchmarks

Das Benchmark-Tool erwartet:
1. Den Namen des Datensatzes (`audio`, `sift1m`, `deep1m`, `glove`, `enron` oder `all`).
2. Den Root-Pfad zu den Daten (z. B. `D:\Data\DEG`).
3. Optionale Flags:
   - `-T <threads>` bzw. `--num_threads <threads>`: Anzahl der Worker-Threads.
   - `-f` bzw. `--force-test`: Bereits vorhandene Indizes/Logs überschreiben und neu berechnen.

### Beispiel: Audio Dataset
```powershell
.\x64\Release\diskann_build_and_test.exe audio D:\Data\DEG -T 8
```

Mit Überschreiben / Erzwingen:
```powershell
.\x64\Release\diskann_build_and_test.exe audio D:\Data\DEG -T 8 --force-test
```

---

## Erwartete Ordnerstruktur der Datensätze

Für einen Datensatz `<dataset>` (z. B. `audio`) sucht das Tool unter:
```
<data_root>\<dataset>\<dataset>\
```
Für Audio unter `D:\Data\DEG`:
```
D:\Data\DEG\audio\audio\
  ├── audio_base.fvecs
  ├── audio_query.fvecs
  ├── audio_explore_query.fvecs
  ├── audio_explore_entry_vertex.ivecs
  ├── audio_explore_groundtruth_top1000.ivecs
  └── audio_groundtruth_top100_nb53387.ivecs (bzw. nb26693)
```

---

## Ausgabe / Ergebnisdateien

Die erzeugten Indizes und Benchmark-Ergebnisse werden im Verzeichnis `<data_root>\<dataset>\diskann\` gespeichert:

- **Statischer Index**:
  - `diskann_R64_L125` (Graphdatei)
  - `diskann_R64_L125.data` (Vektordaten)
  - `diskann_R64_L125.tags` (Tag-Mappings)
  - `diskann_R64_L125_benchmark.log` (Graph-Statistiken, ANNS Recall/QPS, Exploration-Ergebnisse)

- **Dynamische Indizes** (`<data_root>\<dataset>\diskann\dynamic\`):
  - `diskann_R64_L125_AddHalf.da` (+ `.data`, `.tags`, `_benchmark.log`)
  - `diskann_R64_L125_AddAllRemoveHalf.da` (+ `.data`, `.tags`, `_benchmark.log`)
  - `diskann_R64_L125_AddHalfRemoveAndAddOneAtATime.da` (+ `.data`, `.tags`, `_benchmark.log`)
