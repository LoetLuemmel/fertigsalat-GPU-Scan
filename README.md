# Email-Processor Pipeline

Multi-Container-System zur automatischen Online-Verarbeitung von Bestellformularen aus E-Mail-Anhängen.

<img width="1662" height="120" alt="image" src="https://github.com/user-attachments/assets/1e9da693-b250-4388-87ea-32e524e6caa9" />

Das System ist unter dieser Domain live: https://foxera.390er.de

Die URL ist eine "Altlast" und um Aufwendungen gering zu halten, wurde die Foxera App hier gehostet. Mittelfristig soll sie sicherlich auf die fertigsalat.ch Domain umziehen oder wird nur intern ggenutzt.


<img width="1920" height="976" alt="image" src="https://github.com/user-attachments/assets/080a5ba1-0445-44e4-9cda-2a5f26516e09" />


## Architektur

```
Input (.eml) → Orchestrator → OpenCV-Prep → Layout-Analyzer → ML-Inference → Output
                    ↓              ↓              ↓               ↓
              shared/input   shared/intermediate              shared/output
```

### Container

| Container | Image | Funktion |
|-----------|-------|----------|
| `orchestrator` | python:3.11-slim | Workflow-Steuerung, Datei-Watcher |
| `opencv-prep` | python:3.11-slim | EML→PDF→Bild, Deskew, Denoise, Crop |
| `layout-analyzer` | python:3.11-slim | Formularfelder, Linien, Boxen erkennen |
| `ml-inference` | nvcr.io/nvidia/l4t-ml:r36.2.0-py3 | EasyOCR mit GPU-Unterstützung |

## Schnellstart

```bash
# Alle Container bauen
docker compose build

# Container starten
docker compose up -d

# Logs anzeigen
docker compose logs -f
```

<img width="1231" height="85" alt="image" src="https://github.com/user-attachments/assets/7694752b-4069-463f-83af-1441f419e9ec" />


## Projektstruktur

```
email-processor/
├── docker-compose.yml
├── orchestrator/          # Workflow-Steuerung
├── opencv-prep/           # Bildvorverarbeitung
├── layout-analyzer/       # Formular-Erkennung
├── ml-inference/          # OCR mit GPU
└── shared/
    ├── input/             # Eingehende .eml Dateien
    ├── intermediate/      # Temporäre Verarbeitungsdaten
    ├── output/            # Ergebnisse (JSON)
    └── templates/         # Formularvorlagen, Scanner-Tool
```

---

# Interaktiver Bestellungs-Scanner

Web-basiertes Tool zur manuellen OCR-Erkennung und Korrektur.

## Übersicht

Der Scanner verarbeitet E-Mail-Anhänge (.eml mit PDF) und erkennt handgeschriebene Ziffern in Bestellformularen. Er läuft als Web-Anwendung auf dem Jetson und nutzt den `ml-inference` Docker-Container für die GPU-beschleunigte Ziffernerkennung.

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (Frontend)                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Datei laden    │→ │  Zonen          │→ │  Ergebnisse     │  │
│  │  (.eml / Bild)  │  │  definieren     │  │  prüfen/export  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP API
┌────────────────────────────▼────────────────────────────────────┐
│  scanner_server.py (Port 8080)                                  │
│  - /api/parse-eml      → PDF-Extraktion, Orientierung, Deskew   │
│  - /api/detect-zones   → Auto-Erkennung der Tabellenbereiche    │
│  - /api/scan-zone      → Ziffernerkennung via Docker            │
│  - /api/save-training-sample → Training-Daten sammeln           │
└────────────────────────────┬────────────────────────────────────┘
                             │ docker exec
┌────────────────────────────▼────────────────────────────────────┐
│  ml-inference Container (GPU)                                   │
│  - OpenCV Blob-Erkennung                                        │
│  - Trainierte Ziffernerkennung                                  │
│  - EasyOCR für Produktnamen                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Schnellstart

```bash
# Server starten
python3 /home/test/email-processor/shared/templates/scanner_server.py

# Im Browser öffnen
http://192.168.8.100:8080/
```

## Workflow

### 1. Datei laden
- Klick auf **"1. Datei laden"**
- Unterstützt: `.eml` (E-Mail mit PDF-Anhang) oder direkt Bilder
- Bei E-Mails: PDF wird automatisch extrahiert und in Bild konvertiert (300 DPI)
- Automatische Orientierungskorrektur und Deskew

<img width="426" height="108" alt="image" src="https://github.com/user-attachments/assets/a175b963-91f6-4c3e-83dd-1d99f37e7dcb" />


### 2. Zonen definieren
- **Zone auto-erkennen**: Erkennt Tabellenstruktur automatisch
- Oder manuell zeichnen:
  - **Bestellmengen (cyan)**: Bereich mit Mo-Fr Spalten
  - **Produktnamen (grün)**: Spalte mit Produktbezeichnungen
  - **Produktnummern (orange)**: Spalte mit Artikelnummern
- Zeilen/Spalten-Anzahl anpassen falls nötig

<img width="401" height="115" alt="image" src="https://github.com/user-attachments/assets/1859c582-78a0-49b5-853a-03f960980751" />


### 3. Scannen
- Klick auf **"2. Zone scannen"**
- Erkennt handgeschriebene Ziffern in jeder Zelle
- Liest Produktnamen und -nummern via EasyOCR

### 4. Ergebnisse prüfen
- Tabelle zeigt alle erkannten Bestellungen
- **Klick auf Zeile**: Markiert die Zelle im Bild
- **Doppelklick**: Öffnet Dialog zur Korrektur
- **Direkte Eingabe**: Menge im Eingabefeld ändern
- Korrigierte Werte werden als Training-Daten gespeichert

<img width="294" height="636" alt="image" src="https://github.com/user-attachments/assets/4bb84736-dcb3-4b86-9b32-dd4994bff790" />


### 5. Export
- **Export JSON**: Speichert alle Bestellungen als JSON-Datei

<img width="435" height="367" alt="image" src="https://github.com/user-attachments/assets/b3d1fd0e-73d3-4d16-bfa1-f829c37bb4a1" />


## API-Endpunkte

| Endpunkt | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/parse-eml` | POST | EML-Datei verarbeiten, PDF extrahieren |
| `/api/detect-zones` | POST | Tabellenstruktur automatisch erkennen |
| `/api/scan-zone` | POST | Zone scannen, Ziffern erkennen |
| `/api/read-product-codes` | POST | Produktnummern lesen |
| `/api/read-product-names` | POST | Produktnamen lesen |
| `/api/save-training-sample` | POST | Korrigierte Ziffer als Training speichern |
| `/api/training-stats` | GET | Anzahl Training-Samples pro Ziffer |
| `/api/gpu-status` | GET | GPU-Verfügbarkeit prüfen |
| `/api/image` | GET | Zuletzt verarbeitetes Bild |

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `interactive_scanner.html` | Frontend (HTML/CSS/JS) |
| `scanner_server.py` | Backend-Server mit allen APIs |
| `zone_editor.html` | Hilfs-Tool zum Definieren von Template-Zonen |
| `server.py` | Einfacher HTTP-Server für zone_editor |

## Verzeichnisse

| Pfad | Inhalt |
|------|--------|
| `shared/templates/` | Bestellblatt-Vorlagen, Scanner-Dateien |
| `shared/templates/definitions/` | JSON-Definitionen der Formulartypen |
| `shared/intermediate/` | Temporäre Bilder während Verarbeitung |
| `shared/training_data/` | Gesammelte Training-Samples (nach Ziffer) |

## Training-Daten

Korrigierte Erkennungen werden automatisch gespeichert:

<img width="265" height="95" alt="image" src="https://github.com/user-attachments/assets/b44ce84a-f096-442f-ba5c-47f584f6f790" />


```
shared/training_data/
├── 0/
│   ├── sample_1704812345678_12_Mo.png
│   └── ...
├── 1/
├── 2/
...
└── 9/
```

<img width="58" height="56" alt="image" src="https://github.com/user-attachments/assets/c421fe05-e935-484f-9fdd-7bd1e8fc2c64" />


Diese können für Finetuning des OCR-Modells verwendet werden.

## Farbkodierung im Bild

| Farbe | Bedeutung |
|-------|-----------|
| Cyan | Bestellmengen-Zone |
| Grün | Produktnamen-Zone / Geprüfte Erkennung |
| Orange | Produktnummern-Zone |
| Gelb | Erkannte Ziffer (Konfidenz >= 50%) |
| Rot | Unsichere Erkennung (Konfidenz < 50%) |
| Pulsierend | Aktuell ausgewählte Zelle |

<img width="664" height="302" alt="image" src="https://github.com/user-attachments/assets/250aeb1d-47a0-4356-9112-75bc41680be4" />


## Tastenkürzel

- **Klick auf Bild**: Wählt die entsprechende Zeile in der Tabelle
- **Doppelklick auf Tabellenzeile**: Öffnet Korrektur-Dialog
- **0-9 im Dialog**: Setzt die erkannte Ziffer

## Voraussetzungen

- Docker Container `email-ml-inference` muss laufen
- `pdftoppm` oder ImageMagick für PDF-Konvertierung
- Python 3.11+

## Bekannte Einschränkungen

- Nur erste Seite von mehrseitigen PDFs wird verarbeitet
- Sehr schlechte Handschrift kann zu niedrigen Konfidenzwerten führen
