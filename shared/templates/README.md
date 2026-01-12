# Interaktiver Bestellungs-Scanner

Web-basiertes Tool zur OCR-Erkennung von handschriftlichen Bestellmengen aus Formular-PDFs.

## Übersicht

Der Scanner verarbeitet E-Mail-Anhänge (.eml mit PDF) und erkennt handgeschriebene Ziffern in Bestellformularen. Er läuft als Web-Anwendung auf dem Jetson und nutzt den `ml-inference` Docker-Container für die GPU-beschleunigte Ziffernerkennung.

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (Frontend)                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Datei laden    │→ │  Zonen          │→ │  Ergebnisse     │ │
│  │  (.eml / Bild)  │  │  definieren     │  │  prüfen/export  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
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
http://192.168.3.241:8080/
```

## Workflow

### 1. Datei laden
- Klick auf **"1. Datei laden"**
- Unterstützt: `.eml` (E-Mail mit PDF-Anhang) oder direkt Bilder
- Bei E-Mails: PDF wird automatisch extrahiert und in Bild konvertiert (300 DPI)
- Automatische Orientierungskorrektur und Deskew

### 2. Zonen definieren
- **Zone auto-erkennen**: Erkennt Tabellenstruktur automatisch
- Oder manuell zeichnen:
  - **Bestellmengen (cyan)**: Bereich mit Mo-Fr Spalten
  - **Produktnamen (grün)**: Spalte mit Produktbezeichnungen
  - **Produktnummern (orange)**: Spalte mit Artikelnummern
- Zeilen/Spalten-Anzahl anpassen falls nötig

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

### 5. Export
- **Export JSON**: Speichert alle Bestellungen als JSON-Datei

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
- EasyOCR für Produktnamen läuft im CPU-Modus (Speicher-Probleme mit GPU)
- Sehr schlechte Handschrift kann zu niedrigen Konfidenzwerten führen
