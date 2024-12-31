#VisionLock - A Machine Learning Project

Dieses Projekt widmet sich der Entwicklung eines KI-basierten Gesichtserkennungssystems, das auf einem Raspberry Pi 5 mit einem Kamera-Modul implementiert wurde. Ziel war es, ein funktionales, kosteneffizientes und praxisnahes System zu schaffen, das beispielsweise für Sicherheits- und Zugangskontrollanwendungen eingesetzt werden kann.  

## **Funktionen**  
- **Echtzeit-Gesichtserkennung**: Identifiziert Gesichter in Live-Video-Feeds.  
- **Benutzerfreundliche Erweiterung**: Unbekannte Gesichter können hinzugefügt werden, um das System dynamisch zu aktualisieren.  
- **Kosteneffiziente Hardware**: Läuft auf einem Raspberry Pi 5 mit Camera Module 3.  
- **Flexibilität**: Kann für verschiedene Anwendungen angepasst werden, z. B. private Sicherheitslösungen oder kommerzielle Zugangskontrollen.  

## **Technologien**  
- **Hardware**: Raspberry Pi 5, Camera Module 3  
- **Software**:  
  - Python  
  - OpenCV: Bildvorverarbeitung und Darstellung  
  - face_recognition (dlib): Gesichtserkennung und Embedding-Generierung  
  - Machine Learning-Modell basierend auf der ResNet-34-Architektur  
- **Tools**: Code wurde größtenteils mit [bolt.new](https://bolt.new) erstellt.  

## **Funktionsweise**  
1. **Erfassen der Videoframes**: Die Kamera nimmt Frames in Echtzeit auf.  
2. **Vorverarbeitung**: Konvertierung des Bildes in das RGB-Farbschema und Anpassung der Auflösung für bessere Leistung.  
3. **Gesichtserkennung**: Mithilfe von face_recognition werden Gesichter erkannt und ihre 128-dimensionalen Embeddings erzeugt.  
4. **Vergleich mit bekannten Gesichtern**: Die Embeddings werden mit gespeicherten Daten abgeglichen.  
5. **Aktionen basierend auf Erkennung**:  
   - **Bekanntes Gesicht**: Zugriff gewährt.  
   - **Unbekanntes Gesicht**: Popup-Alarm und Möglichkeit zur Datensatz-Erweiterung.  

## **Ergebnisse und Ausblick**  
Tests zeigten, dass das System zuverlässig Gesichter erkennt. Potenzielle Weiterentwicklungen umfassen:  
- Integration zusätzlicher Funktionen wie Emotionserkennung.  
- Optimierung der Hardware-Performance für höhere FPS.  
- Anwendung in kommerziellen Projekten, z. B. in Zusammenarbeit mit einem lokalen Restaurant.  

## **Installation und Nutzung**  
1. **Abhängigkeiten installieren**:  
   ```bash
   pip install opencv-python face_recognition
