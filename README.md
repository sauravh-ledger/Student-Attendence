# 📸 Face Recognition Attendance System

Upload **one photo of the whole class** → the app finds every face, recognises the
registered students, and marks attendance. You can correct mistakes and download it as Excel/CSV.

## How to run (first time)

1. Install **Python 3.10–3.13** from https://www.python.org/downloads/
   - On Windows, tick **"Add Python to PATH"** during install.
2. Unzip this folder anywhere (e.g. Desktop).
3. Run it:
   - **Windows:** double-click `run_windows.bat`
   - **Mac/Linux:** open Terminal in this folder and run `./run_mac_linux.sh`
4. The first run installs libraries (takes a few minutes). Then the app opens in your browser
   at `http://localhost:8501`. Next time it starts in seconds.

To stop the app, close the black terminal window.

## How to use

1. **Register Students** tab → enter roll number + name, upload 2–5 clear face photos, click *Save*.
2. **Take Attendance** tab → upload the class photo(s), click *Find students*.
   - Green box = recognised student, red box = unknown face.
   - Tick/untick *Present* to fix any mistakes, then *Save* or *Download CSV*.
3. **Records** tab → every saved class, plus each student's attendance % (warns below 75%).

**Match strictness** (left sidebar): if wrong people get marked present, move it up;
if real students are missed, move it down a little.

## How it works

| Step | What happens | Model |
|---|---|---|
| 1. Detect | Find every face in the photo | YuNet (OpenCV) |
| 2. Align | Straighten each face using eye/nose/mouth points | — |
| 3. Encode | Turn each face into 128 numbers (a "face encoding") | SFace (OpenCV) |
| 4. Match | Compare with saved students using cosine similarity; each student matched to at most one face | — |
| 5. Save | Write attendance to CSV | pandas |

## Files

- `app.py` — the web page (Streamlit)
- `face_engine.py` — face detection, encoding and matching
- `models/` — the two AI model files
- `students/` — one folder per registered student (`<roll>_<name>/`)
- `attendance_records/` — saved attendance CSVs

## Tips for good results

- Good light, students facing the camera, no hands/masks over faces.
- For big classes, take 2–3 photos (front rows, back rows) and upload them together.
- Registration photos: clear, front-facing, ideally taken on different days.

## Privacy

This stores students' face photos. Get their consent, keep the folder private,
and delete a student's data when they ask (Register tab → *Delete a student*).
