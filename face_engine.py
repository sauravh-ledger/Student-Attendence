"""
face_engine.py  -  the "brain" of the attendance system.

It does 3 jobs:
  1. DETECT  : find every face in a photo           (YuNet model)
  2. ENCODE  : turn each face into 128 numbers      (SFace model)
  3. MATCH   : compare those numbers with the saved students

Both models are small files inside the  models/  folder.
"""

import os
import re
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STUDENTS_DIR = os.path.join(BASE_DIR, "students")
RECORDS_DIR = os.path.join(BASE_DIR, "attendance_records")

DETECTOR_FILE = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_FILE = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
MODEL_URLS = {
    DETECTOR_FILE: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                   "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_FILE: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                     "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# How similar two faces must be (0 to 1) to count as the same person.
# 0.36 is the value recommended by the SFace model's authors.
DEFAULT_THRESHOLD = 0.36

# Photos bigger than this (in pixels on the long side) are shrunk to keep things fast.
MAX_SIDE = 2500

for folder in (MODELS_DIR, STUDENTS_DIR, RECORDS_DIR):
    os.makedirs(folder, exist_ok=True)


# ---------------------------------------------------------------- models ----
def _ensure_models():
    """Download the two model files if they are missing (only happens once)."""
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path) or os.path.getsize(path) < 100_000:
            print(f"Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)


class FaceEngine:
    def __init__(self):
        _ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            DETECTOR_FILE, "", (320, 320),
            score_threshold=0.7,  # how sure it must be that something is a face
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")

    # ----------------------------------------------------------- detect ----
    def detect(self, img_bgr):
        """Return a list of faces. Each face is a row of 15 numbers:
        x, y, width, height, 10 landmark numbers (eyes, nose, mouth), score."""
        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_bgr)
        faces = [] if faces is None else list(faces)

        # A class photo taken from far away has tiny faces. If we found very few,
        # try again on a 2x bigger copy of the photo.
        if len(faces) < 3 and max(h, w) < 1600:
            big = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            self.detector.setInputSize((w * 2, h * 2))
            _, big_faces = self.detector.detect(big)
            if big_faces is not None and len(big_faces) > len(faces):
                faces = [f.copy() for f in big_faces]
                for f in faces:
                    f[:14] /= 2.0  # scale coordinates back to the original photo
        return faces

    # ----------------------------------------------------------- encode ----
    def encode(self, img_bgr, face):
        """Turn one face into a list of 128 numbers (its 'face encoding')."""
        aligned = self.recognizer.alignCrop(img_bgr, face)  # straighten + crop the face
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-10)  # normalise, so we can use dot product

    def encode_single_face(self, img_bgr):
        """For registration photos: encode the biggest face in the photo."""
        faces = self.detect(img_bgr)
        if not faces:
            return None
        biggest = max(faces, key=lambda f: f[2] * f[3])
        return self.encode(img_bgr, biggest)


# ------------------------------------------------------------- images ------
def load_image(file_or_path):
    """Open a photo (from a path or an uploaded file) as an OpenCV BGR image.
    Phone photos are rotated correctly using their EXIF info."""
    img = Image.open(file_or_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return arr


# ------------------------------------------------------- student data ------
def student_folder(roll_no, name):
    safe_name = re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
    safe_roll = re.sub(r"[^A-Za-z0-9-]", "", str(roll_no)).strip()  # no "_" : it separates roll and name
    return os.path.join(STUDENTS_DIR, f"{safe_roll}_{safe_name}")


def list_students():
    """Read the students/ folder. Each sub-folder is one student: '<roll>_<name>'."""
    students = []
    for folder in sorted(os.listdir(STUDENTS_DIR)):
        path = os.path.join(STUDENTS_DIR, folder)
        if not os.path.isdir(path):
            continue
        roll, _, name = folder.partition("_")
        photos = [os.path.join(path, f) for f in sorted(os.listdir(path))
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        students.append({"roll_no": roll, "name": name or roll, "folder": path, "photos": photos})
    return students


def build_database(engine, students):
    """Encode every registration photo. Returns {roll_no: [encodings]}."""
    db = {}
    for s in students:
        encs = []
        for p in s["photos"]:
            try:
                enc = engine.encode_single_face(load_image(p))
            except Exception:
                enc = None
            if enc is not None:
                encs.append(enc)
        if encs:
            db[s["roll_no"]] = np.stack(encs)
    return db


# -------------------------------------------------------------- match ------
def match_faces(engine, img_bgr, db, threshold=DEFAULT_THRESHOLD):
    """Find all faces in a class photo and decide who each one is.

    Returns (faces, assignment):
      faces      -> list of detected faces
      assignment -> {face_index: (roll_no, similarity)} for recognised faces
    Each student can be matched to at most ONE face (the most similar one).
    """
    faces = engine.detect(img_bgr)
    if not faces or not db:
        return faces, {}

    rolls = list(db.keys())
    encodings = np.stack([engine.encode(img_bgr, f) for f in faces])  # (num_faces, 128)

    # similarity[i][j] = how similar face i is to student j (best of their photos)
    similarity = np.zeros((len(faces), len(rolls)), dtype=np.float32)
    for j, r in enumerate(rolls):
        similarity[:, j] = (encodings @ db[r].T).max(axis=1)

    # Greedy matching: take the most similar (face, student) pair first,
    # so the same student is never given to two different faces.
    assignment, used_faces, used_students = {}, set(), set()
    pairs = sorted(((similarity[i, j], i, j) for i in range(len(faces)) for j in range(len(rolls))),
                   reverse=True)
    for sim, i, j in pairs:
        if sim < threshold:
            break
        if i in used_faces or j in used_students:
            continue
        assignment[i] = (rolls[j], float(sim))
        used_faces.add(i)
        used_students.add(j)
    return faces, assignment


def draw_results(img_bgr, faces, assignment, names):
    """Draw a green box + name on recognised faces, red box on unknown faces."""
    out = img_bgr.copy()
    thick = max(2, out.shape[1] // 600)
    font_scale = max(0.5, out.shape[1] / 1800)
    for i, f in enumerate(faces):
        x, y, w, h = [int(v) for v in f[:4]]
        if i in assignment:
            roll, sim = assignment[i]
            color, label = (60, 180, 60), f"{names.get(roll, roll)} ({sim:.2f})"
        else:
            color, label = (40, 40, 220), "Unknown"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(th + 6, y - 4)
        cv2.rectangle(out, (x, ty - th - 6), (x + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out
