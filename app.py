"""
app.py  -  the web page for the Face Recognition Attendance System.

Run it with:   streamlit run app.py
It opens in your browser. There are 3 tabs:
  1. Register Students  -> add each student with 2-5 clear face photos
  2. Take Attendance    -> upload a class photo, check the result, save it
  3. Records            -> see saved attendance and each student's %
"""

import os
import shutil
from datetime import date

import cv2
import pandas as pd
import streamlit as st

import face_engine as fe

st.set_page_config(page_title="Face Attendance", page_icon="📸", layout="wide")


# ------------------------------------------------------------- helpers -----
@st.cache_resource(show_spinner="Loading face models...")
def get_engine():
    return fe.FaceEngine()


@st.cache_resource(show_spinner="Reading student photos...")
def get_database(signature):
    # 'signature' changes whenever photos are added/removed, which refreshes this cache.
    return fe.build_database(get_engine(), fe.list_students())


def students_signature(students):
    return tuple((s["roll_no"], len(s["photos"]),
                  max((os.path.getmtime(p) for p in s["photos"]), default=0)) for s in students)


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


engine = get_engine()
students = fe.list_students()
db = get_database(students_signature(students))
names = {s["roll_no"]: s["name"] for s in students}

# ------------------------------------------------------------- sidebar -----
st.sidebar.title("📸 Face Attendance")
st.sidebar.metric("Registered students", len(students))
threshold = st.sidebar.slider(
    "Match strictness", 0.25, 0.60, fe.DEFAULT_THRESHOLD, 0.01,
    help="Higher = stricter. If wrong people get marked, raise it. "
         "If real students are missed, lower it a little.")
st.sidebar.caption("Tip: take the class photo in good light, with everyone facing the camera. "
                   "For a big class, take 2-3 photos of different rows.")

tab_register, tab_attend, tab_records = st.tabs(
    ["👤 Register Students", "✅ Take Attendance", "📊 Records"])

# ============================================================ REGISTER =====
with tab_register:
    st.header("Register a student")
    st.write("Upload **2 to 5 clear photos** of the student's face "
             "(front-facing, good light, different days or angles if possible).")

    c1, c2 = st.columns(2)
    roll_no = c1.text_input("Roll number", placeholder="e.g. 23CSE101")
    name = c2.text_input("Full name", placeholder="e.g. Rahul Sharma")
    photos = st.file_uploader("Face photos", type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=True, key="reg_photos")

    if st.button("Save student", type="primary"):
        if not roll_no.strip() or not name.strip():
            st.error("Please enter both roll number and name.")
        elif not photos:
            st.error("Please upload at least one photo.")
        elif any(s["roll_no"] == os.path.basename(fe.student_folder(roll_no, name)).partition("_")[0]
                 for s in students):
            st.error("A student with this roll number already exists. Delete them first to re-register.")
        else:
            folder = fe.student_folder(roll_no, name)
            os.makedirs(folder, exist_ok=True)
            saved = 0
            for i, up in enumerate(photos):
                img = fe.load_image(up)
                faces = engine.detect(img)
                if not faces:
                    st.warning(f"No face found in **{up.name}**, skipped it.")
                    continue
                if len(faces) > 1:
                    st.info(f"{up.name} has {len(faces)} faces, using the biggest one.")
                cv2.imwrite(os.path.join(folder, f"photo_{i + 1}.jpg"), img)
                saved += 1
            if saved == 0:
                shutil.rmtree(folder, ignore_errors=True)
                st.error("None of the photos had a clear face. Try different photos.")
            else:
                st.success(f"Saved {name} with {saved} photo(s).")
                if saved < 2:
                    st.info("Adding at least 2 photos makes recognition more reliable.")
                st.rerun()

    st.divider()
    st.subheader(f"Registered students ({len(students)})")
    if not students:
        st.info("No students yet. Add your first one above.")
    else:
        st.dataframe(pd.DataFrame([{"Roll No": s["roll_no"], "Name": s["name"],
                                    "Photos": len(s["photos"]),
                                    "Ready": "✅" if s["roll_no"] in db else "⚠️ no face found"}
                                   for s in students]), hide_index=True)
        with st.expander("Delete a student"):
            labels = {f'{s["roll_no"]} - {s["name"]}': s for s in students}
            pick = st.selectbox("Student", list(labels.keys()))
            if st.button("Delete"):
                shutil.rmtree(labels[pick]["folder"], ignore_errors=True)
                st.success(f"Deleted {pick}")
                st.rerun()

# ============================================================== ATTEND =====
with tab_attend:
    st.header("Take attendance from a class photo")
    if not db:
        st.warning("Register some students first (tab 1).")
    else:
        c1, c2 = st.columns(2)
        subject = c1.text_input("Subject / lecture", value="General")
        day = c2.date_input("Date", value=date.today())
        class_photos = st.file_uploader("Class photo(s)", type=["jpg", "jpeg", "png", "webp"],
                                        accept_multiple_files=True, key="class_photos")

        if st.button("Find students", type="primary", disabled=not class_photos):
            best = {}  # roll_no -> best similarity across all photos
            annotated, total_faces = [], 0
            with st.spinner("Detecting and recognising faces..."):
                for up in class_photos:
                    img = fe.load_image(up)
                    faces, assignment = fe.match_faces(engine, img, db, threshold)
                    total_faces += len(faces)
                    for roll, sim in assignment.values():
                        best[roll] = max(sim, best.get(roll, 0))
                    annotated.append((up.name, to_rgb(fe.draw_results(img, faces, assignment, names)),
                                      len(faces), len(assignment)))
            st.session_state["result"] = {
                "table": pd.DataFrame([{
                    "Roll No": s["roll_no"], "Name": s["name"],
                    "Present": s["roll_no"] in best,
                    "Match score": round(best[s["roll_no"]], 2) if s["roll_no"] in best else None,
                } for s in students]),
                "annotated": annotated, "faces": total_faces,
            }

        res = st.session_state.get("result")
        if res:
            for fname, img, n_faces, n_matched in res["annotated"]:
                st.image(img, caption=f"{fname}: {n_faces} faces found, {n_matched} recognised "
                                      f"(green = recognised, red = unknown)")

            st.subheader("Check and correct")
            st.caption("Tick or untick **Present** if the system made a mistake, then save.")
            edited = st.data_editor(res["table"], hide_index=True,
                                    disabled=["Roll No", "Name", "Match score"], key="editor")
            present = int(edited["Present"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Present", f"{present} / {len(edited)}")
            m2.metric("Absent", len(edited) - present)
            m3.metric("Faces in photo(s)", res["faces"])

            out = edited[["Roll No", "Name"]].copy()
            out["Status"] = edited["Present"].map({True: "Present", False: "Absent"})
            out.insert(0, "Subject", subject)
            out.insert(0, "Date", day.isoformat())
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            safe_subject = "".join(ch for ch in subject if ch.isalnum() or ch in " -_").strip() or "class"
            filename = f"{day.isoformat()}_{safe_subject}.csv"

            b1, b2 = st.columns(2)
            if b1.button("💾 Save attendance", type="primary"):
                out.to_csv(os.path.join(fe.RECORDS_DIR, filename), index=False)
                st.success(f"Saved as attendance_records/{filename}")
            b2.download_button("⬇️ Download CSV (opens in Excel)", csv_bytes, filename, "text/csv")

# ============================================================= RECORDS =====
with tab_records:
    st.header("Saved attendance")
    files = sorted([f for f in os.listdir(fe.RECORDS_DIR) if f.endswith(".csv")], reverse=True)
    if not files:
        st.info("Nothing saved yet.")
    else:
        all_df = pd.concat([pd.read_csv(os.path.join(fe.RECORDS_DIR, f), dtype=str) for f in files])
        summary = (all_df.assign(present=all_df["Status"].eq("Present"))
                   .groupby(["Roll No", "Name"])["present"]
                   .agg(Classes="count", Attended="sum").reset_index())
        summary["Attendance %"] = (100 * summary["Attended"] / summary["Classes"]).round(1)
        st.subheader("Attendance % per student")
        st.dataframe(summary.sort_values("Roll No"), hide_index=True)
        low = summary[summary["Attendance %"] < 75]
        if len(low):
            st.warning(f"{len(low)} student(s) below 75%: " + ", ".join(low["Name"]))
        st.download_button("⬇️ Download summary", summary.to_csv(index=False).encode(),
                           "attendance_summary.csv", "text/csv")

        st.subheader("Single class")
        pick = st.selectbox("Record", files)
        st.dataframe(pd.read_csv(os.path.join(fe.RECORDS_DIR, pick)), hide_index=True)
