# ============================================================
# EDU TRACK AI
# AI-Powered Student Success & Early Warning Platform
# FINAL HACKATHON VERSION
# ============================================================

import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

st.markdown("""
<style>
[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)
# Optional ML
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EduTrack AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
MARKS_FILE = os.path.join(DATA_DIR, "academic_marks.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- HEADER ---------- */

    .topbar {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 18px;
        padding: 14px 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .brand {
        font-size: 22px;
        font-weight: 800;
        color: #172554;
    }

    .muted {
        color: #64748b;
        font-size: 14px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #172554 0%,
            #2563eb 55%,
            #38bdf8 100%
        );
        color: white;
        border-radius: 26px;
        padding: 42px;
        margin-bottom: 25px;
        box-shadow: 0 18px 45px rgba(37, 99, 235, 0.18);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        font-size: 17px;
        margin: 0;
        max-width: 850px;
        opacity: 0.93;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 7px 22px rgba(15, 23, 42, 0.045);
    }

    .module-card {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 20px;
        padding: 22px;
        min-height: 150px;
        margin-bottom: 8px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        transition: 0.2s;
    }

    .module-card:hover {
        transform: translateY(-2px);
    }

    .module-icon {
        font-size: 34px;
    }

    .module-card h3 {
        margin: 7px 0 5px 0;
        color: #172554;
    }

    .module-card p {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }

    /* ---------- METRICS ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        min-height: 125px;
    }

    .metric-icon {
        font-size: 22px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        color: #172554;
        margin-top: 5px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #172554;
        margin: 25px 0 15px 0;
    }

    /* ---------- AI ---------- */

    .ai-card {
        background: linear-gradient(
            135deg,
            #eff6ff,
            #f5f3ff
        );
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 23px;
        margin-bottom: 20px;
    }

    /* ---------- STATUS ---------- */

    .status-good {
        color: #15803d;
        font-weight: 700;
    }

    .status-warning {
        color: #b45309;
        font-weight: 700;
    }

    .status-danger {
        color: #b91c1c;
        font-weight: 700;
    }

    /* ---------- NORMAL BUTTONS ---------- */

    div.stButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #dbe2ec;
    }

    /* ========================================================
       CIRCULAR ATTENDANCE BUTTONS
       ======================================================== */

    /* Normal neutral PRESENT circle */
    div[class*="st-key-att_present_"] button {
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        min-width: 42px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        color: transparent !important;
        box-shadow: none !important;
    }

    /* Selected PRESENT circle */
    div[class*="st-key-att_present_selected_"] button {
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        min-width: 42px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: #22c55e !important;
        border: 3px solid #15803d !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(34,197,94,.25) !important;
    }

    /* Normal neutral ABSENT circle */
    div[class*="st-key-att_absent_"] button {
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        min-width: 42px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        color: transparent !important;
        box-shadow: none !important;
    }

    /* Selected ABSENT circle */
    div[class*="st-key-att_absent_selected_"] button {
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        min-width: 42px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: #ef4444 !important;
        border: 3px solid #b91c1c !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(239,68,68,.25) !important;
    }

    /* Attendance row */
    .attendance-row {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 15px;
        padding: 12px 15px;
        margin-bottom: 8px;
    }

    .attendance-name {
        font-weight: 700;
        color: #172554;
        padding-top: 10px;
    }

    .attendance-id {
        color: #64748b;
        font-size: 13px;
    }

    .attendance-label {
        text-align: center;
        font-weight: 700;
        font-size: 13px;
        color: #475569;
        margin-bottom: 3px;
    }

    /* ---------- HIDE SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        display: none;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "role" not in st.session_state:
    st.session_state.role = None

if "module" not in st.session_state:
    st.session_state.module = None

if "selected_student" not in st.session_state:
    st.session_state.selected_student = None

if "attendance_choices" not in st.session_state:
    st.session_state.attendance_choices = {}

if "attendance_summary" not in st.session_state:
    st.session_state.attendance_summary = None


# ============================================================
# DATA HELPERS
# ============================================================

def find_column(df, names):
    """Find a column using exact or partial matching."""
    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for c in df.columns:
        clean = str(c).strip().lower()
        for name in names:
            if name.lower() in clean:
                return c

    return None


def numeric_series(df, names, default=70.0):
    col = find_column(df, names)

    if col is None:
        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(default)


# ============================================================
# STUDENT DATA
# ============================================================

def create_demo_students():
    """Create demo dataset if students.csv does not exist."""

    rng = np.random.default_rng(42)
    n = 120

    df = pd.DataFrame({
        "Student_ID": [
            f"STU{1000+i}"
            for i in range(n)
        ],

        "Student_Name": [
            f"Student {i+1}"
            for i in range(n)
        ],

        "CGPA": np.round(
            rng.uniform(5.2, 9.6, n),
            2
        ),

        "Attendance": np.round(
            rng.uniform(55, 98, n),
            1
        ),

        "LMS_Score": np.round(
            rng.uniform(45, 98, n),
            1
        ),

        "Engagement": np.round(
            rng.uniform(40, 98, n),
            1
        ),

        "Placement": np.round(
            rng.uniform(45, 96, n),
            1
        ),

        "Skills": np.round(
            rng.uniform(40, 96, n),
            1
        ),

        "Feedback": np.round(
            rng.uniform(45, 100, n),
            1
        )
    })

    df.to_csv(
        STUDENTS_FILE,
        index=False
    )

    return df


def load_students():
    if not os.path.exists(STUDENTS_FILE):
        return create_demo_students()

    try:
        df = pd.read_csv(STUDENTS_FILE)

        if df.empty:
            return create_demo_students()

        return df

    except Exception:
        return create_demo_students()


students = load_students()

IDCOL = find_column(
    students,
    [
        "student_id",
        "student id",
        "id",
        "roll_no",
        "roll number"
    ]
)

NAMECOL = find_column(
    students,
    [
        "student_name",
        "student name",
        "name"
    ]
)

if IDCOL is None:
    students["Student_ID"] = [
        f"STU{i+1:04d}"
        for i in range(len(students))
    ]
    IDCOL = "Student_ID"

if NAMECOL is None:
    students["Student_Name"] = [
        f"Student {i+1}"
        for i in range(len(students))
    ]
    NAMECOL = "Student_Name"


# ============================================================
# ATTENDANCE CSV
# ============================================================

ATTENDANCE_COLUMNS = [
    "date",
    "course",
    "student_id",
    "student_name",
    "status",
    "marked_by"
]


def load_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(
            columns=ATTENDANCE_COLUMNS
        )

    try:
        df = pd.read_csv(
            ATTENDANCE_FILE
        )

        for col in ATTENDANCE_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[ATTENDANCE_COLUMNS]

    except Exception:
        return pd.DataFrame(
            columns=ATTENDANCE_COLUMNS
        )


def save_attendance(df):
    df = df.copy()

    for col in ATTENDANCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[ATTENDANCE_COLUMNS]

    df.to_csv(
        ATTENDANCE_FILE,
        index=False
    )


def save_attendance_records(
    selected_date,
    course,
    faculty_name,
    choices
):
    """
    Insert/update attendance.
    Existing same date + course + student
    records are replaced instead of duplicated.
    """

    attendance = load_attendance()

    if not attendance.empty:
        mask = ~(
            (attendance["date"].astype(str) == str(selected_date))
            &
            (attendance["course"].astype(str) == str(course))
            &
            (
                attendance["student_id"]
                .astype(str)
                .isin([str(x) for x in choices.keys()])
            )
        )

        attendance = attendance[mask]

    new_rows = []

    for _, row in students.iterrows():

        sid = str(row[IDCOL])

        status = choices.get(
            sid,
            "Not Marked"
        )

        new_rows.append({
            "date": str(selected_date),
            "course": str(course),
            "student_id": sid,
            "student_name": str(row[NAMECOL]),
            "status": status,
            "marked_by": str(faculty_name)
        })

    new_df = pd.DataFrame(
        new_rows,
        columns=ATTENDANCE_COLUMNS
    )

    final = pd.concat(
        [attendance, new_df],
        ignore_index=True
    )

    save_attendance(final)


# ============================================================
# ACADEMIC MARKS CSV
# ============================================================

MARK_COLUMNS = [
    "student_id",
    "student_name",
    "subject",
    "marks",
    "max_marks",
    "updated_by"
]


def load_marks():

    if not os.path.exists(MARKS_FILE):
        return pd.DataFrame(
            columns=MARK_COLUMNS
        )

    try:
        df = pd.read_csv(
            MARKS_FILE
        )

        for col in MARK_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[MARK_COLUMNS]

    except Exception:
        return pd.DataFrame(
            columns=MARK_COLUMNS
        )


def save_marks(df):

    df = df.copy()

    for col in MARK_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[MARK_COLUMNS]

    df.to_csv(
        MARKS_FILE,
        index=False
    )


def save_single_mark(
    student_id,
    student_name,
    subject,
    marks,
    max_marks,
    updated_by
):

    df = load_marks()

    if not df.empty:

        df = df[
            ~(
                (df["student_id"].astype(str) == str(student_id))
                &
                (df["subject"].astype(str) == str(subject))
            )
        ]

    new_row = pd.DataFrame([{
        "student_id": str(student_id),
        "student_name": str(student_name),
        "subject": str(subject),
        "marks": float(marks),
        "max_marks": float(max_marks),
        "updated_by": str(updated_by)
    }])

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    save_marks(df)


# ============================================================
# ACADEMIC SCORE
# ============================================================

def academic_score(df):

    marks_col = find_column(
        df,
        [
            "academic_score",
            "academic marks",
            "percentage",
            "marks",
            "average marks"
        ]
    )

    cgpa_col = find_column(
        df,
        [
            "cgpa",
            "gpa"
        ]
    )

    if marks_col is not None:

        result = pd.to_numeric(
            df[marks_col],
            errors="coerce"
        )

        return (
            result
            .clip(0, 100)
            .fillna(70)
        )

    if cgpa_col is not None:

        result = pd.to_numeric(
            df[cgpa_col],
            errors="coerce"
        )

        return (
            result
            .mul(10)
            .clip(0, 100)
            .fillna(70)
        )

    return pd.Series(
        70.0,
        index=df.index
    )


# ============================================================
# ATTENDANCE SCORE
# ============================================================

def calculate_attendance_scores(df):

    attendance = load_attendance()

    fallback = numeric_series(
        df,
        [
            "attendance",
            "attendance_percent",
            "attendance %"
        ],
        70
    )

    if attendance.empty:
        return fallback

    result = []

    for _, row in df.iterrows():

        sid = str(row[IDCOL])

        records = attendance[
            attendance["student_id"]
            .astype(str)
            == sid
        ]

        if records.empty:

            result.append(
                fallback.loc[row.name]
            )

        else:

            present = (
                records["status"]
                .astype(str)
                .eq("Present")
                .sum()
            )

            total = len(records)

            result.append(
                (present / total) * 100
                if total > 0
                else fallback.loc[row.name]
            )

    return pd.Series(
        result,
        index=df.index
    )


# ============================================================
# CALCULATE ALL ANALYTICS
# ============================================================

def calculate_student_analytics(df):

    result = df.copy()

    result["_academic"] = academic_score(
        result
    )

    result["_attendance"] = calculate_attendance_scores(
        result
    )

    result["_lms"] = numeric_series(
        result,
        ["lms_score", "lms", "lms %"],
        70
    )

    result["_engagement"] = numeric_series(
        result,
        [
            "engagement",
            "engagement_score"
        ],
        70
    )

    result["_placement"] = numeric_series(
        result,
        [
            "placement",
            "placement_readiness"
        ],
        70
    )

    result["_skills"] = numeric_series(
        result,
        [
            "skills",
            "skill_score",
            "coding_score"
        ],
        70
    )

    result["_feedback"] = numeric_series(
        result,
        [
            "feedback",
            "feedback_score"
        ],
        70
    )

    # Original weighted Success Score
    result["Success_Score"] = (
        result["_academic"] * 0.30
        + result["_attendance"] * 0.20
        + result["_lms"] * 0.10
        + result["_engagement"] * 0.10
        + result["_placement"] * 0.15
        + result["_skills"] * 0.10
        + result["_feedback"] * 0.05
    ).round(1)

    result["Risk"] = np.select(
        [
            result["Success_Score"] < 45,
            result["Success_Score"] < 65
        ],
        [
            "High Risk",
            "Medium Risk"
        ],
        default="Low Risk"
    )

    result["Segment"] = np.select(
        [
            result["Success_Score"] >= 80,
            result["Success_Score"] >= 65,
            result["Success_Score"] >= 45
        ],
        [
            "High Performer",
            "Steady",
            "Needs Support"
        ],
        default="Critical"
    )

    return result


students = calculate_student_analytics(
    students
)


# ============================================================
# COMMON UI
# ============================================================

def header(title, subtitle=""):

    st.markdown(
        f"""
        <div class="topbar">
            <div style="font-size:25px;font-weight:800;color:#172554;">
                {title}
            </div>
            <div class="muted">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_metrics(items):

    columns = st.columns(
        len(items)
    )

    for col, item in zip(
        columns,
        items
    ):

        label, value, icon = item

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">
                        {icon}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                    <div class="metric-label">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def module_card(
    key,
    icon,
    title,
    description
):

    st.markdown(
        f"""
        <div class="module-card">
            <div class="module-icon">
                {icon}
            </div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    return st.button(
        f"Open {title}",
        key=key,
        width="stretch"
    )


def risk_html(risk):

    if risk == "Low Risk":
        return (
            '<span class="status-good">'
            'Low Risk'
            '</span>'
        )

    if risk == "Medium Risk":
        return (
            '<span class="status-warning">'
            'Medium Risk'
            '</span>'
        )

    return (
        '<span class="status-danger">'
        'High Risk'
        '</span>'
    )


def student_table(df):

    if df.empty:

        st.info(
            "No students found."
        )

        return

    cols = [
        IDCOL,
        NAMECOL,
        "_academic",
        "_attendance",
        "_skills",
        "Success_Score",
        "Risk",
        "Segment"
    ]

    available = [
        c for c in cols
        if c in df.columns
    ]

    view = df[available].copy()

    rename = {
        IDCOL: "Student ID",
        NAMECOL: "Student Name",
        "_academic": "Academic Score",
        "_attendance": "Attendance %",
        "_skills": "Skills",
        "Success_Score": "Success Score",
        "Risk": "Risk",
        "Segment": "Segment"
    }

    view = view.rename(
        columns=rename
    )

    st.dataframe(
        view,
        width="stretch",
        hide_index=True
    )


def back_to_main_portal():

    if st.button(
        "← Back to Main Portal",
        key="back_main"
    ):

        st.session_state.role = None
        st.session_state.module = None
        st.rerun()


def back_to_dashboard():

    if st.button(
        "← Back to Dashboard",
        key="back_dashboard"
    ):

        st.session_state.module = None
        st.rerun()


def logout_button():

    if st.button(
        "Logout",
        key="logout_top"
    ):

        st.session_state.role = None
        st.session_state.module = None
        st.rerun()


# ============================================================
# MAIN PORTAL
# ============================================================

def landing_page():

    st.markdown(
        """
        <div class="hero">
            <h1>🎓 EduTrack AI</h1>
            <p>
                AI-Powered Student Success & Early Warning Platform
            </p>
            <p style="margin-top:12px;">
                Turning attendance, academics, engagement, skills
                and placement data into actionable student-success
                insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">What EduTrack AI provides</div>',
        unsafe_allow_html=True
    )

    features = [
        (
            "📊",
            "Student Analytics",
            "Understand academic performance and success trends."
        ),
        (
            "🤖",
            "AI Insights",
            "Identify students who may need additional support."
        ),
        (
            "📅",
            "Attendance",
            "Faculty can record attendance and administrators can analyze it."
        ),
        (
            "🎯",
            "Placement Readiness",
            "Monitor practical skills and career preparation."
        )
    ]

    cols = st.columns(4)

    for col, (icon, title, desc) in zip(
        cols,
        features
    ):

        with col:

            st.markdown(
                f"""
                <div class="module-card">
                    <div class="module-icon">
                        {icon}
                    </div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Choose your portal</div>',
        unsafe_allow_html=True
    )

    portal_data = [
        (
            "Administrator",
            "👨‍💼",
            "College-wide analytics, AI model and student success insights."
        ),
        (
            "Faculty",
            "👨‍🏫",
            "Attendance, academic marks, student records and interventions."
        ),
        (
            "Student",
            "👨‍🎓",
            "Personal attendance, academics, skills, placement and AI recommendations."
        )
    ]

    cols = st.columns(3)

    for col, (role, icon, desc) in zip(
        cols,
        portal_data
    ):

        with col:

            st.markdown(
                f"""
                <div class="card" style="text-align:center;">
                    <div style="font-size:50px;">
                        {icon}
                    </div>
                    <h2 style="color:#172554;">
                        {role} Portal
                    </h2>
                    <p class="muted">
                        {desc}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                f"Enter {role} Portal",
                key=f"enter_{role}",
                width="stretch"
            ):

                st.session_state.role = role
                st.session_state.module = None
                st.rerun()


# ============================================================
# PORTAL HEADER
# ============================================================

def portal_header(
    role,
    icon
):

    cols = st.columns(
        [8, 1]
    )

    with cols[0]:

        st.markdown(
            f"""
            <div class="topbar">
                <span class="brand">
                    🎓 EduTrack AI
                </span>
                &nbsp;&nbsp;
                <span class="muted">
                    {icon} {role} Portal
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with cols[1]:

        logout_button()


# ============================================================
# FACULTY ATTENDANCE
# ============================================================

def faculty_attendance():
    students=load_students()

    header(
        "📝 Take Attendance",
        "Faculty-only attendance marking"
    )

    st.info(
        "Only Faculty can mark attendance. "
        "Students can view their attendance but cannot edit it."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_date = st.date_input(
            "Attendance Date",
            value=date.today(),
            key="attendance_date"
        )

    with col2:

        course = st.text_input(
            "Course",
            value="Data Structures",
            key="attendance_course"
        )

    with col3:

        faculty_name = st.text_input(
            "Faculty Name",
            value="Faculty",
            key="faculty_name"
        )

    course = course.strip()

    if not course:
        course = "General"

    # Existing records
    attendance = load_attendance()

    existing = attendance[
        (
            attendance["date"].astype(str)
            == str(selected_date)
        )
        &
        (
            attendance["course"].astype(str)
            == str(course)
        )
    ]

    st.markdown(
        '<div class="section-title">Mark Student Attendance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <b>Attendance controls</b>
            <p class="muted">
                ○ Present &nbsp;&nbsp;&nbsp; ○ Absent
                <br>
                Click the circle itself.
                Present becomes green and Absent becomes red.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    choices = {}

    # --------------------------------------------------------
    # Each student
    # --------------------------------------------------------

    for _, row in students.iterrows():

        sid = str(row[IDCOL])
        name = str(row[NAMECOL])

        state_key = (
            f"{selected_date}_"
            f"{course}_"
            f"{sid}"
        )

        # Existing database value
        existing_status = "Not Marked"

        if not existing.empty:

            found = existing[
                existing["student_id"]
                .astype(str)
                == sid
            ]

            if not found.empty:

                existing_status = str(
                    found.iloc[0]["status"]
                )

        # Initialize session state
        if state_key not in st.session_state.attendance_choices:

            if existing_status in [
                "Present",
                "Absent"
            ]:

                st.session_state.attendance_choices[
                    state_key
                ] = existing_status

            else:

                st.session_state.attendance_choices[
                    state_key
                ] = "Not Marked"

        current = st.session_state.attendance_choices[
            state_key
        ]

        # Row
        c1, c2, c3, c4 = st.columns(
            [6, 1.2, 1.2, 2]
        )

        with c1:

            st.markdown(
                f"""
                <div class="attendance-name">
                    {name}
                    <div class="attendance-id">
                        {sid}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # PRESENT
        with c2:

            st.markdown(
                '<div class="attendance-label">Present</div>',
                unsafe_allow_html=True
            )

            if current == "Present":

                clicked = st.button(
                    "✓",
                    key=(
                        f"att_present_selected_"
                        f"{selected_date}_"
                        f"{course}_"
                        f"{sid}"
                    ),
                    help="Present"
                )

            else:

                clicked = st.button(
                    " ",
                    key=(
                        f"att_present_"
                        f"{selected_date}_"
                        f"{course}_"
                        f"{sid}"
                    ),
                    help="Mark Present"
                )

            if clicked:

                st.session_state.attendance_choices[
                    state_key
                ] = "Present"

                st.rerun()

        # ABSENT
        with c3:

            st.markdown(
                '<div class="attendance-label">Absent</div>',
                unsafe_allow_html=True
            )

            if current == "Absent":

                clicked = st.button(
                    "✓",
                    key=(
                        f"att_absent_selected_"
                        f"{selected_date}_"
                        f"{course}_"
                        f"{sid}"
                    ),
                    help="Absent"
                )

            else:

                clicked = st.button(
                    " ",
                    key=(
                        f"att_absent_"
                        f"{selected_date}_"
                        f"{course}_"
                        f"{sid}"
                    ),
                    help="Mark Absent"
                )

            if clicked:

                st.session_state.attendance_choices[
                    state_key
                ] = "Absent"

                st.rerun()

        with c4:

            if current == "Present":

                st.markdown(
                    '<div class="status-good" style="padding-top:10px;">'
                    '🟢 Present'
                    '</div>',
                    unsafe_allow_html=True
                )

            elif current == "Absent":

                st.markdown(
                    '<div class="status-danger" style="padding-top:10px;">'
                    '🔴 Absent'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="muted" style="padding-top:10px;">'
                    'Not selected'
                    '</div>',
                    unsafe_allow_html=True
                )

        choices[sid] = current

    # --------------------------------------------------------
    # Submit attendance
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    unmarked = [
        sid
        for sid, status in choices.items()
        if status == "Not Marked"
    ]

    if unmarked:

        st.warning(
            f"{len(unmarked)} student(s) still need attendance."
        )

    if st.button(
        "💾 Submit Attendance",
        type="primary",
        width="stretch",
        key="submit_attendance"
    ):

        if unmarked:

            st.error(
                "Please mark Present or Absent for every student."
            )

        else:

            save_attendance_records(
                selected_date,
                course,
                faculty_name,
                choices
            )

            present_count = sum(
                value == "Present"
                for value in choices.values()
            )

            absent_count = sum(
                value == "Absent"
                for value in choices.values()
            )

            total = len(choices)

            attendance_percent = (
                present_count / total * 100
                if total
                else 0
            )

            # Refresh analytics
            students = calculate_student_analytics(
                load_students()
            )

            # Calculate averages
            avg_success = students[
                "Success_Score"
            ].mean()

            current_attendance = load_attendance()

            selected_records = current_attendance[
                (
                    current_attendance["date"].astype(str)
                    == str(selected_date)
                )
                &
                (
                    current_attendance["course"].astype(str)
                    == str(course)
                )
            ]

            present_ids = selected_records[
                selected_records["status"] == "Present"
            ]["student_id"].astype(str)

            absent_ids = selected_records[
                selected_records["status"] == "Absent"
            ]["student_id"].astype(str)

            present_students = students[
                students[IDCOL]
                .astype(str)
                .isin(present_ids)
            ]

            absent_students = students[
                students[IDCOL]
                .astype(str)
                .isin(absent_ids)
            ]

            present_avg_success = (
                present_students["Success_Score"].mean()
                if not present_students.empty
                else 0
            )

            absent_avg_success = (
                absent_students["Success_Score"].mean()
                if not absent_students.empty
                else 0
            )

            avg_cgpa = pd.to_numeric(
                students["_academic"],
                errors="coerce"
            ).mean() / 10

            st.session_state.attendance_summary = {
                "total": total,
                "present": present_count,
                "absent": absent_count,
                "attendance_percent": attendance_percent,
                "avg_success": avg_success,
                "present_avg_success": present_avg_success,
                "absent_avg_success": absent_avg_success,
                "avg_cgpa": avg_cgpa
            }

            st.success(
                "Attendance submitted successfully!"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = st.session_state.attendance_summary

    if summary is not None:

        st.markdown(
            '<div class="section-title">'
            'Attendance Summary'
            '</div>',
            unsafe_allow_html=True
        )

        show_metrics(
            [
                (
                    "Total Students",
                    summary["total"],
                    "👥"
                ),
                (
                    "Present",
                    summary["present"],
                    "🟢"
                ),
                (
                    "Absent",
                    summary["absent"],
                    "🔴"
                ),
                (
                    "Attendance %",
                    f'{summary["attendance_percent"]:.1f}%',
                    "📅"
                ),
                (
                    "Average Success Score",
                    f'{summary["avg_success"]:.1f}',
                    "🎯"
                ),
                (
                    "Present Students Avg Success",
                    f'{summary["present_avg_success"]:.1f}',
                    "🟢"
                ),
                (
                    "Absent Students Avg Success",
                    f'{summary["absent_avg_success"]:.1f}',
                    "🔴"
                ),
                (
                    "Average CGPA",
                    f'{summary["avg_cgpa"]:.2f}',
                    "📚"
                )
            ]
        )


# ============================================================
# FACULTY ACADEMIC MARKS
# ============================================================

def faculty_academic_marks():

    header(
        "📚 Academic Marks",
        "Faculty can enter and update academic performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        subject = st.text_input(
            "Subject",
            value="Data Structures",
            key="marks_subject"
        )

    with col2:

        max_marks = st.number_input(
            "Maximum Marks",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            key="marks_max"
        )

    with col3:

        faculty_name = st.text_input(
            "Updated By",
            value="Faculty",
            key="marks_faculty"
        )

    subject = subject.strip()

    if not subject:
        subject = "General"

    marks_data = load_marks()

    st.markdown(
        '<div class="section-title">Student Academic Records</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Student rows
    # --------------------------------------------------------

    for _, row in students.iterrows():

        sid = str(row[IDCOL])
        name = str(row[NAMECOL])

        old = marks_data[
            (
                marks_data["student_id"]
                .astype(str)
                == sid
            )
            &
            (
                marks_data["subject"]
                .astype(str)
                == subject
            )
        ]

        if not old.empty:

            old_marks = float(
                pd.to_numeric(
                    old.iloc[0]["marks"],
                    errors="coerce"
                )
            )

            if np.isnan(old_marks):
                old_marks = 0.0

        else:

            old_marks = float(
                min(
                    max(
                        row["_academic"],
                        0
                    ),
                    max_marks
                )
            )

        c1, c2, c3, c4 = st.columns(
            [1.4, 3, 2, 1.5]
        )

        with c1:

            st.markdown(
                f"""
                <div style="padding-top:12px;">
                    <b>{sid}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div style="padding-top:12px;">
                    {name}
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            mark_value = st.number_input(
                f"Marks - {sid}",
                min_value=0.0,
                max_value=float(max_marks),
                value=min(
                    old_marks,
                    float(max_marks)
                ),
                step=0.5,
                key=f"marks_input_{subject}_{sid}",
                label_visibility="collapsed"
            )

        with c4:

            if st.button(
                "Save",
                key=f"save_marks_{subject}_{sid}"
            ):

                save_single_mark(
                    sid,
                    name,
                    subject,
                    mark_value,
                    max_marks,
                    faculty_name
                )

                st.success(
                    f"Saved marks for {name}"
                )

    # --------------------------------------------------------
    # Current marks
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Current Marks</div>',
        unsafe_allow_html=True
    )

    current_marks = load_marks()

    if current_marks.empty:

        st.info(
            "No academic marks have been saved yet."
        )

    else:

        st.dataframe(
            current_marks.sort_values(
                ["subject", "student_id"]
            ),
            width="stretch",
            hide_index=True
        )


# ============================================================
# FACULTY STUDENT RECORDS
# ============================================================

def faculty_student_records():

    header(
        "👨‍🎓 Student Records",
        "Search and review student information"
    )

    query = st.text_input(
        "Search by Student ID or Name",
        key="student_search"
    )

    view = students.copy()

    if query:

        query = query.strip()

        view = view[
            view[IDCOL]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False
            )
            |
            view[NAMECOL]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False
            )
        ]

    student_table(view)

    st.markdown(
        '<div class="section-title">Student Details</div>',
        unsafe_allow_html=True
    )

    if not view.empty:

        selected_id = st.selectbox(
            "Select Student",
            view[IDCOL]
            .astype(str)
            .tolist(),
            key="faculty_student_detail"
        )

        row = view[
            view[IDCOL]
            .astype(str)
            == str(selected_id)
        ].iloc[0]

        show_metrics(
            [
                (
                    "Student ID",
                    str(selected_id),
                    "🪪"
                ),
                (
                    "Academic Score",
                    f'{row["_academic"]:.1f}%',
                    "📚"
                ),
                (
                    "Attendance",
                    f'{row["_attendance"]:.1f}%',
                    "📅"
                ),
                (
                    "Success Score",
                    f'{row["Success_Score"]:.1f}',
                    "🎯"
                )
            ]
        )


# ============================================================
# ADMIN EXECUTIVE DASHBOARD
# ============================================================

def executive_dashboard():

    header(
        "📊 Executive Dashboard",
        "College-wide student success overview"
    )

    show_metrics(
        [
            (
                "Total Students",
                len(students),
                "👨‍🎓"
            ),
            (
                "Average Success Score",
                f'{students["Success_Score"].mean():.1f}',
                "🎯"
            ),
            (
                "Low Risk",
                int(
                    (
                        students["Risk"]
                        == "Low Risk"
                    ).sum()
                ),
                "🟢"
            ),
            (
                "High Risk",
                int(
                    (
                        students["Risk"]
                        == "High Risk"
                    ).sum()
                ),
                "🚨"
            )
        ]
    )

    st.markdown(
        '<div class="section-title">Performance Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="card"><h3>Success Score Segments</h3></div>',
            unsafe_allow_html=True
        )

        segment_counts = (
            students["Segment"]
            .value_counts()
            .reindex(
                [
                    "High Performer",
                    "Steady",
                    "Needs Support",
                    "Critical"
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            segment_counts
        )

    with c2:

        st.markdown(
            '<div class="card"><h3>Risk Distribution</h3></div>',
            unsafe_allow_html=True
        )

        risk_counts = (
            students["Risk"]
            .value_counts()
            .reindex(
                [
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            risk_counts
        )

    st.markdown(
        '<div class="section-title">Top Students</div>',
        unsafe_allow_html=True
    )

    student_table(
        students
        .sort_values(
            "Success_Score",
            ascending=False
        )
        .head(10)
    )


# ============================================================
# STUDENT 360
# ============================================================

def student_360():

    header(
        "👨‍🎓 Student 360°",
        "Complete student success profile"
    )

    student_ids = (
        students[IDCOL]
        .astype(str)
        .tolist()
    )

    selected = st.selectbox(
        "Select Student",
        student_ids,
        key="student360_select"
    )

    row = students[
        students[IDCOL]
        .astype(str)
        == str(selected)
    ].iloc[0]

    st.markdown(
        f"""
        <div class="card">
            <h2 style="color:#172554;">
                {row[NAMECOL]}
            </h2>
            <p class="muted">
                Student ID: {selected}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_metrics(
        [
            (
                "Success Score",
                f'{row["Success_Score"]:.1f}/100',
                "🎯"
            ),
            (
                "Academic",
                f'{row["_academic"]:.1f}%',
                "📚"
            ),
            (
                "Attendance",
                f'{row["_attendance"]:.1f}%',
                "📅"
            ),
            (
                "Placement",
                f'{row["_placement"]:.1f}%',
                "🎓"
            )
        ]
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="card"><h3>Performance Factors</h3></div>',
            unsafe_allow_html=True
        )

        factor_data = pd.Series(
            {
                "Academic": row["_academic"],
                "Attendance": row["_attendance"],
                "LMS": row["_lms"],
                "Engagement": row["_engagement"],
                "Skills": row["_skills"],
                "Placement": row["_placement"],
                "Feedback": row["_feedback"]
            }
        )

        st.bar_chart(
            factor_data
        )

    with c2:

        st.markdown(
            f"""
            <div class="ai-card">
                <h3>🤖 AI Student Insight</h3>
                <p>
                    Current segment:
                    <b>{row["Segment"]}</b>
                </p>
                <p>
                    Risk level:
                    {risk_html(row["Risk"])}
                </p>
                <p>
                    Focus on the lowest-performing factors
                    to improve the overall Success Score.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# EARLY WARNING
# ============================================================

def early_warning():

    header(
        "🚨 Early Warning",
        "Identify students who may need support"
    )

    selected_risks = st.multiselect(
        "Risk Level",
        [
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],
        default=[
            "High Risk",
            "Medium Risk"
        ]
    )

    view = students[
        students["Risk"]
        .isin(selected_risks)
    ].sort_values(
        "Success_Score"
    )

    show_metrics(
        [
            (
                "Students",
                len(view),
                "👥"
            ),
            (
                "High Risk",
                int(
                    (
                        view["Risk"]
                        == "High Risk"
                    ).sum()
                ),
                "🚨"
            ),
            (
                "Medium Risk",
                int(
                    (
                        view["Risk"]
                        == "Medium Risk"
                    ).sum()
                ),
                "🟠"
            ),
            (
                "Average Success",
                f'{view["Success_Score"].mean():.1f}'
                if not view.empty
                else "0.0",
                "🎯"
            )
        ]
    )

    student_table(view)

    if not view.empty:

        st.download_button(
            "⬇️ Download Early Warning Report",
            view.to_csv(
                index=False
            ),
            "early_warning_report.csv",
            "text/csv",
            width="stretch"
        )


# ============================================================
# SEGMENTATION
# ============================================================

def segmentation():

    header(
        "📊 Student Segmentation",
        "Group students using Success Score"
    )

    segment = st.selectbox(
        "Select Segment",
        [
            "All",
            "High Performer",
            "Steady",
            "Needs Support",
            "Critical"
        ],
        key="segment_select"
    )

    if segment == "All":

        view = students

    else:

        view = students[
            students["Segment"]
            == segment
        ]

    show_metrics(
        [
            (
                "Students",
                len(view),
                "👥"
            ),
            (
                "Average Success",
                f'{view["Success_Score"].mean():.1f}'
                if not view.empty
                else "0.0",
                "🎯"
            ),
            (
                "Average Attendance",
                f'{view["_attendance"].mean():.1f}%'
                if not view.empty
                else "0.0%",
                "📅"
            )
        ]
    )

    student_table(
        view
    )


# ============================================================
# AI MODEL
# ============================================================

def ai_model():

    header(
        "🧠 AI Model & Explainability",
        "Random Forest early-warning model"
    )

    if not SKLEARN_AVAILABLE:

        st.warning(
            "scikit-learn is not installed. "
            "Install requirements.txt to enable the model."
        )

        return

    features = [
        "_academic",
        "_attendance",
        "_lms",
        "_engagement",
        "_placement",
        "_skills",
        "_feedback"
    ]

    X = students[
        features
    ].fillna(0)

    y = (
        students["Success_Score"]
        < 65
    ).astype(int)

    if (
        len(students) < 20
        or y.nunique() < 2
    ):

        st.info(
            "Not enough data variation to train the model."
        )

        return

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X,
        y
    )

    importance = pd.Series(
        model.feature_importances_,
        index=[
            "Academic",
            "Attendance",
            "LMS",
            "Engagement",
            "Placement",
            "Skills",
            "Feedback"
        ]
    ).sort_values(
        ascending=False
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="card"><h3>Feature Importance</h3></div>',
            unsafe_allow_html=True
        )

        st.bar_chart(
            importance
        )

    with c2:

        st.markdown(
            """
            <div class="ai-card">
                <h3>🤖 Model Purpose</h3>
                <p>
                    EduTrack AI uses machine learning as
                    decision support to identify students
                    who may need attention.
                </p>
                <p>
                    It does not automatically make
                    academic decisions about students.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Students Used",
            len(X)
        )


# ============================================================
# INTERVENTIONS
# ============================================================

def interventions():

    header(
        "💡 AI Interventions",
        "Recommended support actions"
    )

    rows = []

    for _, row in students.iterrows():

        if row["_attendance"] < 75:

            action = (
                "Attendance support + "
                "faculty follow-up"
            )

        elif row["_academic"] < 60:

            action = (
                "Academic mentoring + "
                "subject support"
            )

        elif row["_skills"] < 60:

            action = (
                "Coding practice plan + "
                "lab support"
            )

        elif row["_engagement"] < 60:

            action = (
                "Engagement activity + "
                "mentor check-in"
            )

        else:

            action = (
                "Continue current learning plan"
            )

        rows.append(
            [
                row[IDCOL],
                row[NAMECOL],
                row["Segment"],
                row["Risk"],
                action
            ]
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "Student ID",
            "Student Name",
            "Segment",
            "Risk",
            "Recommended Intervention"
        ]
    )

    st.dataframe(
        result,
        width="stretch",
        hide_index=True
    )


# ============================================================
# PROGRESS TRACKING
# ============================================================

def progress_tracking():

    header(
        "📈 Progress Tracking",
        "Monitor student success factors"
    )

    average = pd.Series(
        {
            "Academic":
                students["_academic"].mean(),

            "Attendance":
                students["_attendance"].mean(),

            "LMS":
                students["_lms"].mean(),

            "Engagement":
                students["_engagement"].mean(),

            "Skills":
                students["_skills"].mean(),

            "Placement":
                students["_placement"].mean()
        }
    )

    st.bar_chart(
        average
    )

    show_metrics(
        [
            (
                "Academic",
                f'{average["Academic"]:.1f}%',
                "📚"
            ),
            (
                "Attendance",
                f'{average["Attendance"]:.1f}%',
                "📅"
            ),
            (
                "Skills",
                f'{average["Skills"]:.1f}%',
                "💻"
            ),
            (
                "Placement",
                f'{average["Placement"]:.1f}%',
                "🎯"
            )
        ]
    )


# ============================================================
# ATTENDANCE ANALYTICS
# ============================================================

def attendance_analytics():

    header(
        "📅 Attendance Analytics",
        "Faculty-recorded attendance across courses"
    )

    attendance = load_attendance()

    if attendance.empty:

        st.info(
            "No attendance has been recorded yet."
        )

        return

    course_attendance = (
        attendance
        .groupby("course")["status"]
        .apply(
            lambda x:
            (
                x.eq("Present").mean()
                * 100
            )
        )
        .round(1)
    )

    st.markdown(
        '<div class="section-title">'
        'Course Attendance'
        '</div>',
        unsafe_allow_html=True
    )

    st.bar_chart(
        course_attendance
    )

    show_metrics(
        [
            (
                "Attendance Records",
                len(attendance),
                "📝"
            ),
            (
                "Courses",
                attendance["course"].nunique(),
                "📚"
            ),
            (
                "Present",
                int(
                    (
                        attendance["status"]
                        == "Present"
                    ).sum()
                ),
                "🟢"
            ),
            (
                "Absent",
                int(
                    (
                        attendance["status"]
                        == "Absent"
                    ).sum()
                ),
                "🔴"
            )
        ]
    )

    st.dataframe(
        attendance.sort_values(
            "date",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )


# ============================================================
# FACULTY DASHBOARD
# ============================================================

def faculty_dashboard():

    portal_header(
        "Faculty",
        "👨‍🏫"
    )

    st.markdown(
        """
        <div class="hero">
            <h1>👨‍🏫 Faculty Portal</h1>
            <p>
                Manage attendance, academic marks, student records
                and student-success interventions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_metrics(
        [
            (
                "Students",
                len(students),
                "👨‍🎓"
            ),
            (
                "Average Success",
                f'{students["Success_Score"].mean():.1f}',
                "🎯"
            ),
            (
                "At Risk",
                int(
                    (
                        students["Risk"]
                        != "Low Risk"
                    ).sum()
                ),
                "🚨"
            ),
            (
                "Average Attendance",
                f'{students["_attendance"].mean():.1f}%',
                "📅"
            )
        ]
    )

    st.markdown(
        '<div class="section-title">Faculty Modules</div>',
        unsafe_allow_html=True
    )

    modules = [
        (
            "attendance",
            "📝",
            "Take Attendance",
            "Record daily student attendance."
        ),
        (
            "marks",
            "📚",
            "Academic Marks",
            "Enter and update academic marks."
        ),
        (
            "students",
            "👨‍🎓",
            "Student Records",
            "Search and review student profiles."
        ),
        (
            "early",
            "🚨",
            "Early Warning",
            "Identify students needing support."
        ),
        (
            "interventions",
            "💡",
            "Interventions",
            "Review recommended support actions."
        ),
        (
            "progress",
            "📈",
            "Progress Tracking",
            "Monitor student-success factors."
        )
    ]

    for start in range(
        0,
        len(modules),
        3
    ):

        cols = st.columns(3)

        for col, module in zip(
            cols,
            modules[start:start+3]
        ):

            key, icon, title, desc = module

            with col:

                if module_card(
                    f"faculty_{key}",
                    icon,
                    title,
                    desc
                ):

                    st.session_state.module = key
                    st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def get_student():

    student_ids = (
        students[IDCOL]
        .astype(str)
        .tolist()
    )

    selected = st.selectbox(
        "Select Student",
        student_ids,
        key="student_selector"
    )

    st.session_state.selected_student = selected

    row = students[
        students[IDCOL]
        .astype(str)
        == str(selected)
    ].iloc[0]

    return row


def student_dashboard():

    portal_header(
        "Student",
        "👨‍🎓"
    )

    row = get_student()

    st.markdown(
        f"""
        <div class="hero">
            <h1>
                Welcome, {row[NAMECOL]} 👋
            </h1>
            <p>
                Your personal EduTrack AI student-success dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_metrics(
        [
            (
                "Success Score",
                f'{row["Success_Score"]:.1f}/100',
                "🎯"
            ),
            (
                "Attendance",
                f'{row["_attendance"]:.1f}%',
                "📅"
            ),
            (
                "Academic",
                f'{row["_academic"]:.1f}%',
                "📚"
            ),
            (
                "Placement",
                f'{row["_placement"]:.1f}%',
                "🎓"
            )
        ]
    )

    recommendation = (
        "Focus on attendance and academic support."
        if row["_attendance"] < 75
        else
        "Keep your consistency and continue improving practical skills."
    )

    st.markdown(
        f"""
        <div class="ai-card">
            <h3>🤖 AI Recommendation</h3>
            <p>
                Your current segment is
                <b>{row["Segment"]}</b>.
            </p>
            <p>
                {recommendation}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Student Modules</div>',
        unsafe_allow_html=True
    )

    modules = [
        (
            "myatt",
            "📅",
            "My Attendance",
            "View faculty-recorded attendance."
        ),
        (
            "myacad",
            "📚",
            "My Academics",
            "View marks and academic performance."
        ),
        (
            "coding",
            "💻",
            "Coding & Skills",
            "Track coding and practical skills."
        ),
        (
            "myprogress",
            "📈",
            "My Progress",
            "View your success factors."
        ),
        (
            "placement",
            "🎯",
            "Placement",
            "View placement readiness."
        ),
        (
            "recommend",
            "🤖",
            "AI Recommendations",
            "View personalized recommendations."
        )
    ]

    for start in range(
        0,
        len(modules),
        3
    ):

        cols = st.columns(3)

        for col, module in zip(
            cols,
            modules[start:start+3]
        ):

            key, icon, title, desc = module

            with col:

                if module_card(
                    f"student_{key}",
                    icon,
                    title,
                    desc
                ):

                    st.session_state.module = key
                    st.rerun()


# ============================================================
# STUDENT ATTENDANCE
# ============================================================

def student_attendance():

    header(
        "📅 My Attendance",
        "View-only attendance recorded by Faculty"
    )

    sid = st.session_state.get(
        "selected_student"
    )

    if sid is None:

        st.info(
            "Please select your student profile from the dashboard."
        )

        return

    attendance = load_attendance()

    mine = attendance[
        attendance["student_id"]
        .astype(str)
        == str(sid)
    ]

    if mine.empty:

        st.info(
            "No faculty attendance records are available yet."
        )

        return

    total = len(mine)

    present = (
        mine["status"]
        == "Present"
    ).sum()

    absent = (
        mine["status"]
        == "Absent"
    ).sum()

    percentage = (
        present / total * 100
        if total
        else 0
    )

    show_metrics(
        [
            (
                "Total Classes",
                total,
                "📚"
            ),
            (
                "Present",
                present,
                "🟢"
            ),
            (
                "Absent",
                absent,
                "🔴"
            ),
            (
                "Attendance %",
                f"{percentage:.1f}%",
                "📅"
            )
        ]
    )

    st.markdown(
        '<div class="section-title">Attendance Records</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        mine.sort_values(
            "date",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )

    # Course-wise attendance
    course_summary = (
        mine.groupby("course")["status"]
        .apply(
            lambda x:
            x.eq("Present").mean() * 100
        )
        .round(1)
    )

    if not course_summary.empty:

        st.markdown(
            '<div class="section-title">Course-wise Attendance</div>',
            unsafe_allow_html=True
        )

        st.bar_chart(
            course_summary
        )


# ============================================================
# STUDENT ACADEMICS
# ============================================================

def student_academics():

    header(
        "📚 My Academics",
        "View academic marks and CGPA"
    )

    sid = st.session_state.get(
        "selected_student"
    )

    if sid is None:
        st.info(
            "Please select your student profile."
        )
        return

    row = students[
        students[IDCOL]
        .astype(str)
        == str(sid)
    ].iloc[0]

    show_metrics(
        [
            (
                "Student ID",
                sid,
                "🪪"
            ),
            (
                "Student Name",
                row[NAMECOL],
                "👤"
            ),
            (
                "CGPA",
                f'{row["_academic"] / 10:.2f}',
                "📚"
            ),
            (
                "Academic Score",
                f'{row["_academic"]:.1f}%',
                "⭐"
            )
        ]
    )

    marks = load_marks()

    mine = marks[
        marks["student_id"]
        .astype(str)
        == str(sid)
    ]

    st.markdown(
        '<div class="section-title">Subject-wise Marks</div>',
        unsafe_allow_html=True
    )

    if mine.empty:

        st.info(
            "No subject-wise marks have been entered by Faculty yet."
        )

    else:

        st.dataframe(
            mine,
            width="stretch",
            hide_index=True
        )


# ============================================================
# STUDENT CODING & SKILLS
# ============================================================

def student_coding():

    header(
        "💻 Coding & Skills",
        "Track practical learning and programming skills"
    )

    row = students[
        students[IDCOL]
        .astype(str)
        == str(
            st.session_state.get(
                "selected_student"
            )
        )
    ].iloc[0]

    show_metrics(
        [
            (
                "Skills Score",
                f'{row["_skills"]:.1f}%',
                "💻"
            ),
            (
                "Coding Readiness",
                f'{row["_skills"]:.1f}%',
                "🧠"
            ),
            (
                "Placement Readiness",
                f'{row["_placement"]:.1f}%',
                "🎯"
            )
        ]
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="card">
                <h3>💻 Coding Practice</h3>
                <p class="muted">
                    Build practical programming skills
                    through regular practice.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.selectbox(
            "Practice Topic",
            [
                "Python",
                "C++",
                "Java",
                "Data Structures",
                "Algorithms"
            ],
            key="coding_topic"
        )

    with c2:

        st.markdown(
            """
            <div class="card">
                <h3>🏆 Skill Areas</h3>
                <p>
                    Python<br>
                    Data Structures<br>
                    Problem Solving<br>
                    Algorithms<br>
                    Git & Development
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# STUDENT PROGRESS
# ============================================================

def student_progress():

    header(
        "📈 My Progress",
        "Your EduTrack AI success factors"
    )

    sid = st.session_state.get(
        "selected_student"
    )

    if sid is None:
        st.info(
            "Please select your student profile."
        )
        return

    row = students[
        students[IDCOL]
        .astype(str)
        == str(sid)
    ].iloc[0]

    values = pd.Series(
        {
            "Academic":
                row["_academic"],
            "Attendance":
                row["_attendance"],
            "LMS":
                row["_lms"],
            "Engagement":
                row["_engagement"],
            "Skills":
                row["_skills"],
            "Placement":
                row["_placement"],
            "Feedback":
                row["_feedback"]
        }
    )

    st.bar_chart(
        values
    )

    show_metrics(
        [
            (
                "Success Score",
                f'{row["Success_Score"]:.1f}/100',
                "🎯"
            ),
            (
                "Academic",
                f'{row["_academic"]:.1f}%',
                "📚"
            ),
            (
                "Skills",
                f'{row["_skills"]:.1f}%',
                "💻"
            ),
            (
                "Placement",
                f'{row["_placement"]:.1f}%',
                "🎓"
            )
        ]
    )


# ============================================================
# STUDENT PLACEMENT
# ============================================================

def student_placement():

    header(
        "🎯 Placement",
        "Career preparation and placement readiness"
    )

    sid = st.session_state.get(
        "selected_student"
    )

    row = students[
        students[IDCOL]
        .astype(str)
        == str(sid)
    ].iloc[0]

    show_metrics(
        [
            (
                "Placement Readiness",
                f'{row["_placement"]:.1f}%',
                "🎯"
            ),
            (
                "Coding Skills",
                f'{row["_skills"]:.1f}%',
                "💻"
            ),
            (
                "Academic",
                f'{row["_academic"]:.1f}%',
                "📚"
            ),
            (
                "Success Score",
                f'{row["Success_Score"]:.1f}',
                "⭐"
            )
        ]
    )

    st.markdown(
        """
        <div class="ai-card">
            <h3>🎯 Placement Preparation</h3>
            <p>
                Use this score as a preparation indicator.
                It is not a hiring decision or guarantee.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT AI RECOMMENDATIONS
# ============================================================

def student_recommendations():

    header(
        "🤖 AI Recommendations",
        "Personalized recommendations based on your data"
    )

    sid = st.session_state.get(
        "selected_student"
    )

    row = students[
        students[IDCOL]
        .astype(str)
        == str(sid)
    ].iloc[0]

    recommendations = []

    if row["_attendance"] < 75:

        recommendations.append(
            "Improve attendance and discuss missed classes with your faculty or mentor."
        )

    if row["_academic"] < 65:

        recommendations.append(
            "Spend extra time on subjects with lower academic performance."
        )

    if row["_skills"] < 65:

        recommendations.append(
            "Complete more coding practice and practical lab exercises."
        )

    if row["_engagement"] < 65:

        recommendations.append(
            "Increase participation in learning activities."
        )

    if row["_placement"] < 65:

        recommendations.append(
            "Work on coding, problem solving and placement preparation."
        )

    if not recommendations:

        recommendations.append(
            "Your current indicators are healthy. Maintain consistency and continue improving your practical skills."
        )

    for recommendation in recommendations:

        st.markdown(
            f"""
            <div class="card">
                <b>🤖 Recommendation</b>
                <p>
                    {recommendation}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
# ============================================================
# ADMIN DASHBOARD
# ============================================================

def administrator_dashboard():

    portal_header(
        "Administrator",
        "👨‍💼"
    )

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Administrator Portal</h1>
            <p>
                College-wide student analytics, AI insights,
                early warning and intervention monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_metrics(
        [
            (
                "Students",
                len(students),
                "👨‍🎓"
            ),
            (
                "Average Success",
                f'{students["Success_Score"].mean():.1f}',
                "🎯"
            ),
            (
                "High Risk",
                int(
                    (
                        students["Risk"]
                        == "High Risk"
                    ).sum()
                ),
                "🚨"
            ),
            (
                "Placement Readiness",
                f'{students["_placement"].mean():.1f}%',
                "🎓"
            )
        ]
    )

    st.markdown(
        '<div class="section-title">Administrator Modules</div>',
        unsafe_allow_html=True
    )

    modules = [
        (
            "executive",
            "📊",
            "Executive Dashboard",
            "College-wide KPIs and trends."
        ),
        (
            "360",
            "👨‍🎓",
            "Student 360°",
            "Deep student profile."
        ),
        (
            "early",
            "🚨",
            "Early Warning",
            "Identify at-risk students."
        ),
        (
            "segmentation",
            "📈",
            "Segmentation",
            "Performance-based groups."
        ),
        (
            "model",
            "🧠",
            "AI Model",
            "Random Forest explainability."
        ),
        (
            "interventions",
            "💡",
            "Interventions",
            "Recommended support actions."
        ),
        (
            "progress",
            "📈",
            "Progress Tracking",
            "Student success trends."
        ),
        (
            "attendance",
            "📅",
            "Attendance Analytics",
            "Analyze faculty attendance."
        )
    ]

    for start in range(
        0,
        len(modules),
        4
    ):

        cols = st.columns(4)

        for col, module in zip(
            cols,
            modules[start:start+4]
        ):

            key, icon, title, desc = module

            with col:

                if module_card(
                    f"admin_{key}",
                    icon,
                    title,
                    desc
                ):

                    st.session_state.module = key
                    st.rerun()


# ============================================================
# ROUTING
# ============================================================

if st.session_state.role is None:

    landing_page()

else:

    role = st.session_state.role
    module = st.session_state.module

    # --------------------------------------------------------
    # ADMINISTRATOR
    # --------------------------------------------------------

    if role == "Administrator":

        if module is None:

            administrator_dashboard()

        else:

            back_to_main_portal()

            back_to_dashboard()

            if module == "executive":

                executive_dashboard()

            elif module == "360":

                student_360()

            elif module == "early":

                early_warning()

            elif module == "segmentation":

                segmentation()

            elif module == "model":

                ai_model()

            elif module == "interventions":

                interventions()

            elif module == "progress":

                progress_tracking()

            elif module == "attendance":

                attendance_analytics()

    # --------------------------------------------------------
    # FACULTY
    # --------------------------------------------------------

    elif role == "Faculty":

        if module is None:

            faculty_dashboard()

        else:

            back_to_main_portal()

            back_to_dashboard()

            if module == "attendance":

                faculty_attendance()

            elif module == "marks":

                faculty_academic_marks()

            elif module == "students":

                faculty_student_records()

            elif module == "early":

                early_warning()

            elif module == "interventions":

                interventions()

            elif module == "progress":

                progress_tracking()

    # --------------------------------------------------------
    # STUDENT
    # --------------------------------------------------------

    elif role == "Student":

        if module is None:

            student_dashboard()

        else:

            back_to_main_portal()

            back_to_dashboard()

            if module == "myatt":

                student_attendance()

            elif module == "myacad":

                student_academics()

            elif module == "coding":

                student_coding()

            elif module == "myprogress":

                student_progress()

            elif module == "placement":

                student_placement()

            elif module == "recommend":

                student_recommendations()