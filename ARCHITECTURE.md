# Project Architecture

Data sources
    ↓
Unified Student Dataset (CSV / future DB/API)
    ↓
Data Cleaning + Feature Engineering
    ↓
Student Success Score (transparent weighted score)
    ↓
AI Risk Model (Random Forest probability)
    ↓
Risk Flags + Segmentation
    ↓
Streamlit Dashboard
    ↓
Faculty Action / Intervention Recommendation

Score weights:
Academic 30%
Attendance 20%
LMS 10%
Engagement 10%
Placement 15%
Skills 10%
Feedback 5%
