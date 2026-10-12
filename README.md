# EduTrack AI — Student Success Platform

A hackathon-ready prototype for the Smart Campus Analytics challenge.

## Features
- Unified student data view
- Student Success Score
- AI-style intervention risk prediction
- Risk flags
- Interactive dashboard
- Student 360° detail page
- Explainable score contributions
- Student segmentation
- Recommended interventions
- CSV upload for replacing demo data
- Filter by department, year and risk

## Run on Windows

1. Install Python.
2. Open Command Prompt in this folder.
3. Run:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`
5. Open the local URL shown in the terminal.

## Data
`data/students.csv` is synthetic demo data. For the hackathon, clearly label it as synthetic unless the organizers provide an approved dataset.

## Suggested demo flow
Overview → filter risky students → open Student Detail → explain score → show intervention recommendation → show Segmentation → show Model & Explainability.
