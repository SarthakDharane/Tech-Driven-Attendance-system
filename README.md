# TECH-DRIVEN ATTENDANCE SYSTEM

This project is a Face Recognition-based Smart Attendance System built using Flask, OpenCV, and Machine Learning.

Features

Admin Login Authentication
Face Registration System
Face Recognition for Attendance
Attendance Storage in CSV Files
Multi-Subject Attendance
Automatic Model Training
Attendance Retake Option
Real-time Camera Interface
Tech Stack
Python
Flask
OpenCV
Scikit-Learn
Joblib
Pandas
Waitress (Production WSGI Server)


Prerequisites
Make sure you have the following installed:
Python 3.x
pip
Virtual Environment (optional but recommended)
Git

Installation
1. Download the Project

Download the project from the provided link or copy the project folder.

2. Create Virtual Environment (Optional)

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate     # For Windows

3. Install Dependencies

pip install -r requirements.txt

4. Directory Structure

Make sure your project folder looks like this:

📁 Smart-Attendance-System
│
├── app.py               # Main Flask Application
├── requirements.txt     # Required Packages
├── haarcascade_frontalface_default.xml # Face Detection Cascade
├── static/             # Static Files
│   ├── faces/         # Registered User Faces
│   └── face_recognition_model.pkl # Trained Model
└── Attendance/         # Attendance CSV Files

How to Run the Project

Start the Flask Application

python app.py

Open your browser and navigate to:

http://127.0.0.1:5000/

Default Credentials

Username: ipsit_admin

Password: password

Usage

1. Login to the System

Register New Users under the Add User section.

Start Attendance by selecting the subject.

Attendance will automatically be marked if the registered face is detected.

Download the Attendance from the Attendance folder.

How to Retrain the Model

Every time a new user is added, the model will automatically retrain itself.

Important Notes

Make sure the haarcascade_frontalface_default.xml file is placed in the root directory.

The attendance data is stored inside the Attendance folder with the subject name and current date.

License
This project is for educational purposes only.

Author
SarthakDharane
