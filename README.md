# Face Recognition API

A simple FastAPI-based face recognition service using Python and `face_recognition`.

## 🚀 Features
- Detect and recognize faces from uploaded images  
- Compare new faces with stored known faces  
- Easy to integrate via REST API  

## 🛠 Installation
```bash
git clone <your-repo-url>
cd face
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

▶️ Run the server

uvicorn main:app --reload

Server runs at: http://127.0.0.1:8000

🧠 Requirements

Python 3.11 recommended

Libraries: face_recognition, fastapi, uvicorn, dlib, numpy, pillow

Note:
face_recognition currently doesn’t support Python 3.12+ — use Python 3.11 for stable operation.