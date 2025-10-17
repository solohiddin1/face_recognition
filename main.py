import cv2
from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np

app = FastAPI()

# Example: load known face

known_encodings = []
known_names = []

# First face
img1 = face_recognition.load_image_file("photo.jpg")
known_encodings.append(face_recognition.face_encodings(img1)[0])
known_names.append("Solohiddin")

# Second face
img2 = face_recognition.load_image_file("ozodbek.jpg")
known_encodings.append(face_recognition.face_encodings(img2)[0])
known_names.append("Ozodbek")

# # Third face
# img3 = face_recognition.load_image_file("alice.jpg")
# known_encodings.append(face_recognition.face_encodings(img3)[0])
# known_names.append("Alice")


# known_image = face_recognition.load_image_file("photo.jpg")
# known_encoding = face_recognition.face_encodings(known_image)[0]


@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    # Load uploaded image
    img = face_recognition.load_image_file(file.file)

    # Encode uploaded face(s)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) == 0:
        return {"error": "No face found"}

    # Compare with known faces
    matches = face_recognition.compare_faces(known_encodings, encodings[0])
    distance = face_recognition.face_distance(known_encodings, encodings[0])[0]

    return {
        "match": bool(matches[0]),
        "confidence": float(1 - distance)
    }

@app.get("/recognize-camera/")
def recognize_from_camera():
    cap = cv2.VideoCapture(0)
    result = {"match": False, "name": None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            if True in matches:
                result = {"match": True, "name": known_names[matches.index(True)]}
                cap.release()
                cv2.destroyAllWindows()
                return result

        # Press 'q' to stop manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return result




# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# app = FastAPI()

# class Item(BaseModel):
#     id: int
#     name: str
#     description: str

# @app.post("/items/", response_model=Item)
# def create_item(item: Item):
#     return item

# @app.get("/items/", response_model=List[Item])
# def read_items():
#     return []