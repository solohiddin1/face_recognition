from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np

app = FastAPI()

# Example: load known face
known_image = face_recognition.load_image_file("photo.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    # Load uploaded image
    img = face_recognition.load_image_file(file.file)

    # Encode uploaded face(s)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) == 0:
        return {"error": "No face found"}

    # Compare with known face
    matches = face_recognition.compare_faces([known_encoding], encodings[0])
    distance = face_recognition.face_distance([known_encoding], encodings[0])[0]

    return {
        "match": bool(matches[0]),
        "confidence": float(1 - distance)
    }


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