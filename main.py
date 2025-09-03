# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymilvus import connections, Collection
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import requests
import pyarrow.fs as fs
import os
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Importa le tue funzioni esistenti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(image: Image.Image) -> np.ndarray:
    """Converte un'immagine in embedding 2048-dim con ResNet50"""
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)  # normalizza per COSINE

def search_similar_images(query_img: Image.Image, topk=5):
    emb = get_embedding(query_img)
    results = collection.search(
        data=[emb.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=topk,
        output_fields=["path"]
    )
    return results[0]  # ritorna la prima lista di risultati

app = FastAPI()
hdfs = fs.HadoopFileSystem("namenode", port=9000, user="hadoopuser")

# Inizializza la connessione a Milvus e il modello ResNet50
# Puoi farlo una volta all'avvio dell'app per efficienza
@app.on_event("startup")
async def startup_event():
    connections.connect("default", host="192.168.100.4", port="19530")
    global collection
    collection = Collection("image_embeddings_spark")
    collection.load()
    
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def read_index():
    file_path = os.path.join("/home/hadoopuser", "static", "index.html")
    return FileResponse(file_path)
    
@app.get("/image/")    
def get_image(path: str):
    """
    Restituisce un file immagine da HDFS come risposta HTTP
    Esempio: /image/?path=/user/hadoopuser/flickr30k_images/xxx.jpg
    """
    try:
        with hdfs.open_input_file(path) as f:
            data = f.read()
        return StreamingResponse(BytesIO(data), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

@app.post("/search_similar")
async def search_similar_images_api(file: UploadFile = File(...), count: int=6):
    try:
        # Leggi i dati dell'immagine e convertili in un oggetto PIL.Image
        image_data = await file.read()
        query_img = Image.open(BytesIO(image_data)).convert("RGB")

        # Esegui la ricerca con la tua funzione
        results = search_similar_images(query_img, topk=count)

        # Prepara la risposta JSON
        response_data = []
        for hit in results:
            response_data.append({
                "path": hit.entity.get("path"),
                "similarity": hit.distance
            })

        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
