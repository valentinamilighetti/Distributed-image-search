# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymilvus import connections, Collection
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pyarrow.fs as fs
import os, sys
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = None
collection = None
transform = None

def get_embedding(image: Image.Image) -> np.ndarray:
    """Converte un'immagine in embedding di dim 2048 con ResNet50"""
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = feature_extractor(tensor).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)  # normalizza per COSINE

def search_similar_images(query_img: Image.Image, topk=5):
    """Cerca immagini simili in Milvus data un'immagine di query"""
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

@app.on_event("startup")
async def startup_event():
    global device, feature_extractor, transform, collection

    # inizializza il modello ResNet50 per il calcolo locale degli embedding
    try:
        print("Connessione a Milvus in corso...")
        connections.connect("default", host="192.168.100.4", port="19530", timeout=10)
        collection = Collection("image_embeddings_spark")
        collection.load()
        print("Connessione a Milvus riuscita.")
    except Exception as e:
        print(f"Errore durante la connessione a Milvus: {e}")
        sys.exit("Server terminato: errore di inizializzazione")

    # Inizializza la connessione a Milvus
    try:
        print("caricamento del modello ResNet50...")
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])  
        feature_extractor.to(device)
        feature_extractor.eval()
        transform = weights.transforms() 
        print("Caricamento del modello eseguito")
    except Exception as e:
        print(f"Errore durante l'inizializzazione del modello: {e}")
        sys.exit("Server terminato: errore di inizializzazione")
    
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def read_index():
    file_path = os.path.join("/home/hadoopuser", "static", "index.html")
    return FileResponse(file_path)

# endpoint per ottenere file immagine da HDFS come risposta HTTP
@app.get("/image/")    
def get_image(path: str):
    try:
        with hdfs.open_input_file(path) as f:
            data = f.read()
        return StreamingResponse(BytesIO(data), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

# endpoint per ricerca di immagini simili
@app.post("/search_similar")
async def search_similar_images_api(file: UploadFile = File(...), count: int=6):
    try:
        # Leggi i dati dell'immagine e convertili in un oggetto PIL.Image
        image_data = await file.read()
        query_img = Image.open(BytesIO(image_data)).convert("RGB")

        # Esegui la ricerca sul db
        results = search_similar_images(query_img, topk=count)

        # restituisce i risultati come JSON
        response_data = []
        for hit in results:
            response_data.append({
                "path": hit.entity.get("path"),
                "similarity": hit.distance
            })

        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
