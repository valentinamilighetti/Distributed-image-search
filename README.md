# Ricerca di immagini simili con Hadoop, Spark, PyTorch e Milvus

Questo progetto realizza un motore di ricerca per immagini su un database distribuito, sfruttando tecniche di image embedding e un database vettoriale (Milvus) per l’indicizzazione e la ricerca per similarità.

---

## Architettura del Progetto

- **Hadoop** su cluster a 2 VM (1 master - `namenode`, 1 worker - `datanode1`)
- **Spark 3.5.6** per il calcolo distribuito e l'archiviazione delle immagini su hdfs
- **PyTorch** per la generazione degli embedding delle immagini
- **Milvus** come database vettoriale per l’indicizzazione e la ricerca
- **FastAPI** come backend per l’esposizione delle API
- **Jupyter Notebook** per il calcolo degli embedding e l'archiviazione su Milvus

---

## Configurazione Cluster

### Hadoop
- Sistema operativo: **Lubuntu**
- Configurazione con NAT 
- Installazione di Hadoop 3.4.1 su `namenode` e `datanode1`

### Spark
- Versione: **3.5.6**
- Integrazione con Hadoop via variabili d’ambiente 
- Configurazione di `spark-env.sh` e file `slaves`

### PyTorch + PySpark
- Creazione ambiente virtuale con **Python 3.11**
- Installazione pacchetti: `pyspark`, `pyarrow`, `numpy`, `torch`, `torchvision`, `pymilvus`
- Avvio di **Jupyter Notebook** dal nodo master

### Milvus
- Supporto a due modalità:
  - **Milvus Lite** via `pymilvus`
  - **Milvus Standalone** tramite Docker Compose
- Tre container attivi: `milvus-standalone`, `etcd`, `minio`

### Backend API
- Realizzato con **FastAPI**
- Avvio del server con:
  ```bash
  uvicorn main:app --reload

## Dataset

Il progetto utilizza il dataset Flickr30k Images disponibile su Kaggle
.
Le immagini vengono caricate su HDFS e indicizzate in Milvus dopo l’estrazione degli embedding.

## Risorse Utili

Guida configurazione Hadoop

Installazione Spark

Milvus Quickstart

Image Embedding per ricerca

PyTorch + PySpark distribuito (NVIDIA)
