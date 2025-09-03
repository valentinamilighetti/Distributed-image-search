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

### Python e pacchetti necessari
- Creazione ambiente virtuale con **Python 3.11**
  ```bash
  sudo apt install python3.11 python3.11-venv
  python3.11 -m venv pytorch_env
  source pytorch_env/bin/activate 
  ```
- Installazione pacchetti:
  ```bash
  pip install pyspark==3.5.6
  pip install pyarrow==12.0.1
  pip install numpy==1.26.4 pandas
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install pymilvus

  # solo sul master
  pip install notebook
  pip install jupyter
  ```
- Avvio di **Jupyter Notebook** dal nodo master
  ```bash
  source pytorch_env/bin/activate 
  pyspark
  ```

### Spark
- Versione: **3.5.6**
- Integrazione con Hadoop via variabili d’ambiente 
- Configurazione di `spark-env.sh` e file `slaves`

### Milvus

La versione è **Milvus Standalone** tramite Docker Compose, che funziona tramite container attivi sul nodo master: `milvus-standalone`, `etcd`, `minio`

Installazione di Docker Compose (sul nodo master):
```bash
sudo apt-get update
sudo apt-get install \
  ca-certificates \
  curl \
  gnupg \
  lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  (lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
Per avviare il servizio:
```bash
sudo systemctl start docker
# Abilita l'avvio di Docker all'avvio del sistema
sudo systemctl enable docker
# Verifica che il servizio Docker sia in esecuzione
sudo systemctl status docker
# Verifica l’istallazione di docker con l’immagine hello world:
sudo docker run hello-world
# Gestire Docker come utente non root: creazione del gruppo docker
sudo groupadd docker
# Aggiungi l’user al gruppo
sudo usermod -aG docker $USER
newgrp docker
# Verifica: esegui hello world senza sudo:
docker run hello-world
# Verifica l’istallazione del plugin docker compose
docker compose version
```
Scaricare il file di configurazione Docker per Milvus:
```bash
mkdir -p ~/milvus
cd ~/milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
Fare riferimento al file [docker-compose.yml](docker-compose.yml)) per le modifiche da apportare per adattare le risorse della VM al funzionamento sul nodo master.

Avvia i container Milvus:
```bash
docker compose up -d
# Verifica lo stato dei container
docker compose ps
# Stop Milvus
docker compose down
```
Milvus è accessibile dalla porta 19530

### Backend API
- Realizzato con **FastAPI**
- Installazione:
  ```bash
  pip install fastapi "uvicorn[standard]" python-multipart
- Avvio del server con:
  ```bash
  uvicorn main:app --reload
  ```

## Dataset

Il progetto utilizza il dataset [Flickr30k Images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) e contiene circa 30000 immagini.

## Utilizzo

## Risorse Utili

- [Guida alla configurazione di Hadoop](https://medium.com/analytics-vidhya/setting-up-hadoop-3-2-1-d5c58338cba1)

- [Guida alla configurazione di Spark](https://medium.com/@redswitches/how-to-install-spark-on-ubuntu-965266d290d6)

- [Altra guida per Spark](https://aws.plainenglish.io/how-to-setup-install-an-apache-spark-3-1-1-cluster-on-ubuntu-817598b8e198)

- [Guida per pyspark per notebook](https://www.bmc.com/blogs/jupyter-notebooks-apache-spark/)

- [Milvus Quickstart](https://milvus.io/docs/it/quickstart.md)

- [Integrazione dei file Parquet con Milvus](https://milvus.io/it/blog/milvus-supports-apache-parquet-file-supports.md)

- [Image Embedding per image search (Pinecone)](https://www.pinecone.io/learn/series/image-search/)

- [Image Embedding per image search (Medium)](https://medium.com/thedeephub/image-embeddings-for-enhanced-image-search-f35608752d42)

- [Image Similarity (Hugging Face)](https://huggingface.co/blog/image-similarity)

- [Uso di PyTorch con pyspark (NVIDIA)](https://developer.nvidia.com/blog/distributed-deep-learning-made-easy-with-spark-3-4/)

- [Distribuited Training per Spark ML (databricks)](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/)

- [Esempio di integrazione PyTorch + PySpark distribuito (NVIDIA)](https://github.com/NVIDIA/spark-rapids-examples/tree/branch-23.06/examples/ML%2BDL-Examples/Spark-DL/dl_inference)
