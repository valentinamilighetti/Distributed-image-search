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
- Configurazione VM con NAT 
- Installazione di Hadoop 3.4.1 su `namenode` e `datanode1`

Come cluster Hadoop è stato realizzato un cluster minimale costituito da 2 VM VirtualBox:
- 1 **VM master** chiamata `namenode`, che svolge sia il ruolo di _NameNode_ sia di _DataNode_. NOTA: in un cluster reale è consigliato mantenere il namenode e i datanodes su macchine distinte per una maggiore stabilità e resilienza del cluster.
- 1 **VM worker** chiamata `datanode1`, che svolge il ruolo di _DataNode_

Per l'installazione di Hadoop è stata seguita la seguente guida che spiega passo-passo l'installazione e la configurazione su 3 VM (1 NameNode e 2 DataNode) adattandola al caso specifico di 2 VM.
- [Guida alla configurazione di Hadoop](https://medium.com/analytics-vidhya/setting-up-hadoop-3-2-1-d5c58338cba1)

Di seguito sono descritti i passaggi chiave per l'installazione del cluster e in particolare sono riportati i comandi diversi rispetto alla guida: 
1. creare in VirtualBox una nuova **rete con NAT** a cui saranno connesse le 2 VM, per questo esempio è stata creata una rete privata con indirizzo _192.168.100.1/24_
#### VM Master NameNode
2. creare inizialmente 1 VM con una distribuzione Linux leggera basata su Ubuntu, ad esempio **Lubuntu 24.04**.
3. installare e configurare **SSH**, necessario per la comunicazione tra i nodi del cluster
4. installare **Java openjdk versione 11** (al posto della versione 8)
    ```bash
    sudo apt install openjdk-8-jdk
    ```
5. Scaricare ed estrarre **Hadoop versione 3.4.1** da apache.org
    ```bash
    # download archivio nella home
    sudo wget -P ~ https://dlcdn.apache.org/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz
    # estrarre e rinominare l\'archivio
    tar xzf hadoop-3.4.1.tar.gz
    mv hadoop-3.4.1 hadoop
    # modificare il file di configurazione `hadoop_env.sh`
    nano ~/hadoop/etc/hadoop/hadoop-env.sh
    # aggiungere nel file il riferimento a JAVA_HOME
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
    ```
6. modifica variabili di ambiente per hadoop
    ```bash
    #modificare le variabili di sistema con il comando
    sudo nano /etc/environment
    #aggiungere al file i seguenti valori
      PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/hadoop/bin:/usr/local/hadoop/sbin"
      JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64/"
    ```
7. creare un nuovo utente `hadoopuser` che verrà usato dai nodi del cluster e fornirgli tutti i permessi di root
8. in VirtualBox clonare la VM appena creata per creare la VM worker
9. modificare gli **hostname** di entrambe le VM usando il comando
    ```bash
    sudo nano /etc/hostname
    ```
  in questo caso la VM Master è stata chiamata *namenode*
  mentre la VM worker è stata chiamata *datanode1*
10. modificare in entrambe le VM il file hosts che associa gli indirizzi ip delle macchine con il loro hostname. 
    ```bash
    # ottenere l\'indirizzo ip associato a entrambe le VM
    ip addr
    #modificare il file hosts in ciascuna VM
    sudo nano /etc/hosts
    ```
    ad esempio il file hosts in questo caso è il seguente (da adattare alla configurazione specifica)
    ```
    192.168.100.4 namenode
    192.168.100.5 datanode1
    ```
11. in ciascuna VM creare la chiave SSH e condividerla con l'altra VM
12. da finire....
### Spark
- Versione: **3.5.6**
- Integrazione con Hadoop via variabili d’ambiente 
- Configurazione di `spark-env.sh` e file `slaves`

### PyTorch + PySpark
- Creazione ambiente virtuale con **Python 3.11**
- Installazione pacchetti: `pyspark`, `pyarrow`, `numpy`, `torch`, `torchvision`, `pymilvus`
- Avvio di **Jupyter Notebook** dal nodo master

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
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
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
