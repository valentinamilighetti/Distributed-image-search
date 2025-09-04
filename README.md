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
    sudo apt install openjdk-11-jdk
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
    ```
    aggiungere nel file il riferimento a JAVA_HOME
    ```bash   
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
    ```
6. modifica **variabili di ambiente** per hadoop
    ```bash
    #modificare le variabili di sistema con il comando
    sudo nano /etc/environment
    #aggiungere al file i seguenti valori
      PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/hadoop/bin:/usr/local/hadoop/sbin"
      JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64/"
    ```
7. creare un nuovo utente `hadoopuser` che verrà usato dai nodi del cluster e fornirgli tutti i permessi di root
8. in VirtualBox clonare la VM appena creata per creare la VM worker
#### VM Master e Worker
9. modificare gli **hostname** di entrambe le VM usando il comando
    ```bash
    sudo nano /etc/hostname
    ```
    in questo caso la VM Master è stata chiamata *namenode*
  mentre la VM worker è stata chiamata *datanode1*

10. modificare in entrambe le VM il file hosts che associa gli indirizzi ip delle macchine con il loro hostname. 

  ```bash
    # ottenere indirizzo ip associato a entrambe le VM
    ip addr
    #modificare il file hosts in ciascuna VM
    sudo nano /etc/hosts
  ```

  ad esempio il file hosts in questo caso è il seguente (da adattare alla configurazione specifica)

  ```bash
    192.168.100.4 namenode
    192.168.100.5 datanode1
  ```
11. in ciascuna VM creare la chiave SSH e condividerla con l'altra VM
12. da finire....

### Spark versione 3.5.6
Fare riferimento alla [guida](https://medium.com/@redswitches/how-to-install-spark-on-ubuntu-965266d290d6)
- Scaricare l'archivio Spark
  ```bash
  wget https://archive.apache.org/dist/spark/spark-3.5.6/spark-3.5.6.tgz
  ```
- Creare una directory dedicata dove estrarre il file tar
  ```bash
  mkdir ~/spark
  mv spark-3.5.6.tgz spark/
  cd ~/spark
  tar -xvzf spark-3.5.1.tgz
  ```
- Configurazione delle variabili d'ambiente nel file Bash:
  ```bash
  nano ~/.bashrc

  # aggiungere le seguenti variabili sul nodo master
  export HADOOP_HOME="/usr/local/hadoop"
  export HADOOP_COMMON_HOME=$HADOOP_HOME
  export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
  export HADOOP_HDFS_HOME=$HADOOP_HOME
  export HADOOP_MAPRED_HOME=$HADOOP_HOME
  export HADOOP_YARN_HOME=$HADOOP_HOME
  export SPARK_HOME=~/spark/spark-3.5.6-bin-hadoop3
  export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
  export PYSPARK_PYTHON=~/pytorch_env/bin/python3
  export PYSPARK_DRIVER_PYTHON=jupyter
  export PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip=192.168.100.4 --no-browser --port=8889'
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
  export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob):$HADOOP_CONF_DIR

  # aggiungere le seguenti variabili sul nodo worker
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
  export SPARK_HOME=~/spark/spark-3.5.6-bin-hadoop3
  export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
  export PYSPARK_PYTHON=~/pytorch_env/bin/python3

  # aggiornare le variabili su entrambi i nodi
  source ~/.bashrc
  ```
- Configurazione del file `spark-env.sh` sul nodo master
  ```bash
  cd ~/spark/spark-3.5.6 …../conf$ 
  cp spark-env.sh.template spark-env.sh
  sudo nano spark-env.sh
  
  #aggiungere 
  export SPARK_MASTER_HOST=192.168.0.4
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
  ```
- Creare il file `slaves` sul nodo master e inserire i nomi di master e slave:
  ```bash
  namenode
  Datanode1
  ```
- Per avviare master e slave:
  ```bash
  start-all.sh
  ```

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

### Milvus

La versione è **Milus Standalone** tramite Docker Compose, che funziona tramite 3 container attivi sul nodo master: `milvus-standalone`, `etcd` e `minio`

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
Fare riferimento al file [docker-compose.yml](docker-compose.yml) per le modifiche da apportare per adattare le risorse della VM al funzionamento sul nodo master.

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

## Calcolo degli embedding e caricamento su Milvus
Dopo il download del dataset è necessario il suo caricamento su hdfs, attraverso il comando 
```bash
hadoop distcp file:///home/hadoopuser/flickr30k_images hdfs:///user/hadoopuser/flickr30k_images
```
Le fasi successive sono state svolte attraverso il [notebook Jupyter](image_embedding_spark.ipynb)

### Calcolo degli embedding
- È stato utilizzato [resnet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), un modello pre-addestrato già presente su torchvision.
  ```bash
  import torch
  import torchvision.models as models

  weights = models.ResNet50_Weights.DEFAULT
  model = models.resnet50(weights=weights)
  state_dict = model.state_dict()
  torch.save(state_dict, "/tmp/resnet50_statedict2.pth")

  sc.addFile("/tmp/resnet50_statedict2.pth")
  ```
- Una volta avviata una SparkSession con 2 esecutori (il namenode e il datanode1), viene caricato il dataset da hdfs
  ```bash
  df = spark.read.format("binaryFile").load("hdfs:///user/hadoopuser/flickr30k_images/flickr30k_images/").select("path", "content")
  ```
- 

### Salvataggio degli embedding dal file Parquet al database vettoriale Milvus
- Inizialmente, viene caricato il file Parquet da hdfs
  ```bash
  df_from_parquet = spark.read.parquet("hdfs:///user/hadoopuser/flickr_image_embeddings_parquet/")
  ```


## Ricerca delle immagini per similarità

## Come Eseguire il Progetto
Di seguito sono illustrati i vari passaggi necessari per eseguire la ricerca per similarità delle immagini.
1. Come primo passaggio, avviare hdfs, yarn, Milvus e pyspark sul master:
    ```bash
    start.dfs.sh
    start-yarn.sh
    start-all.sh
    cd ~/milvus
    docker compose up -d
    ```
2. Aprire Jupyter Notebook ed eseguire tutte le celle di [image_embedding_spark.ipynb](image_embedding_spark.ipynb)
    ```bash
    source pytorch_env/bin/activate
    pyspark
    ```
3. Avviare FastAPI
    ```bash
    uvicorn main:app --reload
    ```
4. Collegarsi a [127.0.0.1:8000](http://127.0.0.1:8000/)
5. Caricare un'immagine e selezionare il numero di immagini da mostrare ad output
6. Visualizzare i risultati

Nota: il passaggio 2 viene eseguito solo la prima volta, perchè successivamente gli embedding delle immagini del dataset saranno già presenti nel dataset Milvus.

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
