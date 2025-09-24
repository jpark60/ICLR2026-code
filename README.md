# How To Run the Code
## Download dataset
1. [CICAndMal2017](https://www.unb.ca/cic/datasets/andmal2017.html)
```
mkdir -p /scratch/Malware/CICAndMal
wget -c -i data/CICAndMal2017/cic_url.txt -P /scratch/Malware/CICAndMal
```
2. [IoT23](https://www.stratosphereips.org/datasets-iot23)
```
mkdir -p /scratch/Malware/iot23
wget -c -i data/IoT23/iot23_url.txt -P /scratch/Malware/iot23
```

## Data Preprocess
CICAndAMl2017 : data/CICAndMal2017/CIC_preprocess.py (saves to /scratch/Malware/CICAndMal/processed_data/)
IoT23 : data/IoT23/store_by_capture.py -> capture_preprocess.py (saves to /scratch/Malware/iot23/data/)

## Usage
- training with default arguments(kmeans exemplar selection)
```
python main_all.py 
```