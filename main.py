import urllib.request
import zipfile
from reader import DataReader
import requests
import os

IRIS_URL = 'https://storage.googleapis.com/kaggle-data-sets/19/420/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230503%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230503T133048Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a787874603a9b9f10dc8b7f351a9b58abf1b93440250dfaf931c7589424d84f3ac88252fb53c6574334b4aaf6a8a64a8219d26ec15ce2242dd25674d96fb2837a17a300e71acd42a71d310d08fcbd931c077a0556d3ec1de462e0e9e0904d9c3b63f97f87bbfd26f7d5a576f19873ad886b4553e0e9a5e3e1a10678d2719b9876493d3371dca071648bebb229878d88bdeaa67bea6be74fdde832f15f30cab24ebdb2e55892e99e4a84f6b4431086407c203652c8904d1445f7d12f5191cbda590aa734a88b1751ba3cc4d73cb5f91c54fb65ab86f53ec966b6c4e52718f9a8a12754c3a7e8a802043c0a8ccaf82b336b6b3b9df113dca99b213d2e68fe3f553'
IRIS_PATH = 'datasets'

def fetch_iris_data(iris_url=IRIS_URL, iris_path=IRIS_PATH):
    os.makedirs(iris_path, exist_ok=True)
    zip_path = os.path.join(iris_path, "iris.zip")
    urllib.request.urlretrieve(iris_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(iris_path)
        os.rename(f'{IRIS_PATH}/Iris.csv', f'{IRIS_PATH}/iris.csv')
    os.remove(f'{IRIS_PATH}/database.sqlite')
    os.remove(f'{IRIS_PATH}/iris.zip')

def load_iris_data(iris_path=IRIS_PATH):
    iris_file = os.path.join(iris_path, 'iris.csv')
    return DataReader.load_data_with_labels(iris_file, 'Species')
    

fetch_iris_data()
train_x, train_y = load_iris_data()
