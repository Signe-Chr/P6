import numpy as np
import os
from Image_processor_lung import ImageDims

dir = r"C:\Users\madsd\.cache\kagglehub\datasets\tawsifurrahman\covid19-radiography-database\versions\5\COVID-19_Radiography_Dataset"

def load_some_npy_data(ProcessedRoot, dims = ImageDims, number_of_each_class=1345):
    normal_dir = np.array(os.listdir(os.path.join(ProcessedRoot, "Normal")))
    pneumonia_dir = np.array(os.listdir(os.path.join(ProcessedRoot, "Viral Pneumonia")))
    covid_dir = np.array(os.listdir(os.path.join(ProcessedRoot, "COVID")))

    normal_dir_load = normal_dir[np.random.choice(len(normal_dir), number_of_each_class)]
    covid_dir_load = covid_dir[np.random.choice(len(covid_dir), number_of_each_class)]
    pneumonia_dir_load = pneumonia_dir[np.random.choice(len(pneumonia_dir), number_of_each_class)]

    size = dims[0] * dims[1]

    normal = np.zeros((number_of_each_class, size))
    covid = np.zeros((number_of_each_class, size))
    pneumonia = np.zeros((number_of_each_class, size))

    for i, filename in enumerate(normal_dir_load):
        normal[i] = np.load(os.path.join(ProcessedRoot, "Normal", filename))
    for i, filename in enumerate(covid_dir_load):
        covid[i] = np.load(os.path.join(ProcessedRoot, "COVID", filename))
    for i, filename in enumerate(pneumonia_dir_load):
        pneumonia[i] = np.load(os.path.join(ProcessedRoot, "Viral Pneumonia", filename))
    
    return normal, covid, pneumonia

normal_all, covid_all, pneumonia_all = load_some_npy_data(ProcessedRoot = os.path.join(dir, "Processed"))
