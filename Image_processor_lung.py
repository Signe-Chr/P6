import numpy as np
import os
from PIL import Image
from timeit import default_timer as timer
np.random.seed(42)

dir = r"C:\Users\madsd\.cache\kagglehub\datasets\tawsifurrahman\covid19-radiography-database\versions\5\COVID-19_Radiography_Dataset"

ImageDims = (164, 164)
Np = 500                #Numbers of pretraining, calibration, and test points for each lalbel
Nc = 500
Nt = 250

N = Np + Nc + Nt        #Total number of datapoints for each label

def image_processor(Label, N=0, OutputDims = ImageDims):        
    Path = rf"{dir}\{Label}"
    ImageDir = os.listdir(rf"{Path}\images")
    MaskDir = os.listdir(rf"{Path}\masks")

    RandomIndices = np.arange(len(ImageDir))
    np.random.shuffle(RandomIndices)

    if N == 0:                      #If no N argument is given, N is the number of images in the given folder.
        N = len(RandomIndices)

    output = np.zeros((N, OutputDims[0]*OutputDims[1]))

    for i in range(N):
        image = Image.open(rf"{Path}\images\{ImageDir[RandomIndices[i]]}").convert('L')
        mask = Image.open(rf"{Path}\masks\{MaskDir[RandomIndices[i]]}").convert('L')
        image = np.array(image.resize(mask.size, resample=1)).astype(np.float32)
        mask = np.array(mask).astype(np.float32) / 255.0
        image = image * mask

        m = image.shape[0]
        for j in range(m):
            if image[j,:].any():
                image = np.delete(image, range(j), 0)
                break
        for j in range(m):
            if image[-j-1,:].any():
                image = np.delete(image, range(-j,0), 0)
                break
        for j in range(m):
            if image[:,j].any():
                image = np.delete(image, range(j), 1)
                break
        for j in range(m):
            if image[:,-j-1].any():
                image = np.delete(image, range(-j,0), 1)
                break
        image = np.array(Image.fromarray(image).resize(OutputDims))
        image = image.flatten()
        output[i] = image
    return output

def preprocess_and_save(Label, OutputFolder, OutputDims = ImageDims, Flatten=False):        
    Path = rf"{dir}\{Label}"
    ImageDir = os.listdir(rf"{Path}\images")
    MaskDir = os.listdir(rf"{Path}\masks")

    os.makedirs(os.path.join(OutputFolder, Label), exist_ok=True)

    for i, filename in enumerate(ImageDir):
        image_path = os.path.join(Path, "images", filename)
        mask_path = os.path.join(Path, "masks", MaskDir[i])

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = np.array(image.resize(mask.size, resample=1)).astype(np.float32)
        mask = np.array(mask).astype(np.float32) / 255.0
        image = image * mask

        m = image.shape[0]   #Crop from top, bottom, left, and right until lung is encountered
        for j in range(m):
            if image[j,:].any():
                image = np.delete(image, range(j), 0)
                break
        for j in range(m):
            if image[-j-1,:].any():
                image = np.delete(image, range(-j,0), 0)
                break
        for j in range(m):
            if image[:,j].any():
                image = np.delete(image, range(j), 1)
                break
        for j in range(m):
            if image[:,-j-1].any():
                image = np.delete(image, range(-j,0), 1)
                break

        
        image = np.array(Image.fromarray(image).resize(OutputDims))
        image = image.flatten()

        np.save(os.path.join(OutputFolder, Label, f"{Label}-{i}.npy"), image)
    return

t0 = timer()
print(timer()-t0)