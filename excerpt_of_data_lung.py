import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from Image_processor_lung import ImageDims
from load_data_lung import dir

def plot_example(Label="Viral Pneumonia", OutputDims = ImageDims):
    Path = rf"{dir}\{Label}"
    ImageDir = os.listdir(rf"{Path}\images")
    MaskDir = os.listdir(rf"{Path}\masks")

    image = Image.open(rf"{Path}\images\{ImageDir[4]}").convert('L')
    mask = Image.open(rf"{Path}\masks\{MaskDir[4]}").convert('L')
    image = np.array(image.resize(mask.size, resample=1)).astype(np.float32)
    mask = np.array(mask).astype(np.float32) / 255.0
    image2 = image * mask
    m = image2.shape[0] 
    for j in range(m):
        if image2[j,:].any():
            image2 = np.delete(image2, range(j), 0)
            break
    for j in range(m):
        if image2[-j-1,:].any():
            image2 = np.delete(image2, range(-j,0), 0)
            break
    for j in range(m):
        if image2[:,j].any():
            image2 = np.delete(image2, range(j), 1)
            break
    for j in range(m):
        if image2[:,-j-1].any():
            image2 = np.delete(image2, range(-j,0), 1)
            break
        
    image2 = np.array(Image.fromarray(image2).resize(OutputDims))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Resized Input Image')
    ax1.axis('off')

    ax2.imshow(mask, cmap=plt.cm.gray)
    ax2.set_title('Mask of Lungs')
    ax2.axis('off')

    ax3.imshow(image2, cmap=plt.cm.gray)
    ax3.set_title('Masked Image')
    ax3.axis('off')

    plt.show()
    
plot_example()