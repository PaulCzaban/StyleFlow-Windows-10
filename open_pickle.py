import pickle
import numpy as np
import torch
from utils import Align_face_image

raw_w = pickle.load(open("data/sg2latents.pickle", "rb"))

raw_w['Latent'][000][0] = np.load('data/DW.npy')

raw_TSNE = np.load('data/TSNE.npy')

raw_attr = np.load('data/attributes.npy')

raw_lights2 = np.load('data/light.npy')

