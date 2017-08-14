import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

dim_file = 'dim_results.csv'
bright_file = 'bright_results.csv'
blur_file = 'blur_results.csv'

dim = pd.read_csv(dim_file)
bright = pd.read_csv(bright_file)
blur = pd.read_csv(blur_file)
dim_loss = np.true_divide(dim['loss'],dim['total_detected'])
dim_indoor = dim_loss[0::2]
dim_outdoor = dim_loss[1::2]
indoor = plt.plot(range(1,10),dim_indoor,label='Indoors')
outdoor = plt.plot(range(1,10),dim_outdoor,label='Outdoors')
plt.title('Dimming to black OpenFace performance')
plt.xlabel('Percentage reduction 10%-90%')
plt.ylabel('Avg L2 Loss for detected images')
#plt.axis([0, 9, 0, 150])
plt.legend()
plt.savefig('dim_plot.png')
plt.clf()
bright_loss = np.true_divide(bright['loss'],bright['total_detected'])
bright_indoor = bright_loss[0::2]
bright_outdoor = bright_loss[1::2]
indoor = plt.plot(range(1,len(bright_loss)/2+1),bright_indoor,label='Indoors')
outdoor = plt.plot(range(1,len(bright_loss)/2+1),bright_outdoor,label='Outdoors')
plt.title('Brightness to white OpenFace performance')
plt.xlabel('Percentage reduction 10%-90%')
plt.ylabel('Avg L2 Loss for detected images')
#plt.axis([0, 9, 0, 300])
plt.legend()
plt.savefig('bright_plot.png')
plt.clf()
blur_loss = np.true_divide(blur['loss'],blur['total_detected'])
blur_indoor = blur_loss[0::2]
blur_outdoor = blur_loss[1::2]
indoor = plt.plot(range(len(blur_loss)/2),blur_indoor,label='Indoors')
outdoor = plt.plot(range(len(blur_loss)/2),blur_outdoor,label='Outdoors')
plt.title('Gaussian Blur OpenFace performance')
plt.xlabel('Standard Deviation')
plt.ylabel('Avg L2 Loss for detected images')
plt.axis([0, 3, 0, 50])
plt.legend()
plt.savefig('blur_plot.png')
