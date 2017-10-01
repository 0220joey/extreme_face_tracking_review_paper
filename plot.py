import numpy as np
import pandas as pd
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
dim_file = 'cropped_dim_results.csv'
bright_file = 'cropped_bright_results.csv'
blur_file = 'cropped_blur_results.csv'
baseline_file = 'cropped_results.csv'
bar_width = 0.35
dim = pd.read_csv(dim_file)
bright = pd.read_csv(bright_file)
blur = pd.read_csv(blur_file)
baseline = pd.read_csv(baseline_file)
baseline_indoor = baseline['avg_loss'][0]
baseline_outdoor = baseline['avg_loss'][1]
baseline_indoor_count = baseline['total_detected'][0]
baseline_outdoor_count = baseline['total_detected'][1]
#dim_loss = np.true_divide(dim['loss'],dim['total_detected'])
dim_loss = dim['avg_error']
dim_count = dim['total_detected']
indoor_dim_count = np.insert(dim_count[0::2].values,0,baseline_indoor_count)
outdoor_dim_count = np.insert(dim_count[1::2].values,0,baseline_outdoor_count)
dim_indoor = np.insert(dim_loss[0::2].values,0,baseline_indoor)
dim_outdoor = np.insert(dim_loss[1::2].values,0,baseline_outdoor)
indoor = plt.plot(range(0,10),dim_indoor,label='Indoors')
outdoor = plt.plot(range(0,10),dim_outdoor,label='Outdoors')
plt.title('Dimming to black OpenFace performance')
plt.xlabel('Fade to Black 0%-90%')
plt.ylabel('Avg L2 Loss for detected images')
plt.xticks(np.arange(0, 10, 1.0))
#plt.axis([0, 9, 0, 150])
plt.legend()
plt.savefig('cropped_dim_plot.png')
plt.clf()
plt.title('Dimming to black OpenFace performance')
plt.xlabel('Fade to Black 0%-90%')
plt.ylabel('Number of detected Faces')
plt.bar(np.arange(10),indoor_dim_count,0.35,alpha=0.4,color='b',label='Indoors')
plt.bar(np.arange(10)+bar_width,outdoor_dim_count,0.35,alpha=0.4,color='r',label='Outdoors')
plt.xticks(np.arange(10) + bar_width / 2, \
        ('0', '10%', '20%', '30%', '40%', '50%','60%','70%','80%','90%'))
plt.legend()
plt.savefig('cropped_dim_count.png')
plt.clf()
bright_loss = bright['avg_error']
bright_count = bright['total_detected']
indoor_bright_count = np.insert(bright_count[0::2].values,0,baseline_indoor_count)
outdoor_bright_count = np.insert(bright_count[1::2].values,0,baseline_outdoor_count)
bright_indoor = np.insert(bright_loss[0::2].values,0,baseline_indoor)
bright_outdoor = np.insert(bright_loss[1::2].values,0,baseline_outdoor)
indoor = plt.plot(range(0,len(bright_loss)/2+1),bright_indoor,label='Indoors')
outdoor = plt.plot(range(0,len(bright_loss)/2+1),bright_outdoor,label='Outdoors')
plt.title('Brightness to white OpenFace performance')
plt.xlabel('Fade to white 0%-90%')
plt.ylabel('Avg L2 Loss for detected images')
plt.xticks(np.arange(0, len(bright_loss)/2 + 1, 1.0))
#plt.axis([0, 9, 0, 300])
plt.legend()
plt.savefig('cropped_bright_plot.png')
plt.clf()
plt.title('Brightness to white OpenFace performance')
plt.xlabel('Fade to White 0%-90%')
plt.ylabel('Number of detected Faces')
plt.bar(np.arange(10),indoor_bright_count,0.35,alpha=0.4,color='b',label='Indoors')
plt.bar(np.arange(10)+bar_width,outdoor_bright_count,0.35,alpha=0.4,color='r',label='Outdoors')
plt.xticks(np.arange(10) + bar_width / 2, \
        ('0', '10%', '20%', '30%', '40%', '50%','60%','70%','80%','90%'))
plt.legend()
plt.savefig('cropped_bright_count.png')
plt.clf()
#blur_loss = np.true_divide(blur['loss'],blur['total_detected'])
blur_loss = blur['avg_error']
blur_count = blur['total_detected']
indoor_blur_count = np.insert(blur_count[0::2].values,0,baseline_indoor_count)
outdoor_blur_count = np.insert(blur_count[1::2].values,0,baseline_outdoor_count)
blur_indoor = np.insert(blur_loss[0::2].values,0,baseline_indoor)
blur_outdoor = np.insert(blur_loss[1::2].values,0,baseline_outdoor)
indoor = plt.plot(range(0,len(blur_loss)/2+1),blur_indoor,label='Indoors')
outdoor = plt.plot(range(0,len(blur_loss)/2+1),blur_outdoor,label='Outdoors')
plt.title('Gaussian Blur OpenFace performance')
plt.xlabel('Standard Deviation')
plt.ylabel('Avg L2 Loss for detected images')
plt.xticks(np.arange(0, len(blur_loss)/2 + 1, 1.0))
#plt.axis([0, 3, 0, 50])
plt.legend()
plt.savefig('cropped_blur_plot.png')
plt.clf()
plt.title('Gaussian Blur OpenFace performance')
plt.xlabel('Standard Deviation')
plt.ylabel('Number of detected Faces')
plt.bar(np.arange(5),indoor_blur_count,0.35,alpha=0.4,color='b',label='Indoors')
plt.bar(np.arange(5)+bar_width,outdoor_blur_count,0.35,alpha=0.4,color='r',label='Outdoors')
plt.xticks(np.arange(5) + bar_width / 2, \
        ('No Blur', '0', '1', '2', '3'))
plt.legend()
plt.savefig('cropped_blur_count.png')
plt.clf()
