
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import os
from skimage.io import imread
import natsort
import pandas as pd



idx_to_label = {0: 'NORMAL', 1: 'CNV', 2: 'DR', 3: 'AMD',4: 'RVO', 5: 'OTHERS',6:'CSC'}

def center_crop(img, dim,x1,y1):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(x1), int(y1)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    if(mid_y-ch2 < 0):
        mid_y = ch2
    elif(mid_y+ch2 > height):
        mid_y = height - ch2 
    if(mid_x-cw2 < 0):
        mid_x = cw2
    elif(mid_x+cw2 > width):
        mid_x = width - cw2 

    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def getKey(item):
    for idx, label in idx_to_label.items():
        if label == item:
            return idx


def getValue(key):
    for idx, label in idx_to_label.items():
        if idx == key:
            return label






data = np.zeros((106,608, 608))
label = np.zeros((106,1))
ctlist=os.listdir('/home/hana/optima/exchange/hjebril/challenge/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set')

ctlist_segmentation =os.listdir('/home/hana/optima/exchange/hjebril/challenge/A. Segmentation/2. Groundtruths/a. Training Set/2. Nonperfusion Areas')
# ctlist  = ['10102.bmp','10125.bmp','10162.bmp','10031.bmp','10241.bmp','10163.bmp']
ctlist=natsort.natsorted(ctlist)
ctlist_segmentation=natsort.natsorted(ctlist_segmentation)


common_list = set(ctlist_segmentation).intersection(ctlist)

data = np.zeros((len(common_list),608, 608))
label = np.zeros((len(common_list),1))
for idx, filename in enumerate(common_list):
    image1 = imread('/home/hana/optima/exchange/hjebril/challenge/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set/'+filename)

    segmentation_image = imread('/home/hana/optima/exchange/hjebril/challenge/A. Segmentation/2. Groundtruths/a. Training Set/2. Nonperfusion Areas/'+filename)

    # Plotting
    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(image1)
    ax.grid()
    # Defining the cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color = 'r', linewidth = 1)
    # Creating an annotating box
    annot = ax.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                        bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                        arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False)
    # Function for storing and showing the clicked values
    x_coord = []
    y_coord = []
    def onclick(event):
        global coord
        x_coord.append(event.xdata)
        y_coord.append(event.ydata)
        x = event.xdata
        y = event.ydata
        
        # printing the values of the selected point
        print([x,y]) 
        annot.xy = (x,y)
        text = "({:.2g}, {:.2g})".format(x,y)
        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw() #redraw the figure
        
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # Unzipping the coord list in two different arrays
    x1 = x_coord[0]
    y1 = y_coord[0]
    print(x1, y1)
    image = center_crop(image1, (608,608),x1, y1)
    segment_crop = center_crop(segmentation_image, (608,608),x1, y1)
    print('cropped',image.shape)
    # plt.imshow(image)
    # plt.show()
    im = Image.fromarray(image)
    im.save('/home/hana/optima/exchange/hjebril/challenge/C. Diabetic Retinopathy Grading/1. Original Images/6m_cropped_enface/'+filename)


    im2 = Image.fromarray(segment_crop)
    im2.save('/home/hana/optima/exchange/hjebril/challenge/C. Diabetic Retinopathy Grading/1. Original Images/6m_cropped_segmentation/'+filename)


#     data[idx,:] = image
#     # read labels from the excel file
#     dfs = pd.read_excel('Text labels 6m.xlsx', sheet_name='Sheet1')
#     x = dfs.loc[dfs['ID'] == int(filename.split('.')[0])]['Disease'].values
# #     print(filename.split('.')[0])
# #     print(x)
#     label[idx,:] = getKey(x[0])