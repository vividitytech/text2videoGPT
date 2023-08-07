import os
import imageio
import cv2
import numpy as np
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def getTextFiles(a_dir, ext='.txt'):
    out = []
    for file in os.listdir(a_dir):
        if file.endswith(".txt"):
            out.append(os.path.join(a_dir, file))
    return out

def load2(filename):
    data = []
    cap = cv2.VideoCapture(filename)
 
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
 
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            done = True
        # Display the resulting frame
        #cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # Break the loop
        else: 
            break
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (32, 32)#(width, height)
 
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        data.append(resized)

def loadvideo(filename, w=32, h=32, ft='ffmpeg'):
    video = []
    vid = imageio.get_reader(filename, ft)
    # movie = np.array([im for im in vid.iter_data()], dtype=np.uint8)
    for image in vid.iter_data():
        #video.append(image.resize((32, 32)))
        video.append(cv2.resize(image, (w, h),interpolation = cv2.INTER_AREA))
        #print(image.mean())
    return np.array(video,dtype=np.uint8)

datapath = "/home/gangchen/Downloads/project/datasets/UCF-101"

def loadData(datapath, width=72, height=72):
    dirs = get_immediate_subdirectories(datapath)
    data = []
    for d in dirs:
        files = getTextFiles(os.path.join(datapath, d))
        for f in files:
            file_in = open(f, 'r')
            row = dict()
            for line in file_in:
                a,b = line.rstrip().split(',')
            
                #row['question'] = a
                #row['answer'] = b
                #data.append(row)
                #row.clear()
                video = loadvideo(b, width,height)
                data.append({'question': a, 'answer':video, 'name':d})
            file_in.close()
    return data

def getlabels(data):
    names = []
    for kv in data:
        names.append(kv['name'])
    uniqueNames = set(names)
    labels = dict()
    i = 0
    for name in uniqueNames:
        labels[name] = i
        i = i + 1

    return labels
