'''
Python program used to convert the csv annotations of the dataset d2 to YOLO annotations
Can be called from the main folder by running 'python tools/convert_d2_ann.py'
'''
import pandas as pd
import os
import shutil

commonPath = "datasets/d2/"
trainFolder = os.path.join(commonPath, "train/")
testFolder = os.path.join(commonPath, "test/")
valFolder = os.path.join(commonPath, "valid/")

trainCsvAnn = pd.read_csv(os.path.join(trainFolder, "_annotations.csv"))
testCsvAnn = pd.read_csv(os.path.join(testFolder, "_annotations.csv"))
valCsvAnn = pd.read_csv(os.path.join(valFolder, "_annotations.csv"))

def clean():
    '''
    Clean the label folders
    '''
    trainLabelFolder = os.path.join(trainFolder, "labels")
    testLabelFolder = os.path.join(testFolder, "labels")
    valLabelFolder = os.path.join(valFolder, "labels")

    # Delete the label folders
    if os.path.exists(trainLabelFolder):
        shutil.rmtree(trainLabelFolder)
    if os.path.exists(testLabelFolder):
        shutil.rmtree(testLabelFolder)
    if os.path.exists(valLabelFolder):
        shutil.rmtree(valLabelFolder)
    
    os.mkdir(trainLabelFolder)
    os.mkdir(testLabelFolder)
    os.mkdir(valLabelFolder)

def write_YOLO_annotation(row, folder):
    '''
    Write the YOLO annotation in a txt file
    '''
    # get the name if the image and replace the 'jpg' at the end by 'txt'
    filename = row["filename"][:-3] + "txt"
    filepath = folder + "labels/" + filename
    
    file = open(filepath, "a")
    
    # write the annotations into the file
    data = ""
    for i in range(1,6):
        data = data + " " + str(row.iloc[i]) 

    file.write(data + "\n")

    file.close()

def to_Yolo(df, folder):
    '''
    Convert the annotations to the YOLO format 
    '''
    # The original annotaion follows the scheme [image_name, width, height, class_name, xmin,ymin, xmax, ymax]
    # we want the following annotation scheme [class_id, xpos, ypos, width, height]
    
    # Get the original size of the images (we assume they all have the same)
    imWidth = df.iloc[1,1]
    imHeight = df.iloc[1,2]
    
    # Create the YOLO annotations 
    # drop the orginal image size columns, this is done to have the correct order on the annotation scheme
    df = df.drop(columns = ["width", "height"])
    
    # Transform the class_name field in class_id 
    df = df.sort_values(by=["class"])  # to ensure the factorize function encounter the classes in the same order

    df["class"], classes = pd.factorize(df["class"])
    print(classes) # To make sure all the classes follow the same order depending on the working folder
    
    # Create the x and y postitions of the bounding boxes
    df["xpos"] = df.apply(lambda row: ((row.xmin + row.xmax)/2)/imWidth, axis=1)
    df["ypos"] = df.apply(lambda row: ((row.ymin + row.ymax)/2)/imHeight, axis=1)
    
    # Create the width and height of the bounding boxes
    df["width"] = df.apply(lambda row: (row.xmax - row.xmin)/imWidth, axis=1)
    df["height"] = df.apply(lambda row: (row.ymax - row.ymin)/imHeight, axis=1)
    
    # remove the useless columns 
    df = df.drop(columns = ["xmin", "ymin", "xmax", "ymax"])

    # create the YOLO labels
    df.apply(lambda row: write_YOLO_annotation(row, folder), axis=1)

clean()
to_Yolo(trainCsvAnn, trainFolder)
to_Yolo(testCsvAnn, testFolder)
to_Yolo(valCsvAnn, valFolder)

