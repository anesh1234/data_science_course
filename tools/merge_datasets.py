'''
This program is used to merge all 3 datasets
It can move the the images and the annotations to the unified dataset folder.
It can also change the labels of the original annotations (in YOLO format) to merge all classes
It can also separate the 3 datasets into train, test and valid folders with the separation 70%, 20%, 10% 
N.B: each dataset is separated then merged 


This program does not unify the image sizes.
It is handled by the YOLO model by filling the imgsz parameter
For instance, the size of the images of each datasets are:
    - d1: 640x640
    - d2: 512x256
    - d3: 640x342
'''

'''
Read this to understand how the classes were handled

Classes in d1
['Abcess', 'Badly Decayed', 'Caries', 'Crown', 'Normal', 'Overhang', 'Post', 'RCT', 'Restoration']
   0:3           1:7            2:7     3:6      4:99        5:4      6:6     7:5         8:6

Classes in d2
['Cavity', 'Fillings', 'Impacted Tooth', 'Implant']
    0:7        1:6          2:2             3:6

Classes in d3
['Normal', 'Caries', 'impacted tooth', 'Broken Down Crow/Root', 'Infected', 'Fractured']
    0:99      1:7           2:2                  3:0                4:3          5:1   

Classes in common between the datasets
Normal d1 --- d3 Normal
Impacted Tooth d2 --- d3 impacted tooth
Caries d1 --- d3 Caries --- d2 Cavity --- d1 Badly Decayed --> Tooth Decay

To remove some granularity
Abcess d1 --- d3 Infected -> Infection
Restoration d1 --- d1 Crown --- d1 Post --- d2 Fillings --- d2 Implants -> Restoration


More granularity
Unique classes: 16
['Abcess', 'Badly Decayed', 'Broken Down Crow/Root', 'Caries', 'Cavity', 'Crown', 'Fillings', 'Fractured', 'Impacted Tooth', 'Implant', 'Infected', 'Normal', 'Overhang', Post', 'RCT', 'Restoration']

Less granularity
Unique classes: 8
['Broken Down Crow/Root', 'Fractured', 'Impacted Tooth','Infection', 'Overhang', 'RCT', 'Restoration', 'Tooth Decay']
'''

import os
import shutil
import glob
from sklearn.model_selection import train_test_split

commonPath = "datasets/"

d1Path = os.path.join(commonPath, "d1/")
d2Path = os.path.join(commonPath, "d2/")
d3Path = os.path.join(commonPath, "d3/")

processingDir = os.path.join(commonPath, "processing")
d1Processing = os.path.join(processingDir, "d1/")
d2Processing = os.path.join(processingDir, "d2/")
d3Processing = os.path.join(processingDir, "d3/")

destination = os.path.join(commonPath, "final/")

def CreateProcessingDir():
    '''
    Create the folder that will host the result of the merge of the 3 datasets
    If the folder already exists, it will be removed and then recreated
    '''
    # Delete destination directory
    if os.path.exists(processingDir):
        shutil.rmtree(processingDir)

    if os.path.exists(destination):
        shutil.rmtree(destination)

    # Create the directories
    os.mkdir(processingDir)

    os.mkdir(d1Processing)
    os.mkdir(os.path.join(d1Processing, "images"))
    os.mkdir(os.path.join(d1Processing, "labels"))
    
    os.mkdir(d2Processing)
    os.mkdir(os.path.join(d2Processing, "images"))
    os.mkdir(os.path.join(d2Processing, "labels"))

    os.mkdir(d3Processing)
    os.mkdir(os.path.join(d3Processing, "images"))
    os.mkdir(os.path.join(d3Processing, "labels"))

    os.mkdir(destination)
    os.mkdir(os.path.join(destination, "train"))
    os.mkdir(os.path.join(destination, "train/images"))
    os.mkdir(os.path.join(destination, "train/labels"))
    os.mkdir(os.path.join(destination, "test"))
    os.mkdir(os.path.join(destination, "test/images"))
    os.mkdir(os.path.join(destination, "test/labels"))
    os.mkdir(os.path.join(destination, "valid"))
    os.mkdir(os.path.join(destination, "valid/images"))
    os.mkdir(os.path.join(destination, "valid/labels"))

def replaceLabelsOld(char, dataset):
    '''
    function kept to have more granularity
    Substitute char depending on the dataset used
    Before
    Unique classes: 16
    ['Abcess', 'Badly Decayed', 'Broken Down Crow/Root', 'Caries', 'Cavity', 'Crown', 'Fillings', 'Fractured', 'Impacted Tooth', 'Implant', 'Infected', 'Normal', 'Overhang', Post', 'RCT', 'Restoration']
    '''
    # Instanciate a dictionnary containing the values that need to be changed depending on the dataset being "translated"    
    # The substitutions are explaned on line 16
    if dataset == os.path.join(d1Processing, "labels"):
        # Values that change 2->3, 3->5, 4->11, 5->12, 6->13, 7->14, 8->15
        classes = {"2": "3", "3": "5", "4": "11", "5": "12", "6": "13", "7": "14", "8": "15"}
    elif dataset == os.path.join(d2Processing, "labels"):
        # Values that change 0->4, 1->6, 2->8, 3->9
        classes = {"0": "4", "1": "6", "2": "8", "3": "9"}
    elif dataset == os.path.join(d3Processing, "labels"):
        # Values that change 0->11, 1->3, 2->8, 4->10, 5->7
        classes = {"0": "11", "1": "3", "2": "8", "4": "10", "5": "7"}
    else:
        classes = {}
    
    if char in classes:
        return classes[char]
    else:
        return char

def replaceLabels(char, dataset):
    '''
    Substitute char depending on the dataset used
    Unique classes: 8
    ['Broken Down Crow/Root', 'Fractured', 'Impacted Tooth','Infection', 'Overhang', 'RCT', 'Restoration', 'Tooth Decay']
    '''
    # Instanciate a dictionnary containing the values that need to be changed depending on the dataset being "translated"    
    # The substitutions are explaned on line 16
    if dataset == os.path.join(d1Processing, "labels"):
        # Values that change 0->3, 1->7, 2->7, 3->6, 4->deleted, 5->4, 7->5, 8->6
        #   0:3           1:7            2:7     3:6      4:99        5:4      6:6     7:5         8:6
        classes = {"0": "3", "1": "7", "2": "7", "3": "6", "4": "99", "5": "4", "6": "6" ,"7": "5", "8": "6"}
    elif dataset == os.path.join(d2Processing, "labels"):
        # Values that change 0->7, 1->6, 3->6
        #    0:7        1:6          2:2             3:6
        classes = {"0": "7", "1": "6", "2": "2", "3": "6"}
    elif dataset == os.path.join(d3Processing, "labels"):
        # Values that change 0->deleted, 1->7, 3->1, 4->3, 5->1
        #     0:99      1:7           2:2                  3:0                4:3          5:1
        classes = {"0": "99", "1": "7", "2": "2", "3": "0", "4": "3", "5": "1"}
    else:
        classes = {}

    # 99 is used to signal that the line should be dropped
    # In this case only the "Normal" class is dropped
    
    if char in classes:
        return classes[char]
    else:
        return char


def unifyDataset(labelPath):
    '''
    Change the label of the annotations depending on the datasets
    '''
    for file in glob.glob("*", root_dir=labelPath):
        # Load the file in order to edit the labels
        with open(os.path.join(labelPath, file), "r") as read:
            tmp = read.readlines()
        
        for_write = []
        # Rewrite each label (the labels are the first 'word' of each line)
        for line in tmp:
            label = line[0] # Only works because there are no labels above 9
            # label is 99 only if the line has to be dropped
            newLabel = replaceLabels(label, labelPath) 
            if newLabel != "99":
                # Keep the line
                modified_line = newLabel + line[1:]
                for_write.append(modified_line)

        # Write the changes (Overwrite the file if something was there)
        with open(os.path.join(labelPath, file), "w") as write:
            write.writelines(for_write)

def processDataset(datasetPath, processingPath):
    '''
    Move the dataset found at datasetPath to processing path
    Then change the labels depending on the dataset used
    '''
    trainDir = os.path.join(datasetPath, "train/")
    testDir = os.path.join(datasetPath, "test/")
    valDir = os.path.join(datasetPath, "valid/")

    lstdir = [trainDir, testDir, valDir]

    for dir in lstdir:
        # Iterates through the images and label folders 
        for subdir in os.listdir(dir):
            # There is also a labels.cache file
            if subdir in ["images", "labels"]:
                subdirPath = os.path.join(dir, subdir)
                destpath = os.path.join(processingPath, subdir)
                shutil.copytree(subdirPath, destpath, dirs_exist_ok=True)
    # Change the labels depending on the dataset used
    unifyDataset(os.path.join(processingPath, "labels"))

def moveFilesToDest(src,filenames, dest):
    for file in filenames:
        shutil.copy(os.path.join(src,file), dest)

def splitDataset(datasetPath, destination, seed):
    '''
    Split the dataset into train, test adn validation datasets
    The proportions are 70%, 20% and 10% respectively
    '''
    destTrain = os.path.join(destination, "train")
    destTest = os.path.join(destination, "test")
    destVal = os.path.join(destination, "valid")

    srcImg = os.path.join(datasetPath, "images")
    srcAnn = os.path.join(datasetPath, "labels")

    X = os.listdir(srcImg)
    y = os.listdir(srcAnn)
    
    # For some reason the order of the files is different 
    # between X and y so we sort them to have the same order
    X.sort()
    y.sort()

    # Split the dataset into training and the rest
    X_train, X_tmp, y_train, y_tmp = train_test_split(X,y, test_size=0.3, random_state=seed)
    # Split the rest into test and validation
    X_test, X_valid, y_test, y_valid = train_test_split(X_tmp,y_tmp, test_size=0.66, random_state=seed)

    # move X_train and y_train to the train directory
    moveFilesToDest(srcImg, X_train, os.path.join(destTrain, "images"))
    moveFilesToDest(srcAnn, y_train, os.path.join(destTrain, "labels"))

    # move X_test and y_test to the test directory
    moveFilesToDest(srcImg, X_test, os.path.join(destTest, "images"))
    moveFilesToDest(srcAnn, y_test, os.path.join(destTest, "labels"))

    # move X_val and y_val to the valid directory
    moveFilesToDest(srcImg, X_valid, os.path.join(destVal, "images"))
    moveFilesToDest(srcAnn, y_valid, os.path.join(destVal, "labels"))
    

def writeDataOld():
    '''
    Write data.yaml file
    '''
    with open(os.path.join(destination,"data.yaml"), "a") as file:
        file.write("train: train/images\n")
        file.write("val: valid/images\n")
        file.write("test: test/images\n")
        file.write("\n")
        file.write("train_label_dir: train/labels\n")
        file.write("val_label_dir: valid/labels\n")
        file.write("test_label_dir: test/labels\n")
        file.write("\n")
        file.write("nc: 16\n")
        file.write("['Abcess', 'Badly Decayed', 'Broken Down Crow/Root', 'Caries', 'Cavity', 'Crown', 'Fillings', 'Fractured', 'Impacted Tooth', 'Implant', 'Infected', 'Normal', 'Overhang', Post', 'RCT', 'Restoration']\n")

def writeData():
    '''
    Write data.yaml file
    '''
    with open(os.path.join(destination,"data.yaml"), "a") as file:
        file.write("train: train/images\n")
        file.write("val: valid/images\n")
        file.write("test: test/images\n")
        file.write("\n")
        file.write("train_label_dir: train/labels\n")
        file.write("val_label_dir: valid/labels\n")
        file.write("test_label_dir: test/labels\n")
        file.write("\n")
        file.write("nc: 8\n")
        file.write("['Broken Down Crow/Root', 'Fractured', 'Impacted Tooth', 'Infection', 'Overhang', 'RCT', 'Restoration', 'Tooth Decay']")

CreateProcessingDir()
processDataset(d1Path, d1Processing)
processDataset(d2Path, d2Processing)
processDataset(d3Path, d3Processing)

splitDataset(d1Processing, destination, 42)
splitDataset(d2Processing, destination, 42)
splitDataset(d3Processing, destination, 42)

writeData()
