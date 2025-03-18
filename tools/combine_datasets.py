# TODO: create function to write README.md an data.yaml
'''
The goal of the program is to combine all 3 datasets

Varying image size depending on the dataset (maybe, 
conflic in classes 

Unify the size:
    - d1: 640x640
    - d2: 512x256
    - d3: 640x342
'''

'''
Read this to understand how the classes were handled

Classes in d1
['Abcess', 'Badly Decayed', 'Caries', 'Crown', 'Normal', 'Overhang', 'Post', 'RCT', 'Restoration']

Classes in d2
['Cavity', 'Fillings', 'Impacted Tooth', 'Implant']

Classes in d3
['Normal', 'Caries', 'impacted tooth', 'Broken Down Crow/Root', 'Infected', 'Fractured']


lower/upper case should not matter
Classes in common between the datasets
Normal d1 --- d3 Normal
Impacted Tooth d2 --- d3 impacted tooth
Caries d1 --- d3 Caries


Unique classes (16)
['Abcess', 'Badly Decayed', 'Broken Down Crow/Root', 'Caries', 'Cavity', 'Crown', 'Fillings', 'Fractured', 'Impacted Tooth', 'Implant', 'Infected', 'Normal', 'Overhang', Post', 'RCT', 'Restoration']
'''
import os
import shutil
import glob

commonPath = "datasets/"

destination = os.path.join(commonPath, "final/")

destTrain = os.path.join(destination, "train")
destTest = os.path.join(destination, "test")
destVal = os.path.join(destination, "valid")
destDirs = [destTrain, destTest, destVal]

def createDest():
    '''
    Create the folder that will host the result of the merge of the 3 datasets
    If the folder already exists, it will be removed and then recreated
    '''
    # Delete destination directory
    if os.path.exists(destination):
        shutil.rmtree(destination)

    # Create the directories
    os.mkdir(destination)

    os.mkdir(destTrain)
    os.mkdir(os.path.join(destTrain, "images"))
    os.mkdir(os.path.join(destTrain, "labels"))
    
    os.mkdir(destTest)
    os.mkdir(os.path.join(destTest, "images"))
    os.mkdir(os.path.join(destTest, "labels"))

    os.mkdir(destVal)
    os.mkdir(os.path.join(destVal, "images"))
    os.mkdir(os.path.join(destVal, "labels"))


def replaceLabels(char, dataset):
    # Instanciate a dictionnary containing the values that need to be changed depending on the dataset being "translated"    
    if dataset == os.path.join(commonPath + "d1/"):
        # Values that change 2->3, 3->5, 4->11, 5->12, 6->13, 7->14, 8->15
        classes = {"2": "3", "3": "5", "4": "11", "5": "12", "6": "13", "7": "14", "8": "15"}
    elif dataset == os.path.join(commonPath + "d2/"):
        # Values that change 0->4, 1->6, 2->8, 3->9
        classes = {"0": "4", "1": "6", "2": "8", "3": "9"}
    elif dataset == os.path.join(commonPath + "d3/"):
        # Values that change 0->11, 1->3, 2->8, 4->10, 5->7
        classes = {"0": "11", "1": "3", "2": "8", "4": "10", "5": "7"}
    else:
        classes = {}
    
    if char in classes:
        return classes[char]
    else:
        return char



def unifyDataset(datasetPath):
    trainDir = os.path.join(datasetPath, "train/")
    testDir = os.path.join(datasetPath, "test/")
    valDir = os.path.join(datasetPath, "valid/")

    lstdir = [trainDir, testDir, valDir]

    for dirnum in range(3):
        for subdir in os.listdir(lstdir[dirnum]):
            subdirPath = os.path.join(lstdir[dirnum], subdir)
            destPath = os.path.join(destDirs[dirnum],subdir)
            
            if subdir == "images":
                # Just copy the images folder
                shutil.copytree(subdirPath, destPath, dirs_exist_ok=True)
            if subdir == "labels":
                # We need to edit the annotations in order to have a working dataset
                for file in glob.glob("*", root_dir=subdirPath):
                    # Copy the file content into a list to modify the content
                    shutil.copy(os.path.join(subdirPath,file), destPath)

                    with open(os.path.join(destPath, file), "r") as read:
                        tmp = read.readlines()
                    
                    for_write = []
                    for line in tmp:
                        label = line[0] # Only works because there are no labels above 9
                        modified_line = replaceLabels(label, datasetPath) + line[1:]
                        for_write.append(modified_line)

                    with open(os.path.join(destPath, file), "w") as write:
                        write.writelines(for_write)


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
        file.write("nc: 16\n")
        file.write("['Abcess', 'Badly Decayed', 'Broken Down Crow/Root', 'Caries', 'Cavity', 'Crown', 'Fillings', 'Fractured', 'Impacted Tooth', 'Implant', 'Infected', 'Normal', 'Overhang', Post', 'RCT', 'Restoration']\n")

createDest()

d1Path = os.path.join(commonPath + "d1/")
d2Path = os.path.join(commonPath + "d2/")
d3Path = os.path.join(commonPath + "d3/")

writeData()
unifyDataset(d1Path)
unifyDataset(d2Path)
unifyDataset(d3Path)
