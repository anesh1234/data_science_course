import pandas as pd
import os

def countNoClassOcc(dirs):
    classOcc = [0, 0, 0, 0, 0, 0, 0, 0]
    for directory in dirs:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                try:
                    df = pd.read_csv(os.path.join(directory, filename), header=None, sep=' ', names=range(100))
                except pd.errors.EmptyDataError:
                    print('No content, skipping')
                    continue

                for row in df.itertuples():
                    class_index = int(row[1])
                    if class_index in range(8):
                        classOcc[class_index] += 1
                    else:
                        print('\nERROR - CLASS OUT OF RANGE\n')
    return classOcc

# Check final directories
finalDir = ['datasets/final/train/labels', 'datasets/final/valid/labels']
finalDir_sep = [['datasets/final/train/labels'], ['datasets/final/valid/labels']]

# Check directory D1 with new mappings
d1 = ['datasets/processing/d1/labels']

# Check directory D2 with new mappings
d2 = ['datasets/processing/d2/labels']

# Check directory D3 with new mappings
d3 = ['datasets/processing/d3/labels']



# Get total number of class occurences in all relevant directories
listDirs = [finalDir, d1, d2, d3]
names = ['final', 'D1', 'D2', 'D3']

for index, list in enumerate(listDirs):
    countPerClass = countNoClassOcc(list)
    print(f'{names[index]}', '\n')
    for i in range(len(countPerClass)):
        print(f'Class {i} count: {countPerClass[i]}')
    print('\n')

# Get number of class occurences in the train and val directories of the "final" dataset
split_names = ['train', 'val']

for index, path in enumerate(finalDir_sep):
    countPerClass = countNoClassOcc(path)
    print(f'{split_names[index]}', '\n')
    for i in range(len(countPerClass)):
        print(f'Class {i} count: {countPerClass[i]}')
    print('\n')