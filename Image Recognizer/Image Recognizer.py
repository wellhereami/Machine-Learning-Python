import os
import pandas
import numpy
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

datapath = "animals"

data = {"img":[], "labels":[]}

animalList = os.listdir(datapath)
for folder in animalList:
    folderpath = os.path.join(datapath, folder)
    photoList = os.listdir(folderpath)
    for photo in photoList:
        photopath = os.path.join(folderpath, photo)
        img = imread(photopath, as_gray=True)
        img = resize(img, (80,80))
        data["img"].append(img)
        data["labels"].append(folder)

animalDataframe = pandas.DataFrame(data)

labelEncoder = LabelEncoder()
animalDataframe['encodedLabels'] = labelEncoder.fit_transform(animalDataframe['labels'])

X = data["img"]
y = animalDataframe["encodedLabels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
orientations = 9
pixels = (8,8)
cells = (2,2)

XTrainFeatures = []
for trainingImage in range(0, len(X_train)):
    hogFeatures = hog(X_train[trainingImage], orientations=orientations, pixels_per_cell=pixels, cells_per_block=cells, block_norm='L2-Hys')
    XTrainFeatures.append(hogFeatures)

standardScaler = StandardScaler()
XTrainScaled = standardScaler.fit_transform(XTrainFeatures)

sgd = SGDClassifier(random_state=1, max_iter=1000, tol=1e-3)
sgd.fit(XTrainScaled, y_train)

XTestFeatures = []
for trainingImage in range(0, len(X_test)):
    hogFeatures = hog(X_test[trainingImage], orientations=orientations, pixels_per_cell=pixels, cells_per_block=cells, block_norm='L2-Hys')
    XTestFeatures.append(hogFeatures)

XTestScaled = standardScaler.fit_transform(XTestFeatures)

prediction = sgd.predict(XTestScaled)

print(numpy.array(prediction == y_test)[:25])
print('')
print("Percentage correct: ", 100*numpy.sum(prediction == y_test)/len(y_test))
stop = " "
while stop.lower() != "yes":
    photopath = input("Enter your photo path: ")
    testImg = imread(photopath, as_gray=True)
    testImg = resize(testImg, (80,80))

    testHog = hog(testImg, orientations=orientations, pixels_per_cell=pixels, cells_per_block=cells, block_norm='L2-Hys')
    testArr = numpy.array(testHog)
    testArr = testArr.reshape(1, -1)

    prediction = sgd.predict(testArr)
    print(labelEncoder.inverse_transform(prediction))
    
    stop=input("Would you like to stop testing? ")