import ROOT

from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, MaxPooling2D, Flatten

# Setup TMVA
ROOT.TMVA.Tools.Instance()
ROOT.TMVA.PyMethodBase.PyInitialize()

output = ROOT.TFile.Open("TMVA.root", "RECREATE")
factory = ROOT.TMVA.Factory(
    "MNIST", output,
    "!V:!Silent:Color:DrawProgressBar:AnalysisType=multiclass")

# Load data
dataloader = ROOT.TMVA.DataLoader("dataset")

data = ROOT.TFile.Open("mnist.root")
tree_digit = []
for i in range(10):
    tree_digit.append(data.Get("train_digits/train_digit{}".format(i)))
    dataloader.AddTree(tree_digit[i], "digit{}".format(i))

for i in range(28 * 28 * 1):
    dataloader.AddVariable("x[{}]".format(i), "x_{}".format(i), "")

dataloader.PrepareTrainingAndTestTree(
    ROOT.TCut(""), "SplitMode=Random:NormMode=None:!V")

# Define model
model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(28 * 28 * 1, )))
model.add(Conv2D(4, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Store model to file
model.save("model.h5")
model.summary()

# Book methods
factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras",
                   "H:!V:FilenameModel=model.h5:NumEpochs=10:BatchSize=100")

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
