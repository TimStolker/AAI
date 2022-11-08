"""
MNIST opdracht B: "Conv Dense"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je alleen nog als laatste layer een dense layer.

De opdracht bestaat uit drie delen: B1 tm B3.

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
Zorg ervoor dat je de onderstaande code volledig begrijpt.

Tip: stap in de Debugger door de code, en bestudeer de tussenresultaten.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from MavTools_NN import ViewTools_NN

#tf.random.set_seed(0) #for reproducability

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#print(x_test[0])
print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.show()
plt.grid(False)

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# This time, we don't flatten the images, because that would destroy the 
# locality benefit of the convolution layers.

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Opdracht B1: 
    
Voeg als hidden layers ALLEEN Convolution en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,97.

Voorbeelden van layers:
    layers.Dropout(getal)
    layers.Conv2D(getal, kernel_size=(getal, getal), padding="valid" of "same")
    layers.Flatten()
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
je ervaringen daarbij en probeer die ervaringen te verklaren.

Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_B.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.

Door een filter van 3x3 te nemen kunnen de kleinere datails meegenomen worden. 
denk bijvoorbeeld aan het getal 6 en 8, deze lijken op elkaar waardoor het detail van de getallen belangrijk is.
Door 20 filters te nemen kunnen er genoeg features uitgepikt worden.
Vervolgens wordt de grootte van de image gehalveerd, dit betekend dat de maximale waarde in 4 pixels wordt opgeslagen in 1 pixel.
Hierdoor zijn de features duidelijker.
Door dit twee keer te doen wordt de image 7x7 waardoor alle belangrijke features verdikt zijn.
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(.1),
            layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
    return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht B2:
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape

Als eerst wordt er een image van 28x28 in het model gestopt
In de eerste conv2d worden er 20 plaatjes gemaakt van ieder 28x28 omdat de padding "same" is
Vervolgens wordt door MaxPooling2D de plaatjes gehalveerd (lengte en breedte /2) = 28/2 en 28/2 = 14x14
Daarna worden de plaatsjes weer door een conv2d gegooid die als output ook 20 plaatjes heeft, dus het blijft 14x14 voor 20 plaatjes
Daarna weer door de conv2d waardoor de plaatjes weer gehalveerd worden 14/2 en 14/2 = 7x7 (ook hier blijft padding "same")

Daarna wordt het geflattend, dus 7*7 pixels * 20 plaatjes = 7*7*20 = 980. Dit is een array van 980 elementen.
In de dense layer zijn er vervolgens 10 neuronen die de output van het model maken. Dit is een array van 10 elementen.
"""

"""
Opdracht B3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.
Bij conv2d_1: 3*3 pixels + 1 bias * 20 images = (3*3+1)*20 = 200
bij conv2d_2: 3*3 pixels * 20 images + 1 bias * 20 images = (3*3*20+1)*20 = 3620
bij dense: 980 pixels + 1 bias * 10 neuronen = (980+1)*10 = 9810
"""

"""
# Train the model
"""
batch_size = 2048 # Larger often means faster training, but requires more system memory.
                  # if you get allocation accord
epochs = 5    # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
#loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'     # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(lr=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = True # Set this to false if you've changed your model.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if (bInitialiseWeightsFromFile):
    model.load_weights("Spoiler_b_weights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_B_weights.h5" here.

print ("\n");
print (ViewTools_NN.getColoredText(255,255,0,"Just type CTRL+C anytime if you feel that you've waited for enough episodes."))
print ("\n");
# # NB: validation split van 0.2 ipv 0.1 gaf ook een boost: van 99.49 naar 99.66
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])    
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

model.summary()
model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0])

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer                 # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat = None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
