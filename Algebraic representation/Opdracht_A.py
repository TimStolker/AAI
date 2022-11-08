"""
MNIST opdracht A: "Only Dense"      (by Marius Versteegen, 2021)
Bij deze opdracht ga je handgeschreven cijfers klassificeren met 
een "dense" network.

De opdracht bestaat uit drie delen: A1, A2 en A3 (zie verderop)

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
    
--> Zorg ervoor dat je de onderstaande code volledig begrijpt. <--
******************************************************************

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

print(x_test[0])
print("Showing first test-image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples\n")

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# we can flatten the images, because we use Dense layers only
x_train_flat = x_train.reshape(x_train.shape[0], inputShape[0]*inputShape[1])
x_test_flat  = x_test.reshape(x_test.shape[0], inputShape[0]*inputShape[1])

"""
Opdracht A1: 
    
Voeg ALLEEN Dense en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,95.

Voorbeelden van layers:
    layers.Dropout( getal ),
    layers.Dense(units=getal, activation='sigmoid'),
    layers.Dense(units=getal, activation='relu'),

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

[ervaring: Zonder GPU, op I5 processor < 5 min, 250 epochs bij batchsize 4096]

Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_A.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.

De reden dat er een dropout is gekozen van 0.1 is zodat er geen overfitting plaatsvind. Als er geen dropout wordt gedaan
kan het zo zijn dat je train data 100% accuraat is, maar de test data een slecht resultaat krijgt, dit heeft te maken
met de schijdingslijnen die te precies voor de training data gekozen is.

De keus van eerste dense layer van 100 is om de een begin te hebben om naar de uiteindelijke output van 10 te gaan.
Door een grootte van 100 te kiezen zijn er niet gigantisch veel parameters om te onthouden, dit draagt bij aan de performance
Door een 2de layer van 20 neuronen te hebben zit je al een stuk dichterbij de output, vervolgens heeft de output layer
10 neuronen, omdat dit het aantal verschillennde classes zijn. (getal 0 t/m 9)
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Dropout(.1),
            layers.Dense(units=100, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=20, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
    return model

model = buildMyModel(x_train_flat[0].shape)
model.summary()

"""
Opdracht A2: 

Verklaar met kleine berekeningen het aantal parameters dat bij elke laag
staat genoemd in bovenstaande model summary.

Dropouts hebben geen parameters, dus daar hoeven geen berekeningen voor gedaan te worden.

Dense(100) heeft 784 inputs + 1 bias over 100 neuronen, dus 784+1*100 = 78500
Dense(20) heeft 100 inputs + 1 bias over 20 neuronen, dus 100+1*20 = 2020
Dense(10) heeft 20 inputs + 1 bias over 10 neuronen, dus 20+1*10 = 210

"""

"""
# Train the model
"""
batch_size = 10   # Larger can mean faster training (especially when using the gpu: 4096), 
                  # but requires more system memory. Select it properly for your system.
                    
epochs = 5     # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
#loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'        # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(lr=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = True # Set this to false if you've changed your model, or if you want to restart using different (random) weights.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if (bInitialiseWeightsFromFile):
    model.load_weights("Spoiler_A_weights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_A_weights.h5" here.
print (ViewTools_NN.getColoredText(255,255,0,"\nJust type CTRL+C anytime if you feel that you've waited for enough episodes.\n"))

try:
    model.fit(x_train_flat, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard\n")

"""
# Evaluate the trained model
"""
score = model.evaluate(x_test_flat, y_test, verbose=0)
print("Test loss:", score[0])       # 0.0394
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]), "\n")   # 0.9903

model.summary()
model.save_weights('myWeights.h5')

prediction = model.predict(x_test_flat)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0],"\n")

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer = nLastLayer                 # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)

"""
Opdracht A3: 

1 Leg uit wat de betekenis is van de output images die bovenstaand genenereerd worden met "printFeatureMapsForLayer".

Voor iedere print krijg je 2 images te zien; die van de input en bij welke classe hij wordt gepredict.
De eerste is bijvoorbeeld een 7, dit wordt dan geplot als de pixels die gekleurd zijn. De tweede image die je te zien krijgt
zijn 10 vakjes waarbij de prediction wit is, beginnend bij 0.

2 Leg uit wat de betekenis zou zijn van wat je ziet als je aan die functie de index van de eerste dense layer meegeeft.

Dan zie je 100 verschillende features van het input getal.

3 Leg uit wat de betekenis zou zijn van wat je ziet als je aan die functie de index van de tweede dense layer meegeeft.

Dan zie je 20 verschillende features die een combinatie zijn van de vorige 100 features.
"""
