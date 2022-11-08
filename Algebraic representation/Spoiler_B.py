"""
Spoiler bij "Conv Dense opdracht"
Als het niet lukt om zelf een geschikte configuratie te vinden, mag je onderstaande
gebruiken voor opdracht B. Je kunt dan eventueel ook trainingstijd besparen door
reeds getrainde weights te downloaden: de file Spoiler_B_Weights.h5
Noem dan nog steeds welke configuraties je geprobeerd hebt,
en waarom deze volgens jou beter werkt.
"""

from tensorflow import keras
from tensorflow.keras import layers

# The model below mostly reaches a test acc of less than 0.97 in about 120 to 150 epochs.
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