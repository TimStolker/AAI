"""
Spoiler bij "Only Dense opdracht"
Als het niet lukt om zelf een geschikte configuratie te vinden, mag je onderstaande
gebruiken voor opdracht A. Je kunt dan eventueel ook trainingstijd besparen door
reeds getrainde weights te downloaden: de file Spoiler_A_Weights.h5
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
            layers.Dropout(.1),
            layers.Dense(units=100, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=20, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
    return model
