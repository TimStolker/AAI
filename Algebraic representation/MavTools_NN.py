'''
MavTools_NN is a collection of useful tools for neural networks by Marius Versteegen
'''
import numpy as np
from tensorflow import keras
from matplotlib import pyplot
import matplotlib

# class MavTools
# description: 
#   ViewTools_NN is a class that offers some utility methods to inspect neural networks
class ViewTools_NN():
    
    # function printFeatureMapsForLayer
    # Description: 
    #   printFeatureMapsForLayer prints the outputs, the featuremaps of a layer.
    #   If x_test_flat (one hot vectors) is not used for training the network, 
    #   just fill in None for x_test_flat. The images in x_test will be used instead.
    #   If you use x_test_flat, still provide x_test, to allow the corresponding input image in 2D (non flattened).
    #   (which is easier for us humans to interpret)
    # Parameters:
    #   nLayer 
    #       this is the layer for which you'd like to print/view the output/featuremap.
    #   x_test_flat 
    #       this is the numpy array with flattened input images. 
    #       Only use this if you flattened the input images. If not, fill in None.
    #   x_test
    #       this is the numpy array with input images. It is assumed/required that each image has the 
    #       next shape: (width,height,nofBytesPerPixel)
    #       ... so far, I've only tested this function for cases where nofBytesPerPixel==1
    #   sampleIndex
    #       this is the index of the input image for which you'd like to inspect the
    #       layer outputs (images) of the specified layer.
    #   baseFilenameForSave
    #       this is the base filename (without extension) that you'd like to save to (as png).
    #       for example, if you specify "c:\featuremapOutput_", then outputfiles like featuremapOutput_1.png 
    #       are saved to your c root. if you leave out the path, the outputfiles will be saved to 
    #       the local directory.
    #       If you specify None, the layer images will not be saved.
    # Note:
    #   The subscribt _test is used to show that we're talking about a sampleset.
    #   Of course, the train sets can be used equally well.
    @staticmethod
    def printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, sampleIndex, baseFilenameForSave):
        
        sample = x_test_flat[sampleIndex] if(type(x_test_flat)==np.ndarray) else x_test[sampleIndex]
        samples = np.array([sample])    # testset with only one sample
        
        bSave = (baseFilenameForSave != None) and (baseFilenameForSave != '')
        
        # redefine model to output right after the selected layer
        model2 = keras.Model(inputs=model.inputs, outputs=model.layers[nLayer].output)
        #print("------------")
        #pyplot.figure()
        #pyplot.colorbar()
        pyplot.grid(False)
        image = np.squeeze(x_test[sampleIndex]) #Undo the expand dimension earlier to be able to pyplot it.
        pyplot.imshow(image, cmap='gray')
        pyplot.show()
    
        #model2.summary()
        
        feature_maps = model2.predict(samples)
        feature_map = feature_maps[0] # there's only one sample in samples, so there's only one set of features
        
        #print("featuremap shape:")
        #print(feature_map.shape)
        
        if (len(feature_map.shape)==1):
            # the output is a list of numbers, not a list of images
            # we need images to plot below, so let's convert it
            # to a list of 1x1 images:
            feature_map = np.expand_dims(feature_map, -2)
            feature_map = np.expand_dims(feature_map, -2)
        
        nofMaps = feature_map.shape[2]
        square = int(np.ceil(np.sqrt(nofMaps)))
        nofRows = square
        nofCols = square
        ix = 1
        for _ in range(nofRows):
            for _ in range(nofCols):
                if (ix > nofMaps):
                    break
                
                # specify axis (position) of subplot and turn off its ticks
                # the position is determined by the index ix in a grid of
                # nofRows * nofCols entries.
                ax = pyplot.subplot(nofRows, nofCols, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # plot filter channel in grayscale
                pyplot.imshow(feature_map[:, :, ix-1], cmap='gray', norm=None, filternorm=False, vmin=0, vmax=1)
                
                if (bSave):
                    filenameForSave = baseFilenameForSave + str(ix) +'.png'
                    matplotlib.image.imsave(filenameForSave,feature_map[:, :, ix-1])
                    
                #print(feature_map[:, :, ix-1])
                ix += 1
                
        # show the figure that has been specified above
        pyplot.show()
    
    @staticmethod    
    def getColoredText(r,g,b,text):
        return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)