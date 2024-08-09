import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout

# Task 1
def light_pixels(image, lightness, channel):
    # Determine the channel index based on the channel string
    if channel=='red':
        channel_index=0
    elif channel=='green':
        channel_index=1
    elif channel=='blue':
        channel_index=2
    # Determine height and width
    rows, cols= image.shape[:2]
    
    # Create an empty list to store the mask
    mask = []
    
    # Iterate over each row in the image
    for i in range(rows):
        row = []  # Temporary list to store each row's mask
        for j in range(cols):
            # Check if the pixel value in the specified channel is above the lightness threshold
            if image[i][j][channel_index] > lightness:
                row.append(1)
            else:
                row.append(0)
        mask.append(row)  # Add the row mask to the final mask
    
    # Convert the list of lists to a numpy array before returning
    return np.array(mask)

    
# Task 2
def histogram(image, buckets, channel):
    #defining channel index value 
    if channel=='red':
        channel_index = 0
    elif channel=='green':
        channel_index = 1
    elif channel=='blue':
        channel_index = 2
    
    # Extracting the specific channel from the image
    channel_data = image[:, :, channel_index]
    #Flattens the 2D array of the channel data into a 1D list of pixel values
    pixels = [item for sub_channel in channel_data for item in sub_channel]
    #print('pixels:', pixels)
    #calculating bucket size
    bucket_size = 256 // buckets
    #print(bucket_size)
    # Initializing the histogram array
    hist = [0] * buckets
    for x in pixels:
        # Determine which bucket the pixel value falls into
        bucket_index = min(x // bucket_size, buckets - 1) # Ensure index is within range
        #print(bucket_index)
        hist[bucket_index] += 1
    
    return hist
     
# Task 3
def build_deep_nn(rows, columns, channels, layer_options):
    # Initializing the model
    model = Sequential()
    
    # Adding the Flatten layer
    model.add(Flatten(input_shape=(rows, columns, channels)))
    
    # Iterating over the layer options and add layers accordingly
    for i in range(len(layer_options)):
        hidden_size = layer_options[i][0]
        activation = layer_options[i][1]
        dropout_rate = layer_options[i][2]
        # Adding a Dense layer with the given size and activation function
        model.add(Dense(hidden_size, activation=activation))
        
        # Adding a Dropout layer if layer_options[i][2] > 0
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    return model

if __name__ == "__main__":
     import doctest
     doctest.testmod()
