from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import keras
import numpy as np
from PIL import Image


app = Flask(__name__)

loaded_model = keras.models.load_model('mobilenet1.h5')
datagen= keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_ds=datagen.flow_from_directory(
    'historical2/',
    target_size=(224,224),
    batch_size=32,
    subset="training")
test_ds=datagen.flow_from_directory(
    'historical2/',
    target_size=(224,224),
   batch_size=32,
   subset="validation")
import pandas as pd

pd.set_option("max_colwidth", None)

# Create the pandas DataFrame

df = pd.read_csv("first.csv")
df1= pd.read_csv("second.csv")

# print dataframe.
print(df.head())



@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./Images/" + imagefile.filename
    imagefile.save(image_path)


    image = keras.utils.load_img(image_path,target_size=(224,224,3))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    result1 = loaded_model.predict(image)
    result1 = np.array(result1)
    result1 = result1.ravel()

    # Initialize max_val and max_index to the first element of the array
    max_val1 = result1[0]
    max_index1 = 0

    # Iterate through the array using a for loop
    for index, element in enumerate(result1):
        if element > max_val1:
            max_val1 = element
            max_index1 = index
    z = train_ds.class_indices

    # Value to find
    value_to_find1 = max_index1

    # Initialize an empty list to store keys with the desired value
    keys_with_value1 = ""

    # Iterate through the dictionary items
    for key, value in z.items():
        if value == value_to_find1:
            keys_with_value1 = key

    Bangla = df[df["Name"] == keys_with_value1].Bangla_Caption
    # convert Series to string without index
    Bangla = Bangla.to_string(index=False)
    English= df[df["Name"] == keys_with_value1].English_Caption
    English = English.to_string(index=False)
    ne = df1[df1["Name"] == keys_with_value1].Nearest_tourist_Place

    ne = ne.to_string(index=False)

    return render_template('index.html',image=image_path,prediction=keys_with_value1,dataframeb=Bangla,dataframee=English,near=ne)





if __name__ == '__main__':
    app.run(port=3000, debug=True)