from tensorflow import keras

# load model dari .h5
model = keras.models.load_model("model999.h5")

# save ulang dengan format .keras
model.save("model999.keras", save_format="keras")
