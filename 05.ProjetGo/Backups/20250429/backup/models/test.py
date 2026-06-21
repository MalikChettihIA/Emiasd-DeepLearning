from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

if __name__ == "__main__":

    # Detheve_Vrel_24_03_2025.h5
    #model = load_model("Detheve_Vrel_24_03_2025.h5", compile=True)
    model = load_model("../JMDessalas_12avr.h5", compile=True)
    model.summary()
    plot_model(model, to_file="model_graph2.png", show_shapes=True, show_layer_names=True)