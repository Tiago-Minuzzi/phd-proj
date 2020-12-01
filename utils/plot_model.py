import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Files
architecture = 'model_nadd_2416241525_4_architecture.png'
model_history = "model_nadd_2416241525_4_saved_history.pkl"
model = "model_nadd_2416241525_4.hdf5"
model_loss = "model_nadd_2416241525_4_val_loss.png"
model_accuracy = "model_nadd_2416241525_4_val_accuracy.png"
cm = "model_nadd_2416241525_4_confusion_matrix.png"
# Functions
def perc_val(data):
    dtl = []
    for dt in data:
        dtl.append(dt*100)
    return(dtl)

def load_history(history):
    with open(history,"rb") as hist:
        loaded = pickle.load(hist)
    return loaded

# Load model and model history
model = load_model(model)
model_history = load_history(model_history)
# Plot model architecture
tf.keras.utils.plot_model(model,
                          show_shapes = True,
                          show_layer_names = True,
                          to_file = architecture)
# Get values from history
lv_list = model_history['loss']
vlv_list = model_history['val_loss']
av_list = perc_val(model_history['acc'])
vav_list = perc_val(model_history['val_acc'])
# Model loss
plt.figure(figsize=(9,6), dpi=200)
epoch_count = range(1, len(model_history['val_loss']) + 1)
plt.plot(epoch_count, lv_list)
plt.plot(epoch_count, vlv_list)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.savefig(model_loss)
# Model accuracy
plt.figure(figsize=(9,6), dpi=200)
epoch_count = range(1, len(model_history['val_acc']) + 1)
plt.plot(epoch_count, av_list)
plt.plot(epoch_count, vav_list)
plt.title('Model accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig(model_accuracy)

# pred = mdl_ld.predict(x_test_3d)
# pred = np.argmax(pred, axis = 1)
# lab = np.argmax(ynn_test,axis = 1)
# # Classification report
# print(classification_report(lab,pred))
# # Confusion matrix
# categorias = ['NT','TE']
# sns.heatmap(confusion_matrix(lab,pred),
#             annot=True,
#             fmt='d',
#             cmap="Blues",
#             cbar=False,
#             xticklabels=categorias,
#             yticklabels=categorias)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.figure(figsize=(8,8))
# plt.title("Confusion matrix")
# plt.saveimg(cm)