
# !pip install nlpaug
# !pip install transformers
# !pip install textaugment

import os
import nltk
import random
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textaugment import EDA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from collections import Counter

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from google.colab import files
uploaded = files.upload()

# !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

"""# Text augmentation"""

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def plot_results(history_original, history_augmented, method):
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))

  ax[0].plot(history_original.history['loss'], label='Training Loss original')
  ax[0].plot(history_original.history['val_loss'], label='Validation Loss original')

  ax[0].plot(history_augmented.history['loss'], label=f'Training Loss augmened using {method}')
  ax[0].plot(history_augmented.history['val_loss'], label=f'Validation Loss augmened using {method}')

  ax[0].legend()
  ax[0].grid()
  ax[0].set_title('Loss')

  ax[1].plot(history_original.history['accuracy'], label='Training Accuracy original')
  ax[1].plot(history_original.history['val_accuracy'], label='Validation Accuracy original')
  ax[1].plot(history_augmented.history['accuracy'], label=f'Training Accuracy augmened using {method}')
  ax[1].plot(history_augmented.history['val_accuracy'], label=f'Validation Accuracy augmened using {method}')

  ax[1].legend()
  ax[1].grid()
  ax[1].set_title('Accuracy')

  plt.tight_layout()
  plt.show()


def valid_string(sentence):
    return isinstance(sentence, str) and len(sentence.strip()) > 0

def get_confusion_matrix_for_text(model, y_pred, y_test,title):
  cm = confusion_matrix(y_test, y_pred)

  plt.figure(figsize=(6, 5))

  class_names = np.unique(y_test)

  sns.set(font_scale=1)  # for label size
  sns.heatmap(cm, cmap= "Blues", linecolor = 'black', annot = True, fmt='', xticklabels=class_names, yticklabels=class_names)

  plt.xlabel("Predicted Labels", fontsize=16)
  plt.ylabel("True Labels", fontsize=16)

  plt.title(title)
  plt.show()

def custom_synonym_replacement(sentence, n):
    tokens = nltk.word_tokenize(sentence)
    num_replacements = min(n, len(tokens))

    for _ in range(num_replacements):
        words_perturbed = set()
        while len(words_perturbed) < num_replacements:
            if len(words_perturbed) == len(tokens):
                break
            word_idx = random.randint(0, len(tokens) - 1)
            token = tokens[word_idx]
            if token in words_perturbed or token not in eda.model.wv.vocab or token in eda.stop_words:
                continue
            synonyms = eda._synonyms(token)
            if len(synonyms) > 0:
                tokens[word_idx] = random.sample(synonyms, 1)[0]
            words_perturbed.add(token)

    return " ".join(tokens)

def augment_and_balance(X, y, factor=1, augmentation_func=None, n=1):
    augmented_X = list(X)
    augmented_y = list(y)

    if not augmentation_func:
      augmenter = EDA()
      augmentation_func = augmenter.synonym_replacement

    if factor != 1:
      for i in range(len(X)):
          for _ in range(factor):
              try:
                  new_text = augmentation_func(X[i], n)
                  augmented_X.append(new_text)
                  augmented_y.append(y[i])
              except:
                  continue

    classes = np.unique(augmented_y)
    class_counts = Counter(augmented_y)
    max_count = max(class_counts.values())

    for class_ in classes:
        class_count = class_counts[class_]
        class_samples = [x for x, label in zip(augmented_X, augmented_y) if label == class_]
        if class_count < max_count:
            sampling_indices = np.random.choice(list(range(class_count)), size=max_count-class_count)

            additional_samples = []
            for i in sampling_indices:
              try:
                  new_sample = augmentation_func(class_samples[i], n)
                  additional_samples.append(new_sample)
              except Exception as e:
                  print(f"Failed to augment sample {class_samples[i]} due to: {str(e)}")

            additional_labels = [class_] * len(additional_samples)

            augmented_X.extend(additional_samples)
            augmented_y.extend(additional_labels)

    augmented_X, augmented_y = shuffle(augmented_X, augmented_y)

    return augmented_X, augmented_y

def load_text_data():
  data_raw = pd.read_csv("/content/drive/MyDrive/cleaned_reviews.csv")
  stop_words = set(stopwords.words('english'))

  data = data_raw.dropna(subset=['cleaned_review'])
  data = data[data['cleaned_review'].apply(valid_string)]

  data['text'] = data['cleaned_review'].str.lower()
  data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

  return data_raw, data['text'], data['sentiments']

data_raw, X, y = load_text_data()

data_raw['sentiments'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def run_text_augmentation_test(X_train, y_train, X_test, y_test, factor, augm_func, n, method):
  X_train_augmented, y_train_augmented = augment_and_balance(X_train, y_train, factor, augm_func, n)

  X_train_augmented = vectorizer.transform(X_train_augmented)

  model = LogisticRegression(max_iter=1000)
  model.fit(X_train_augmented, y_train_augmented)

  y_pred = model.predict(X_test)

  print(f"Model trained on dataset augmented by {method}:")
  print(classification_report(y_test, y_pred, digits=5))

  get_confusion_matrix_for_text(model, y_pred, y_test, f'Confusion matrix for trained on dataset with {method}')

category_counts = pd.DataFrame(y)['sentiments'].value_counts()

# Plot the count as a pie chart
plt.pie(category_counts, labels=category_counts.index, autopct='%.1f%%')
plt.title('Category Counts')

# Show the plot
plt.show()

eda = EDA()

vectorizer = TfidfVectorizer()
X_train_original_1 = vectorizer.fit_transform(X_train)
X_test_original_1 = vectorizer.transform(X_test)

model_original = LogisticRegression(max_iter=1000)
model_original.fit(X_train_original_1, y_train)
y_pred_original = model_original.predict(X_test_original_1)

print("Model trained on original dataset:")
print(classification_report(y_test, y_pred_original, digits=5))

get_confusion_matrix_for_text(model_original, y_pred_original, y_test, 'Confusion matrix for trained on original dataset')

run_text_augmentation_test(X_train, y_train, X_test_original_1, y_test, 2, eda.synonym_replacement, 2, 'synonym replacement')

run_text_augmentation_test(X_train, y_train, X_test_original_1, y_test, 2, eda.random_swap, 3, 'random swap')

run_text_augmentation_test(X_train, y_train, X_test_original_1, y_test, 2, eda.random_insertion, 2, 'random insertion')

"""# Image augmentation"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import random

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_training_data('/content/drive/MyDrive/chest_xray/train')
test = get_training_data('/content/drive/MyDrive/chest_xray/test')
val = get_training_data('/content/drive/MyDrive/chest_xray/val')

l = []
for i in train:
    if(i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")

# Calculate counts for each category
counts = {i: l.count(i) for i in set(l)}

# Plot pie chart
sns.set_style('darkgrid')
plt.pie(counts.values(), labels=counts.keys(), autopct='%.1f%%')
plt.axis('equal')
plt.show()

plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

def display_augmented_images(datagen, x_train, y_train, num_pairs=5):
    fig = plt.figure(figsize=(3 * num_pairs, 6))

    for i in range(num_pairs):
        original_image, original_label = x_train[i], y_train[i]
        generator = datagen.flow(np.expand_dims(original_image, axis=0),
                                 np.expand_dims(original_label, axis=0),
                                 batch_size=1)
        augmented_image, augmented_label = generator.next()

        plt.subplot(2, num_pairs, i + 1)
        plt.imshow(np.squeeze(original_image), cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, num_pairs, i + num_pairs + 1)
        plt.imshow(np.squeeze(augmented_image), cmap='gray')
        plt.title("Augmented Image")
        plt.axis('off')

    plt.show()

def create_custom_datagen(x_train, **kwargs):
    datagen = ImageDataGenerator(**kwargs)

    datagen.fit(x_train)

    return datagen

def run_test(x_train, y_train, x_val, y_val,**kwargs):
    datagen = create_custom_datagen(x_train, **kwargs)

    display_augmented_images(datagen, x_train, y_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

    model = get_cnn_model()
    history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

    print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
    print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

    return model, history

def get_predictions_audio(model, x_test, y_test):
    predictions = model.predict_classes(x_test)
    predictions = predictions.reshape(1,-1)[0]

    print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

    cm = confusion_matrix(y_test,predictions)
    cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

    plt.figure(figsize = (10,10))
    sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)

    plt.show()

datagen = ImageDataGenerator()
datagen.fit(x_train)

display_augmented_images(datagen, x_train, y_train)

def get_cnn_model():
    model = Sequential()

    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1 , activation = 'sigmoid'))

    model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

    return model

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

model_original = get_cnn_model()

history_original = model_original.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

get_predictions_audio(model_original, x_test, y_test)

model_rotation, history_rotation = run_test(x_train, y_train, x_val, y_val,
                                            rotation_range = 30,
                                            horizontal_flip = True,
                                            vertical_flip=True)

get_predictions_audio(model_rotation, x_test, y_test)

model_zoom, history_zoom = run_test(x_train, y_train, x_val, y_val,
                                            zoom_range = 0.2,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1
                                           )

get_predictions_audio(model_zoom, x_test, y_test)

model_brightness, history_brightness = run_test(x_train, y_train, x_val, y_val,
                                            brightness_range=(0.5, 1.2),
                                            zoom_range = 0.2
                                            )

get_predictions_audio(model_brightness, x_test, y_test)

"""# Audio augmentation"""
import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

def extract_features(audio, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file")
        return None

    return mfccs_scaled

def change_speed(audio, sample_rate):
    speed_change = np.random.uniform(low=0.8, high=1.2)
    tmp = librosa.effects.time_stretch(audio, rate=speed_change)
    return tmp

def change_pitch(audio, sample_rate):
    pitch_change = np.random.uniform(low=-5, high=5)
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_change)

def random_shift(audio, sample_rate):
    shift = np.random.randint(low=-600, high=600)
    return np.roll(audio, shift)

def generate_augmented_data(dataset, labels, augmentation_func=None, num_augmentations=1, factor=1):
    augmented_data = []
    augmented_labels = []

    for _ in range(factor):
      for i, (features, audio, sample_rate) in enumerate(dataset):
          augmented_data.append(features)

          if augmentation_func:
              for _ in range(num_augmentations):
                  audio_changed = augmentation_func(audio, sample_rate)

                  features_changed = extract_features(audio_changed, sample_rate)
                  augmented_data.append(features_changed)
                  augmented_labels.append(labels[i])

          augmented_labels.append(labels[i])

    return np.array(augmented_data), np.array(augmented_labels)

def get_model(X_train):
  model = Sequential()
  model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
  model.add(Dropout(0.5))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(len(labels), activation='softmax'))

  model.compile(loss="sparse_categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

  return model

def plot_confusion_matrix_audio(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,cmap= "Blues", linecolor = 'black', annot = True, fmt='', xticklabels = classes, yticklabels = classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    if title:
        plt.title(title)
    plt.show()

def train_with_augmentation(X_train_augmented_cnn, y_train_augmented, X_test_cnn, y_test, augmentation_name, epochs=20, model=None):
    print(f"Training with {augmentation_name} augmentation.")

    if model is None:
      model = get_model(X_train_augmented_cnn)

    history = model.fit(X_train_augmented_cnn, y_train_augmented, epochs=epochs, batch_size=32,validation_data=(X_test_cnn, y_test))
    score = model.evaluate(X_test_cnn, y_test, batch_size=32, verbose=1)
    augmented_accuracy = score[1]

    print(f"Test accuracy ({augmentation_name} dataset):", augmented_accuracy)

    return model, history

def get_predictions(model, X_test, y_test, title=None):
  y_pred = model.predict(X_test)

  y_pred_classes = np.argmax(y_pred, axis=1)

  print(classification_report(y_test, y_pred_classes, digits=5, target_names=label_encoder.classes_))

  plot_confusion_matrix_audio(y_test, y_pred_classes, classes=label_encoder.classes_, title=title)

def plot_waveplots(X_train, augmented_fn, title):
  original_audio = X_train[random.randint(0, len(X_train))]
  augmented_audio = augmented_fn(original_audio[1], original_audio[2])

  plt.figure(figsize=(12, 4))

  plt.subplot(2, 1, 1)
  librosa.display.waveshow(original_audio[1], sr=sample_rate)
  plt.title('Original Audio', fontweight='bold', color='red')

  plt.subplot(2, 1, 2)
  librosa.display.waveshow(augmented_audio, sr=sample_rate)
  plt.title(f'Augmented Audio ({title})', fontweight='bold', color='red')

  plt.tight_layout()
  plt.show()

# Load dataset
data_path = '/content/Animals/'
subfolders = os.listdir(data_path)
labels = []
dataset = []

for subfolder in subfolders:
    for file_name in os.listdir(data_path + subfolder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_path, subfolder, file_name)
            audio, sample_rate = librosa.load(file_path)
            features = extract_features(audio, sample_rate)
            dataset.append((features, audio, sample_rate))
            labels.append(subfolder)

len(labels)

counts = {i: labels.count(i) for i in set(labels)}

sns.set_style('darkgrid')
plt.pie(counts.values(), labels=counts.keys(), autopct='%.1f%%')
plt.axis('equal')
plt.show()

label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    np.array(dataset, dtype=object), label_encoded,  test_size=0.2, random_state=42
)

X_train_features = np.array([x[0] for x in X_train])
X_test_features = np.array([x[0] for x in X_test])

X_train_cnn = X_train_features[:, :, np.newaxis]
X_test_cnn = X_test_features[:, :, np.newaxis]

model_cnn_1_orig = get_model(X_train_cnn)

history_audio_original = model_cnn_1_orig.fit(X_train_cnn, y_train, epochs=30, batch_size=32, validation_data=(X_test_cnn, y_test))

score = model_cnn_1_orig.evaluate(X_test_cnn, y_test, batch_size=32, verbose=1)

original_accuracy = score[1]
print("\n\nTest accuracy (original dataset):", original_accuracy)

get_predictions(model_cnn_1_orig, X_test_cnn, y_test, title='Confusion matrix for original dataset')

augmentations = [change_speed, change_pitch, random_shift]
augmentation_names = ['Speed Change', 'Pitch Change', 'Random Shift']

X_train_augmented_speed, y_train_augmented_speed = generate_augmented_data(X_train, y_train, augmentation_func=augmentations[0], num_augmentations=2, factor=1)
X_train_augmented_speed_cnn = X_train_augmented_speed[:, :, np.newaxis]

plot_waveplots(X_train, change_speed, 'Speed Change')

model_cnn_1_speed, history_cnn_1_speed = train_with_augmentation(X_train_augmented_speed_cnn, y_train_augmented_speed, X_test_cnn, y_test, augmentation_names[0], 30)

get_predictions(model_cnn_1_speed, X_test_cnn, y_test, title='Confusion matrix for changed speed dataset')

plot_results(history_audio_original, history_cnn_1_speed, 'Speed change')

X_train_augmented_pitch, y_train_augmented_pitch = generate_augmented_data(X_train, y_train, augmentation_func=augmentations[1], num_augmentations=2, factor=1)
X_train_augmented_pitch_cnn = X_train_augmented_pitch[:, :, np.newaxis]

plot_waveplots(X_train, change_pitch, 'Pitch change')

model_cnn_1_pitch, history_cnn_1_pitch = train_with_augmentation(X_train_augmented_pitch, y_train_augmented_pitch, X_test_cnn, y_test, augmentation_names[1], 30)

get_predictions(model_cnn_1_pitch, X_test_cnn, y_test, title='Confusion matrix for changed pitch dataset')

plot_results(history_audio_original, history_cnn_1_pitch, 'Pitch change')

X_train_augmented_shift, y_train_augmented_shift = generate_augmented_data(X_train, y_train, augmentation_func=augmentations[2], num_augmentations=2)
X_train_augmented_speed_cnn = X_train_augmented_shift[:, :, np.newaxis]

plot_waveplots(X_train, random_shift, 'Random shift')

model_cnn_1_shift, history_cnn_1_shift = train_with_augmentation(X_train_augmented_shift, y_train_augmented_shift, X_test_cnn, y_test, augmentation_names[2], 30)

get_predictions(model_cnn_1_shift, X_test_cnn, y_test, title='Confusion matrix for random shift dataset')

plot_results(history_audio_original, history_cnn_1_shift, 'Shift change')