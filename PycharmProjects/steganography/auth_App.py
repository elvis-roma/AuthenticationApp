import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

class DataPreparation:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.df = None
        self.scaler = MinMaxScaler()
        self.X = None
        self.y = None

    def generate_data(self):
        np.random.seed(42)
        login_times = np.random.randint(0, 24, size=(self.num_samples,))
        session_durations = np.random.randint(1, 100, size=(self.num_samples,))
        data = {
            'login_time': login_times,
            'session_duration': session_durations,
            'authenticated': np.random.randint(0, 2, size=(self.num_samples,))
        }
        self.df = pd.DataFrame(data)

    def preprocess_data(self):
        features = self.df[['login_time', 'session_duration']]
        self.X = self.scaler.fit_transform(features)
        self.y = self.df['authenticated']

class DeepLearningModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y, epochs=20, batch_size=32):
        X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        history = self.model.fit(X_reshaped, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return history

class Steganography:
    @staticmethod
    def lsb_embed(image_path, secret_data, output_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        secret_data_bin = ''.join(format(byte, '08b') for byte in secret_data.encode('utf-8'))

        idx = 0
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                if idx < len(secret_data_bin):
                    img_array[i, j, 0] = (img_array[i, j, 0] & ~1) | int(secret_data_bin[idx])
                    idx += 1

        img_result = Image.fromarray(img_array)
        img_result.save(output_path)

    @staticmethod
    def lsb_extract(image_path, data_length):
        img = Image.open(image_path)
        img_array = np.array(img)
        secret_data_bin = ""

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                if len(secret_data_bin) < data_length * 8:
                    secret_data_bin += str(img_array[i, j, 0] & 1)

        secret_data = ''.join(chr(int(secret_data_bin[i:i+8], 2)) for i in range(0, len(secret_data_bin), 8))
        return secret_data

class AuthApp:
    def __init__(self, master, dl_model, data_prep):
        self.master = master
        self.dl_model = dl_model
        self.data_prep = data_prep
        master.title("Authentication and Steganography")

        self.label = tk.Label(master, text="User Authentication System")
        self.label.pack()

        self.authenticate_button = tk.Button(master, text="Authenticate User", command=self.authenticate_user)
        self.authenticate_button.pack()

        self.stego_button = tk.Button(master, text="Embed Secret Data", command=self.embed_data)
        self.stego_button.pack()

        self.extract_button = tk.Button(master, text="Extract Secret Data", command=self.extract_data)
        self.extract_button.pack()

    def authenticate_user(self):
        # Simulate user authentication
        login_time = np.random.randint(0, 24)
        session_duration = np.random.randint(1, 100)
        data = np.array([[login_time, session_duration]])
        data_reshaped = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        prediction = self.dl_model.model.predict(data_reshaped)
        if prediction > 0.5:
            messagebox.showinfo("Authentication", "User authenticated successfully.")
        else:
            messagebox.showwarning("Authentication", "Authentication failed.")

    def embed_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            secret_data = "auth_token"
            output_path = "stego_image.png"
            Steganography.lsb_embed(file_path, secret_data, output_path)
            messagebox.showinfo("Steganography", "Data embedded successfully.")

    def extract_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            extracted_data = Steganography.lsb_extract(file_path, len("auth_token"))
            messagebox.showinfo("Steganography", f"Extracted Data: {extracted_data}")

if __name__ == "__main__":
    data_prep = DataPreparation()
    data_prep.generate_data()
    data_prep.preprocess_data()
    X, y = data_prep.X, data_prep.y

    input_shape = (1, X.shape[1])
    dl_model = DeepLearningModel(input_shape)
    history = dl_model.train_model(X, y)

    root = tk.Tk()
    app = AuthApp(root, dl_model, data_prep)
    root.mainloop()
