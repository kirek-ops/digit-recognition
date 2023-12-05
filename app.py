import tkinter as tk
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw

model = tf.keras.models.load_model('digits_recognition.model')

class DigitRecognition:
    def __init__ (self, root):
        self.root = root
        self.root.title("Digits recognition")

        self.canvas = tk.Canvas(root, width = 800, height = 800, bg = "white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.label_result = tk.Label(root, text = "Prediction")
        self.label_result.pack()

        self.clear_button = tk.Button(root, text = "Clear", command = self.clear_canvas)
        self.clear_button.pack()

        self.save_button = tk.Button(root, text = "Predict", command = self.predict)
        self.save_button.pack()

        self.image = Image.new("RGB", (800, 800), color = "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint (self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill = "black", width = 35)
        self.draw.line([x1, y1, x2, y2], fill = "black", width = 35)

    def clear_canvas (self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (800, 800), color = "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text = "Prediction")

    def predict (self):
        resized_image = self.image.resize((28, 28), Image.HAMMING)
        path = "image.png"
        resized_image.save(path)

        img = cv2.imread(path)[:,:,0]
        img = np.invert(np.array([img]))
        img = img / np.linalg.norm(img)

        img[0] *= 10
        img[0][img[0] > 1] = 1

        prediction = model.predict(img)

        self.label_result.config(text = f"It's probably a {np.argmax(prediction)}")

        
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognition(root)
    root.mainloop()
