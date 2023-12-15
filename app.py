import tkinter as tk
import tensorflow as tf
import numpy as np
import cv2
import csv
from PIL import Image, ImageDraw

model = tf.keras.models.load_model('digits_recognition.model')

data_path = 'feedback_data.csv'
with open(data_path, 'w', newline = '') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image', 'Digit'])  

class DigitRecognition:
    def __init__ (self, root):
        self.root = root
        self.root.title("Digits recognition")

        self.canvas = tk.Canvas(root, width = 800, height = 800, bg = "white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.label_result = tk.Label(root, text = "Prediction")
        self.label_result.pack()

        self.entry_correct_digit = tk.Entry(root, width = 5) 
        self.entry_correct_digit.pack()

        self.label_feedback = tk.Label(root, text = "")
        self.label_feedback.pack()

        self.check_button = tk.Button(root, text = "Enter", command = self.correct_checker)
        self.check_button.pack()

        self.clear_button = tk.Button(root, text = "Clear", command = self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = tk.Button(root, text = "Predict", command = self.predict)
        self.predict_button.pack()

        self.image = Image.new("RGB", (800, 800), color = "white")
        self.draw = ImageDraw.Draw(self.image)

        self.predicted_last_digit = None
        self.new_image = None

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill = "black", width = 35)
        self.draw.line([x1, y1, x2, y2], fill = "black", width = 35)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (800, 800), color = "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text = "Prediction")
        self.label_feedback.config(text = "")
        self.entry_correct_digit.delete(0, 'end')

    def predict(self):
        resized_image = self.image.resize((28, 28), Image.HAMMING)
        path = "image.png"
        resized_image.save(path)

        img = cv2.imread(path)[:, :, 0]
        img = np.invert(np.array([img]))
        img = img / np.linalg.norm(img)

        img[0] *= 30
        img[0][img[0] > 1] = 1

        self.new_image = np.array(img)

        prediction = model.predict(img)

        predicted_digit = np.argmax(prediction)
        self.label_result.config(text = f"It's probably a {predicted_digit} \n If this prediction is wrong, please enter a correct digit")

        self.predicted_last_digit = predicted_digit
        

    def correct_checker (self):
        correct_digit_text = self.entry_correct_digit.get()

        if correct_digit_text.isdigit():
            correct_digit = int(correct_digit_text)
            if correct_digit != self.predicted_last_digit:
                self.label_feedback.config(text = "Thanks for feedback") 
                with open(data_path, 'a', newline = '') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(((self.new_image).tolist(), correct_digit))  
                

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognition(root)
    root.mainloop()
