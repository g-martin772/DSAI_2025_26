import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

canvas_size = 850
img_size = 28

class DrawWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a digit")
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_predict = tk.Button(self.root, text="Save & Predict", command=self.save)
        self.button_predict.pack(side=tk.LEFT)
        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear)
        self.button_clear.pack(side=tk.LEFT)
        self.image = Image.new("L", (canvas_size, canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.model = load_model("mnist_model.keras")

    def paint(self, event):
        draw_radius = 16
        x1, y1 = (event.x - draw_radius), (event.y - draw_radius)
        x2, y2 = (event.x + draw_radius), (event.y + draw_radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def save(self):
        img_small = self.image.resize((img_size, img_size), Image.LANCZOS)
        img_arr = np.array(img_small).astype('float32') / 255.0
        img_arr = 1.0 - img_arr
        img_arr = img_arr.flatten().reshape(1, img_size * img_size)
        pred = self.model.predict(img_arr)
        digit = np.argmax(pred)
        tk.messagebox.showinfo("Prediction", f"Predicted digit: {digit}")
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (canvas_size, canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    import tkinter.messagebox
    dw = DrawWindow()
    dw.run()