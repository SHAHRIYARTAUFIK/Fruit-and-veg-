# 🥝🥕 Fruit & Vegetable Detector

> **Detect, localise, and name** fruits and vegetables from any uploaded photo — right in your browser.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![HuggingFace](https://img.shields.io/badge/HuggingFace-ViT--36--classes-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)
![License](https://img.shields.io/github/license/SHAHRIYARTAUFIK/Fruit-and-veg-)
  
--- 

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🟩 **Bounding Boxes** | YOLOv8 draws coloured boxes around each item |
| 🏷️ **Name Labels** | Every box shows the fruit/veg name + confidence |
| 🤖 **36-Class Classifier** | HuggingFace ViT covers 36 produce types |
| 🖼️ **Easy Upload** | Drag-and-drop or paste from clipboard |
| ⚡ **Fast** | First-time model download only; instant after that |

---

## 🚀 Setup & Run (VSCode)

### 1 — Clone the repo
```bash
git clone https://github.com/SHAHRIYARTAUFIK/Fruit-and-veg-.git
cd Fruit-and-veg-
```

### 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the app
```bash
python app.py
```

Your browser opens automatically at **http://localhost:7860** 🎉

> **First run:** YOLOv8 (~6 MB) and the ViT model (~330 MB) are downloaded automatically.  
> Subsequent runs start immediately from cache.

---

## 📋 Supported Classes (36 total)

**Fruits**
> Apple · Banana · Coconut · Grape · Kiwi · Lemon · Mango · Orange · Paprika · Pear · Pineapple · Pomegranate · Strawberry · Watermelon

**Vegetables**
> Beetroot · Bell Pepper · Broccoli · Cabbage · Capsicum · Carrot · Cauliflower · Chilli Pepper · Corn · Cucumber · Eggplant · Garlic · Ginger · Jalapeño · Lettuce · Onion · Peas · Potato · Raddish · Spinach · Sweetcorn · Sweetpotato · Tomato · Turnip

---

## 🗂️ Project Structure

```
Fruit-and-veg-/
├── app.py              ← Main Gradio application
├── requirements.txt    ← Python dependencies
├── .gitignore          ← Git ignore rules
└── README.md           ← This file
```

---

## 🔧 How It Works

```
Upload Photo
     │
     ├──▶ YOLOv8n (COCO trained)
     │         Detects apple, banana, orange, broccoli, carrot
     │         Draws coloured bounding boxes + confidence labels
     │
     └──▶ HuggingFace ViT Classifier
               Runs on full image AND each detected crop
               Returns top-6 predictions across all 36 classes

     ──▶ Results panel + annotated image displayed side-by-side
```

---

## 🧰 Models Used

| Model | Source | Size |
|-------|--------|------|
| **YOLOv8n** | [Ultralytics](https://ultralytics.com) | ~6 MB |
| **fruits-and-vegetables-detector-36** | [HuggingFace](https://huggingface.co/jazzmacedo/fruits-and-vegetables-detector-36) | ~330 MB |

---

## 💡 Tips for Best Results

- ☀️ Use **well-lit, clear** photos
- 🖼️ Produce should **fill most of the frame**
- 🔍 **Close-up shots** of individual items work great
- 📷 Avoid heavy shadows or blurry images
- 🛒 Works well with **1–6 items** per image

---

## 📄 License

MIT © [SHAHRIYARTAUFIK](https://github.com/SHAHRIYARTAUFIK)
