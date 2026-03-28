import warnings
warnings.filterwarnings("ignore")
  

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

FOOD_COCO_LABELS = {
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "bowl",
}
 
BOX_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#FF8C42", "#6BCB77",
]

EMOJI_MAP = {
    "apple": "🍎", "banana": "🍌", "orange": "🍊", "grape": "🍇",
    "strawberry": "🍓", "watermelon": "🍉", "mango": "🥭", "pineapple": "🍍",
    "pear": "🍐", "peach": "🍑", "cherry": "🍒", "lemon": "🍋",
    "avocado": "🥑", "tomato": "🍅", "coconut": "🥥", "kiwi": "🥝",
    "broccoli": "🥦", "carrot": "🥕", "corn": "🌽", "cucumber": "🥒",
    "garlic": "🧄", "onion": "🧅", "pepper": "🌶️", "potato": "🥔",
    "eggplant": "🍆", "lettuce": "🥬", "mushroom": "🍄",
    "beetroot": "🫚", "spinach": "🥬", "sweetpotato": "🍠",
    "pomegranate": "🍷", "capsicum": "🫑", "paprika": "🌶️",
    "cauliflower": "🥦", "cabbage": "🥬", "ginger": "🫚",
    "bell pepper": "🫑", "chilli pepper": "🌶️", "peas": "🟢",
}

FRUITS = {
    "apple", "banana", "orange", "grape", "strawberry", "watermelon",
    "mango", "pineapple", "pear", "peach", "cherry", "lemon", "avocado",
    "tomato", "coconut", "kiwi", "pomegranate", "sweetpotato",
}

VEGETABLES = {
    "broccoli", "carrot", "corn", "cucumber", "garlic", "onion", "pepper",
    "potato", "eggplant", "lettuce", "mushroom", "beetroot", "spinach",
    "capsicum", "paprika", "cauliflower", "cabbage", "ginger", "bell pepper",
    "chilli pepper", "peas",
}

def get_emoji(label: str) -> str:
    label = label.lower()
    for key, emo in EMOJI_MAP.items():
        if key in label:
            return emo
    return "🌿"

def get_category(label: str) -> str:
    label_lower = label.lower()
    if label_lower in FRUITS:
        return "Fruit"
    if label_lower in VEGETABLES:
        return "Vegetable"
    for fruit in FRUITS:
        if fruit in label_lower:
            return "Fruit"
    for veg in VEGETABLES:
        if veg in label_lower:
            return "Vegetable"
    return "Unknown"

_detr_pipeline = None
_vit_pipeline  = None

def load_models():
    global _detr_pipeline, _vit_pipeline
    if _detr_pipeline is None:
        from transformers import pipeline
        _detr_pipeline = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            revision="no_timm",
            threshold=0.5,
        )
    if _vit_pipeline is None:
        from transformers import pipeline
        _vit_pipeline = pipeline(
            "image-classification",
            model="jazzmacedo/fruits-and-vegetables-detector-36",
            top_k=5,
        )
    return _detr_pipeline, _vit_pipeline

def classify(clf, region: Image.Image) -> tuple:
    preds = clf(region)
    if not preds:
        return "Unknown", 0.0
    top = preds[0]
    return top["label"].replace("_", " ").title(), top["score"]

def detect_produce(image: np.ndarray):
    if image is None:
        return None, "⚠️ Please upload an image first."

    detr, vit = load_models()
    pil_img = Image.fromarray(image).convert("RGB")
    W, H    = pil_img.size
    draw    = ImageDraw.Draw(pil_img)

    raw_detections = detr(pil_img)

    food_boxes = []
    for det in raw_detections:
        label = det["label"].lower()
        score = det["score"]
        if any(fl in label for fl in FOOD_COCO_LABELS) and score >= 0.40:
            box = det["box"]
            food_boxes.append((
                int(box["xmin"]), int(box["ymin"]),
                int(box["xmax"]), int(box["ymax"]),
                score
            ))

    annotated = []

    for idx, (x1, y1, x2, y2, _) in enumerate(food_boxes):
        crop = pil_img.crop((x1, y1, x2, y2))
        vit_label, vit_conf = classify(vit, crop)
        color = BOX_COLORS[idx % len(BOX_COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        category = get_category(vit_label)
        badge = f"  {category}: {vit_label}  {vit_conf:.0%}  "
        bw    = len(badge) * 8
        by    = max(y1 - 26, 0)
        draw.rectangle([x1, by, x1 + bw, by + 26], fill=color)
        draw.text((x1 + 4, by + 4), badge, fill="white")

        annotated.append((vit_label, vit_conf))

    whole_preds = vit(pil_img)
    top_whole   = [
        (p["label"].replace("_", " ").title(), p["score"])
        for p in whole_preds
    ]

    if not food_boxes and top_whole:
        best_label, best_conf = top_whole[0]
        color = BOX_COLORS[0]
        pad   = 12
        draw.rectangle([pad, pad, W - pad, H - pad], outline=color, width=4)
        category = get_category(best_label)
        badge = f"  {category}: {best_label}  {best_conf:.0%}  "
        bw    = len(badge) * 8
        draw.rectangle([pad, pad, pad + bw, pad + 26], fill=color)
        draw.text((pad + 4, pad + 4), badge, fill="white")

    report = "## 🔍 Detection Results\n\n"

    if annotated:
        report += "### 📦 Detected & Identified (DETR + ViT)\n"
        for lbl, conf in annotated:
            emo = get_emoji(lbl)
            category = get_category(lbl)
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            report += f"- {emo} **It is a {category}**: **{lbl}** `{bar}` {conf:.0%}\n"
        report += "\n"

    if top_whole:
        report += "### 🤖 Whole-Image Classification (36 classes)\n"
        for lbl, score in top_whole:
            emo = get_emoji(lbl.lower())
            category = get_category(lbl)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            report += f"- {emo} **It is a {category}**: **{lbl}** `{bar}` {score:.0%}\n"

    if not annotated and not top_whole:
        report += (
            "😕 **Nothing detected.**\n\n"
            "Tips:\n"
            "- Use a clear, well-lit photo 💡\n"
            "- Fruit/veg should fill most of the frame 🖼️\n"
            "- Try a close-up shot 🔍\n"
        )

    return np.array(pil_img), report


css = """
#title {
    text-align: center;
    font-size: 2.4em;
    font-weight: 800;
    background: linear-gradient(90deg, #4ade80, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
#subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.05em;
    margin-bottom: 10px;
}
#badge {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #4ade80;
    border-radius: 99px;
    padding: 4px 18px;
    font-size: 0.85em;
    margin-bottom: 24px;
}
.detect-btn {
    background: linear-gradient(90deg, #4ade80, #22d3ee) !important;
    color: #111 !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
    border: none !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="🥝 Fruit & Vegetable Detector",
    css=css,
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.HTML('<div id="title">🥝🥕 Fruit & Vegetable Detector</div>')
    gr.HTML('<div id="subtitle">Upload any photo — AI locates and accurately names every fruit or vegetable</div>')
    gr.HTML('<div style="text-align:center"><span id="badge">✅ DETR locates · ViT 36-class names accurately</span></div>')

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            inp_image = gr.Image(
                label="📤 Upload Your Image",
                type="numpy",
                height=420,
                sources=["upload", "clipboard"],
            )
            detect_btn = gr.Button(
                "🔍  Detect Fruits & Vegetables",
                elem_classes="detect-btn",
                size="lg",
            )
            gr.Markdown(
                "**Supported:** JPG · PNG · WEBP · BMP\n\n"
                "**36 classes:** Apple · Banana · Mango · Kiwi · Orange · Grape · "
                "Carrot · Broccoli · Tomato · Potato · Onion · Cucumber · and more!\n\n"
                "> 📌 **How it works:**\n"
                "> 1. **DETR** finds where objects are\n"
                "> 2. **ViT** accurately names each one\n"
            )

        with gr.Column(scale=1):
            out_image = gr.Image(label="📸 Annotated Result", height=420, interactive=False)
            out_text  = gr.Markdown()

    detect_btn.click(fn=detect_produce, inputs=inp_image, outputs=[out_image, out_text])

    gr.Markdown(
        "---\n"
        "**Models:** "
        "[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) (detect) + "
        "[jazzmacedo/ViT-36-class](https://huggingface.co/jazzmacedo/fruits-and-vegetables-detector-36) (name)  |  "
        "**Repo:** [SHAHRIYARTAUFIK/Fruit-and-veg-](https://github.com/SHAHRIYARTAUFIK/Fruit-and-veg-)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, show_error=True)
