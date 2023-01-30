from flask import Flask, request, jsonify, render_template
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import torchvision
from torchvision.io.image import read_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from itertools import groupby
from PIL import Image
import io

device = torch.device('cpu')

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and\
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def render():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('imagefile')
        ssd_threshold = float(request.form['SSDthreshold'])
        yolo_threshold = float(request.form['YOLOthreshold'])
        if file is None or file.filename == '':
            return render_template('index.html', alert='Upload a file first!')
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        img_path = 'static/images/' + file.filename.rsplit('.', maxsplit=1)[0]
        orig_img_path = img_path + '_original.png'
        ssd_img_path = img_path + '_ssd.png'
        yolo_img_path = img_path + '_yolo.png'
    try:

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(img)
        save_image(tensor, orig_img_path)

        ssd_model_weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        ssd_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=ssd_model_weights,
            device=device)
        ssd_model.eval()

        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        def draw_box(x1, y1, x2, y2, ax, label, score):
            rect = patches.Rectangle((x1, y1), (x2-x1), (y2-y1), linewidth=2, edgecolor='r', facecolor='none')
            props = dict(boxstyle='round', facecolor='red', alpha=0.5, edgecolor='None')
            ax.add_patch(rect)
            pad = 5
            ax.text(x1+pad, y1+pad, f"{label} {score.detach().numpy():.2f}", color='w', va='top', bbox=props)

        def ssd_detection(orig_img_path, dest_img_path, title='SSDLite', score_threshold=0.01):
            """Show a detection result from SSDLite model on a given axis."""

            ssd_model.score_thresh = score_threshold

            img = read_image(orig_img_path)

            preprocess = ssd_model_weights.transforms()
            batch = [preprocess(img)]
            prediction = ssd_model(batch)[0]

            labels = [
                ssd_model_weights.meta["categories"][i]
                for i in prediction["labels"]
            ]
            boxes = prediction["boxes"]
            scores = prediction["scores"]

            img = np.moveaxis(img.numpy(), 0, 2)

            fig = Figure(dpi=300)
            ax = fig.subplots()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.detach().numpy()
                draw_box(x1, y1, x2, y2, ax=ax, label=label, score=score)
            ax.imshow(img)
            # ax.set_title(title)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            fig.savefig(dest_img_path, format='jpg', bbox_inches='tight')

        def yolo_detection(orig_img_path, dest_img_path, title='YOLOv5', score_threshold=0.01):
            """Show a detection result from YOLOv5 model on a given axis."""
            
            yolo_model.conf = score_threshold

            result = yolo_model(orig_img_path)

            img = result.ims[0]

            fig = Figure(dpi=300)
            ax = fig.subplots()

            for *box, score, label_key in result.pred[0]:
                x1, y1, x2, y2 = list(map(lambda x: x.item(), box))
                draw_box(x1, y1, x2, y2, ax=ax, label=result.names[label_key.item()], score=score)
            ax.imshow(np.asarray(img))
            # ax.set_title(title)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            fig.savefig(dest_img_path, format='jpg', bbox_inches='tight')

        ssd_detection(orig_img_path, ssd_img_path, score_threshold=ssd_threshold)
        yolo_detection(orig_img_path, yolo_img_path, score_threshold=yolo_threshold)

        return render_template('index.html', orig_image_path=orig_img_path, ssd_img_path=ssd_img_path, yolo_img_path=yolo_img_path)

    except Exception as exc:
        return jsonify({'error': exc})

if __name__ == '__main__':
    app.run(port=5050, debug=True, host="0.0.0.0")