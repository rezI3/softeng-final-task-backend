from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import json
from torch import nn
import torch
import numpy as np
app = Flask(__name__)


@app.route('/hello', methods=["GET"])
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ukiyoe', methods=["POST"])
def aws_test():
    from aws import RekognitionImage
    data = request.get_json()

    if type(data) != dict:
        data = json.loads(data)

    img = data["base64"]

    img = base64.b64decode(img)
    img = BytesIO(img)

    rekognition = RekognitionImage(img.getvalue())
    json_data = rekognition.detect_faces()

    print(json_data)

    json_data = json_data['FaceDetails']
    json_data = json_data[0]['Landmarks']

    x_list = []

    for obj in json_data:
        typo, x, y = obj.values()
        x_list.append(x)
        x_list.append(y)

    x = np.array(x_list)
    x = torch.tensor(x, dtype=torch.float32)

    # input_data に rekogのresponseが全部入ってる
    # input_dataをjson形式からtensor型に変換する

    """ input_dataを推論器に渡して、出力(画像を返す) """
    output = detect_ukiyoe_face(x)
    print(output)

    response = {"base64": output}
    return jsonify(response)


def detect_ukiyoe_face(x):

    INPUT_SIZE = 60
    OUTPUT_SIZE = 1000

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.fc = nn.Sequential(
                nn.Linear(INPUT_SIZE, 500),
                nn.ReLU(),
                nn.Linear(500, OUTPUT_SIZE),

            )

        def forward(self, x):
            x = self.fc(x)
            return x

    model = Model()
    model.load_state_dict(torch.load('/Users/katoyuga/Downloads/model.pth'))
    y = model(x)
    y_label = torch.argmax(y, dim=0)
    print(y_label)
    return y_label.item()


@app.route("/reverse", methods=["POST"])
def parse():
    data = request.get_json()
    if type(data) != dict:
        data = json.loads(data)

    # print(data)

    # base64をPillow形式に変換
    img = data["base64"]
    img = base64.b64decode(img)  # base64に変換された画像データをバイナリに変換
    img = BytesIO(img)  # _io.BytesIO pillowで扱えるように変換
    img = Image.open(img)  # Pillow形式に変換

    # ネガポジ反転する処理
    im_invert = ImageOps.invert(img)

    # Pillow形式に変換をbase64に変換する処理
    buffered = BytesIO()
    im_invert.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte)
    img_str = img_base64.decode('utf-8')

    response = {'base64': img_str}

    # print(response)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)