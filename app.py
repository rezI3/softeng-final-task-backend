from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import json
import torch
import numpy as np
import pandas as pd
from aws import RekognitionImage
from ai import Model

app = Flask(__name__)

model = Model()


@app.route('/hello', methods=["GET"])
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ukiyoe', methods=["POST"])
def aws_test():
    data = request.get_json()

    if type(data) != dict:
        data = json.loads(data)

    img = data["base64"]

    img = base64.b64decode(img)
    img = BytesIO(img)

    rekognition = RekognitionImage(img.getvalue())
    # json_data に rekogのresponseが全部入ってる
    json_data = rekognition.detect_faces()

    json_data = json_data['FaceDetails']
    json_data = json_data[0]['Landmarks']

    x_list = []

    for obj in json_data:
        typo, x, y = obj.values()
        x_list.append(x)
        x_list.append(y)

    # input_dataをjson形式からtensor型に変換する
    x = np.array(x_list)
    x = torch.tensor(x, dtype=torch.float32)

    """ input_dataを推論器に渡して、出力(画像を返す) """
    output = detect_ukiyoe_face(x)

    print(output)
    # 出力結果に対応する浮世絵顔を用意
    labels = pd.read_csv('resources/label_data.csv')
    filename = labels['singleface_filename'].values[output]
    path = 'resources/arc_extracted_face_images/' + str(filename)
    print(filename)
    print(path)
    ukiyoe_img = Image.open(path)

    # 画像をjsonに変換
    buffered = BytesIO()
    ukiyoe_img.save(buffered, format="JPEG")
    ukiyoe_byte = buffered.getvalue()
    ukiyoe_base64 = base64.b64encode(ukiyoe_byte)
    ukiyoe_str = ukiyoe_base64.decode('utf-8')

    response = {"base64": ukiyoe_str}
    return jsonify(response)


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

    return jsonify(response)


def detect_ukiyoe_face(x):
    model.load_state_dict(torch.load('resources/model.pth'))
    y = model(x)
    y_label = torch.argmax(y, dim=0)
    return y_label.item()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
