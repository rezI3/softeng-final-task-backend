from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import json

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route("/ukiyoe", methods=["GET", "POST"])
def parse():
    data = request.get_json()
    dict_data = json.loads(data)

    # base64をPillow形式に変換
    img = dict_data["img"]
    img = base64.b64decode(img)  # base64に変換された画像データをバイナリに変換
    img = BytesIO(img)  # _io.BytesIO pillowで扱えるように変換
    img = Image.open(img)  # Pillow形式に変換

    # 以後pillowのメソッドを使用して、画像操作を行う
    # ネガポジ反転する処理
    im_invert = ImageOps.invert(img)

    # Pillow形式に変換をbase64に変換する処理
    buffered = BytesIO()
    im_invert.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte)
    img_str = img_base64.decode('utf-8')

    responce = {'ukiyo_face': img_str}

    return jsonify(responce)


if __name__ == '__main__':
    app.run(debug=True)
