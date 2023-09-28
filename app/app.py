from os.path import join, dirname, exists
from os import makedirs, scandir, remove
from flask import Flask, send_from_directory, send_file, flash, jsonify, abort, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import qrcode
import base64
from io import BytesIO
from PIL import Image
import scripts

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder='/upload')
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_folder(workspace, folder):
    path = join(workspace, folder)
    if not exists(path):
        makedirs(path)

@app.route('/api/upload1', methods=['POST'])
@cross_origin()
def create_task1():
    folder = request.form['code']
    create_folder(UPLOAD_FOLDER, folder)
        
    direct = app.config['UPLOAD_FOLDER']+"/"+folder
    for file in scandir(direct):
        remove(file.path)    
    
    if 'file' in request.form:
        encoded_image = request.form['file']
        decoded_image = base64.b64decode(encoded_image.split(',')[1])
        image = Image.open(BytesIO(decoded_image))
        filename = 'image.png'
        image.save(join(app.config['UPLOAD_FOLDER']+"/"+folder, 'image.png'))
    else:
        file1 = request.files['image']
        filename = secure_filename(file1.filename)
        file1.save(join(app.config['UPLOAD_FOLDER']+"/"+folder, filename))
    try:
        scripts.generate_all_with_processing(app.config['UPLOAD_FOLDER'] + "/" + folder + "/", filename)
        return jsonify({
            'url': [
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result1.png",
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result2.png",
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result3.png",
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result4.png",
            ],
            'json': [
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result1.json",
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result2.json",
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result3.json",
                f'https://{request.headers.get("Host")}/'+app.config['UPLOAD_FOLDER']+"/"+folder+"/result4.json",
                ]
        }), 200
    except Exception as e:
        print(e)
        return {"message": "Не удалось выполнить генерацию изображения"}, 400

@app.route('/api/qr', methods=['GET'])
@cross_origin()
def create_qr():
    hash = request.args.get('hash')
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data('https://mypoint.mydev.kz/assembly/'+hash)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="#FAFAFA")

    # Сохранение QR-кода в файл
    image_path = "qrcode.jpg"  # Путь к файлу изображения QR-кода
    img.save(image_path)
    return send_file(image_path, mimetype="image/png")
   
@app.route('/upload/<path:path>/<path:name>')
@cross_origin()
def send_static(path, name):
    return send_from_directory(app.config['UPLOAD_FOLDER'] + "/" + path, name)

@app.route('/')
@cross_origin()
def index():
    return "MyPointArt Generator API! "

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3021, debug=True)