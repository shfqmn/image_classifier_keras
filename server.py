import keras
from flask import Flask, request, jsonify
from predict import run
 
# set the project root directory as the static folder, you can set others.
app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    img = request.form['image']
    return run(img)
    
if __name__ == '__main__':
    app.run()