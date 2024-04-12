from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!!!</p>"

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    return(request.files)
        #f = request.files['the_file']
        #f.save('/var/www/uploads/uploaded_file.txt')