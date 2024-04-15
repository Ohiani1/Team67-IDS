from flask import Flask
from flask import request, jsonify
from dotenv import load_dotenv, find_dotenv
import os
import pprint
import datetime
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId


app = Flask(__name__)
load_dotenv(find_dotenv())

password = os.environ.get("MONGODB_PWD")
connection_string = f"mongodb+srv://ohianiabdulkadir7:{password}@team67-ids.bsj4xtb.mongodb.net/?retryWrites=true&w=majority&appName=Team67-IDS"

client = MongoClient(connection_string)
model_runs_db = client.model_Runs
collection = model_runs_db.model_Runs


@app.route('/save', methods=['POST'])
def save_run():

    json_data = request.json
    model = json_data['model']
    metrics = json_data['metrics']
    timestamp = datetime.datetime.now()

    if model and metrics:
        id = collection.insert_one({
            'model':model,
            'metrics':metrics,
            'timestamp':timestamp
        })

        resp = jsonify("run added successfully")
        resp.status_code = 200
        return resp
    
    else:
        return not_found()
    

@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status':404,
        'message':'Not Found ' + request.url
    }
    
    resp = jsonify(message)

    resp.status_code = 404

    return resp












if __name__ == "__main__":
    app.run()

    

