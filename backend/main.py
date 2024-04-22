from flask import Flask
from flask import request, jsonify, render_template 
import os
import pprint
import datetime
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId
from models.LCCDE_IDS import LCCDE_IDS
from models.Tree_based_IDS import tree_based_IDS
from models.MTH_IDS import MHT_IDS


app = Flask(__name__)

connection_string = f"mongodb+srv://ohianiabdulkadir7:MCMtakB28bAFxRJN@team67-ids.bsj4xtb.mongodb.net/?retryWrites=true&w=majority&appName=Team67-IDS"

client = MongoClient(connection_string)
model_runs_db = client.model_Runs
collection = model_runs_db.model_Runs

app.template_folder = os.path.join(os.path.dirname(__file__), '../Frontend')
app.static_folder = os.path.join(os.path.dirname(__file__), '../Frontend/static')

@app.route('/')
def index():
    return render_template('index.html')
#########################################################################################################################################

## upload runs with this api
@app.route('/save', methods=['POST'])
def save_run():

    json_data = request.json
    model = json_data['model']
    metrics = json_data['metrics']
    dataset = json_data['dataset']
    timestamp = datetime.datetime.now()

    if model and metrics:
        id = collection.insert_one({
            'model':model,
            'dataset':dataset,
            'metrics':metrics,
            'timestamp':timestamp
        })

        resp = jsonify("run added successfully")
        resp.status_code = 200
        return resp
    
    else:
        return not_found()
    
#########################################################################################################################################

## get runs with this api
@app.route('/prevruns', methods=['GET'])
def prevRuns():
    prevRuns = collection.find()
    resp = dumps(prevRuns)
    return resp
    
#########################################################################################################################################

## run a model with this api
@app.route('/run/<modelId>/<dataset>', methods=['GET', 'POST'])
@app.route('/run/<modelId>/<dataset>/<lr>/<ne>/<md>', methods=['GET', 'POST'])
def run(modelId, dataset, lr=None, ne=None, md=None):
    if modelId == "Decision Tree":
        result = tree_based_IDS(dataset)
    elif modelId == "LCCDE":
        result = LCCDE_IDS(dataset)
    elif modelId == "MHT":
        result = MHT_IDS(file=dataset, lr=lr, ne=ne ,md=md)
    else:
        print(modelId + "we've reached end!")
        return not_found()

    print(modelId + " we've reached end!")
    return jsonify(result)
        
#########################################################################################################################################



@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status':404,
        'message':'Not Found ' + request.url
    }
    
    resp = jsonify(message)

    resp.status_code = 404

    return resp





## main function to run the server
if __name__ == "__main__":
    app.run()

    

