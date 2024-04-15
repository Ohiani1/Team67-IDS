from flask import Flask
from flask import request, jsonify
from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId


app = Flask(__name__)
load_dotenv(find_dotenv())

password = os.environ.get("MONGODB_PWD")
connection_string = f"mongodb+srv://ohianiabdulkadir7:{password}@team67-ids.bsj4xtb.mongodb.net/?retryWrites=true&w=majority&appName=Team67-IDS"

client = MongoClient(connection_string)

if __name__ == "__main__":
    app.run()

