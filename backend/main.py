from flask import Flask
from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient
load_dotenv(find_dotenv())

password = os.environ.get("MONGODB_PWD")
connection_string = f"mongodb+srv://ohianiabdulkadir7:{password}@team67-ids.bsj4xtb.mongodb.net/?retryWrites=true&w=majority&appName=Team67-IDS"

client = MongoClient(connection_string)

dbs = client.list_database_names()
print(dbs)
