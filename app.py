from flask import Flask, jsonify
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

mongo_uri = os.getenv("MONGO_URI")
if mongo_uri is None:
    print("Error: MongoDB URI is not set in the environment.")
else:
    print(f"MongoDB URI loaded: {mongo_uri}")

client = MongoClient(mongo_uri, server_api=ServerApi('1'))

def test_db_connection():
    try:
        client.admin.command('ping')
        return True
    except Exception as e:
        error_message = f"Lỗi kết nối MongoDB: {str(e)}"
        print(error_message)  
        return error_message  

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to AHP Location Decision System!"})

@app.route("/test-db", methods=["GET"])
def test_db():
    result = test_db_connection()
    if result is True:
        return jsonify({"message": "Kết nối MongoDB thành công!"})
    else:
        return jsonify({"message": "Lỗi kết nối MongoDB!", "error": result}), 500

if __name__ == "__main__":
    app.run(debug=True)
