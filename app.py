from flask import Flask, jsonify
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)

# Cấu hình MongoDB Atlas
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

def test_db_connection():
    try:
        mongo.db.command("ping")
        return True
    except Exception as e:
        print(f"Lỗi kết nối MongoDB: {e}")
        return False

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to AHP Location Decision System!"})

@app.route("/test-db", methods=["GET"])
def test_db():
    if test_db_connection():
        return jsonify({"message": "Kết nối MongoDB thành công!"})
    else:
        return jsonify({"message": "Lỗi kết nối MongoDB!"}), 500

if __name__ == "__main__":
    app.run(debug=True)
