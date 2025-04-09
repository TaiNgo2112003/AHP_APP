from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import numpy as np
import os
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS
from pymongo import MongoClient

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://ahp-app-fe.onrender.com"]}}, supports_credentials=True)
# MongoDB Connection4
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MongoDB URI is not set in environment variables")
client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True) 
db = client.ahp_store_location

# Collections
criteria_col = db.criteria
locations_col = db.locations
pairwise_col = db.pairwise_comparisons
results_col = db.results
users_col = db.users

# Helper function for AHP calculations
def calculate_ahp_weights(pairwise_matrix):
    print("Pairwise Matrix:\n", pairwise_matrix)

    # Normalize the matrix
    normalized_matrix = pairwise_matrix / pairwise_matrix.sum(axis=0)
    print("Normalized Matrix:\n", normalized_matrix)

    # Calculate priority vector (eigenvector)
    priority_vector = normalized_matrix.mean(axis=1)
    print("Priority Vector (Weights):\n", priority_vector)

    # Calculate consistency
    n = pairwise_matrix.shape[0]
    lambda_max = (pairwise_matrix @ priority_vector / priority_vector).mean()
    print("Lambda max:", lambda_max)

    consistency_index = (lambda_max - n) / (n - 1)
    print("Consistency Index:", consistency_index)

    random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    consistency_ratio = consistency_index / random_index.get(n, 1.49)
    print("Consistency Ratio:", consistency_ratio)

    return {
        'weights': priority_vector.tolist(),
        'consistency_ratio': consistency_ratio,
        'is_consistent': consistency_ratio < 0.1
    }


# Criteria API Endpoints
@app.route('/api/criteria', methods=['GET', 'POST'])
def manage_criteria():
    if request.method == 'GET':
        criteria = list(criteria_col.find({}))
        return jsonify([{
            '_id': str(c['_id']),
            'name': c['name'],
            'description': c.get('description', ''),
            'created_at': c.get('created_at', '')
        } for c in criteria])
    
    elif request.method == 'POST':
        data = request.json
        new_criterion = {
            'name': data['name'],
            'description': data.get('description', ''),
            'created_at': datetime.utcnow()
        }
        result = criteria_col.insert_one(new_criterion)
        return jsonify({
            '_id': str(result.inserted_id),
            **new_criterion
        }), 201

# Pairwise Comparisons API Endpoints
@app.route('/api/pairwise', methods=['POST'])
def handle_pairwise_comparisons():
    try:
        data = request.json
        user_id = data.get('user_id')
        criteria_ids = data.get('criteria_ids', [])
        comparisons = data.get('comparisons', [])

        if not user_id or not criteria_ids or not comparisons:
            return jsonify({'error': 'Missing required fields'}), 400

        # Create pairwise matrix
        n = len(criteria_ids)
        matrix = np.ones((n, n))

        for comp in comparisons:
            i = criteria_ids.index(comp['criterion_a'])
            j = criteria_ids.index(comp['criterion_b'])
            value = comp['value']
            matrix[i][j] = value
            matrix[j][i] = 1 / value

        # Calculate AHP weights
        ahp_result = calculate_ahp_weights(matrix)

        # 🔥 Convert np.float64 & np.bool_ to native Python types
        ahp_result['weights'] = [float(w) for w in ahp_result['weights']]
        ahp_result['consistency_ratio'] = float(ahp_result['consistency_ratio'])
        ahp_result['is_consistent'] = bool(ahp_result['is_consistent'])

        # Store the comparison
        comparison_record = {
            'user_id': user_id,
            'criteria_ids': criteria_ids,
            'comparisons': comparisons,
            'weights': ahp_result['weights'],
            'consistency_ratio': ahp_result['consistency_ratio'],
            'is_consistent': ahp_result['is_consistent'],
            'created_at': datetime.utcnow()
        }
        result = pairwise_col.insert_one(comparison_record)

        return jsonify({
            '_id': str(result.inserted_id),
            **ahp_result,
            'criteria_ids': criteria_ids
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Location API Endpoints
@app.route('/api/locations', methods=['GET', 'POST'])
def manage_locations():
    if request.method == 'GET':
        locations = list(locations_col.find({}))
        return jsonify([{
            '_id': str(loc['_id']),
            'name': loc['name'],
            'address': loc['address'],
            'coordinates': loc.get('coordinates', {}),
            'scores': loc.get('scores', {}),
            'created_at': loc.get('created_at', '')
        } for loc in locations])
    
    elif request.method == 'POST':
        data = request.json
        new_location = {
            'name': data['name'],
            'address': data['address'],
            'coordinates': data.get('coordinates', {}),
            'scores': data.get('scores', {}),
            'created_at': datetime.utcnow()
        }
        result = locations_col.insert_one(new_location)
        return jsonify({
            '_id': str(result.inserted_id),
            **new_location
        }), 201

# Evaluation API Endpoints
@app.route('/api/evaluate', methods=['POST'])
def evaluate_locations():
    data = request.json
    user_id = data['user_id']
    criteria_ids = data['criteria_ids']
    location_ids = data['location_ids']
    weights = data['weights']
    
    # Get all locations with their scores
    locations = list(locations_col.find({'_id': {'$in': [ObjectId(id) for id in location_ids]}}))
    
    # Calculate weighted scores for each location
    results = []
    for loc in locations:
        total_score = 0
        scores = []
        
        for i, criterion_id in enumerate(criteria_ids):
            criterion_score = loc['scores'].get(criterion_id, 0)
            weighted_score = criterion_score * weights[i]
            scores.append({
                'criterion_id': criterion_id,
                'raw_score': criterion_score,
                'weighted_score': weighted_score
            })
            total_score += weighted_score
        
        results.append({
            'location_id': str(loc['_id']),
            'location_name': loc['name'],
            'total_score': total_score,
            'scores': scores
        })
    
    # Sort by total score
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Store the evaluation result
    evaluation_record = {
        'user_id': user_id,
        'criteria_ids': criteria_ids,
        'location_ids': location_ids,
        'weights': weights,
        'results': results,
        'created_at': datetime.utcnow()
    }
    result = results_col.insert_one(evaluation_record)
    
    return jsonify({
        '_id': str(result.inserted_id),
        'results': results
    })

# User History API Endpoints
@app.route('/api/user/<user_id>/history', methods=['GET'])
def get_user_history(user_id):
    comparisons = list(pairwise_col.find({'user_id': user_id}).sort('created_at', -1))
    evaluations = list(results_col.find({'user_id': user_id}).sort('created_at', -1))
    
    return jsonify({
        'comparisons': [{
            '_id': str(c['_id']),
            'criteria_ids': c['criteria_ids'],
            'weights': c['weights'],
            'consistency_ratio': c['consistency_ratio'],
            'created_at': c['created_at']
        } for c in comparisons],
        'evaluations': [{
            '_id': str(e['_id']),
            'location_ids': e['location_ids'],
            'top_location': e['results'][0] if e['results'] else None,
            'created_at': e['created_at']
        } for e in evaluations]
    })

if __name__ == "__main__":
    app.run(debug=True)





# mongo_uri = os.getenv("MONGO_URI")
# if mongo_uri is None:
#     print("Error: MongoDB URI is not set in the environment.")
# else:
#     print(f"MongoDB URI loaded: {mongo_uri}")

# client = MongoClient(mongo_uri, server_api=ServerApi('1'))

# def test_db_connection():
#     try:
#         client.admin.command('ping')
#         return True
#     except Exception as e:
#         error_message = f"Lỗi kết nối MongoDB: {str(e)}"
#         print(error_message)
#         return error_message  