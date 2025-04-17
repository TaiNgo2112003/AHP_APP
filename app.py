from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import numpy as np
import os
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://ahp-app-fe.onrender.com"]}}, supports_credentials=True)

# MongoDB Connection
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
# Thành phần | Đánh giá
# Chuẩn hoá cột (normalized_matrix) | ✅ Đúng theo AHP: mỗi phần tử được chia cho tổng cột
# Vector ưu tiên (priority vector)  | ✅ Được tính bằng trung bình dòng sau khi chuẩn hoá
# Tính lambda_max                   | ✅ Đúng công thức: λmax=(A⋅w)w\lambda_{\text{max}} = \frac{(A \cdot w)}{w}λmax​=w(A⋅w)​
# CI, CR                            | ✅ Tính chính xác với CI = (λ_max - n) / (n - 1) và so với Random Index
# is_consistent                     | ✅ Kiểm tra nếu CR < 0.1 (hoặc 10%)
# Helper function for AHP calculations
def calculate_ahp_weights(pairwise_matrix):
    """
    Calculate AHP weights from pairwise comparison matrix
    
    Args:
        pairwise_matrix (numpy.ndarray): Square matrix of pairwise comparisons
        
    Returns:
        dict: {
            'weights': list of weights,
            'consistency_ratio': float,
            'is_consistent': bool,
            'pairwise_matrix': original matrix as list,
            'normalized_matrix': normalized matrix as list,
            'lambda_max': float,
            'consistency_index': float
        }
    """
    # Normalize the matrix
    column_sums = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / column_sums

    # Calculate priority vector (eigenvector)
    priority_vector = normalized_matrix.mean(axis=1)

    # Calculate consistency
    n = pairwise_matrix.shape[0]
    lambda_max = (pairwise_matrix @ priority_vector / priority_vector).mean()
    consistency_index = (lambda_max - n) / (n - 1)

    # Random Index values for different matrix sizes
    random_index = {
        1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    consistency_ratio = consistency_index / random_index.get(n, 1.49)
 
    return {
        'weights': priority_vector.tolist(),
        'consistency_ratio': float(consistency_ratio),
        'is_consistent': bool(consistency_ratio < 0.1),
        'pairwise_matrix': pairwise_matrix.tolist(),
        'normalized_matrix': normalized_matrix.tolist(),
        'lambda_max': float(lambda_max),
        'consistency_index': float(consistency_index)
    }

def validate_comparisons(comparisons, criteria_ids):
    """
    Validate pairwise comparisons
    
    Args:
        comparisons (list): List of comparison dicts
        criteria_ids (list): List of criteria IDs
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    required_fields = ['criterion_a', 'criterion_b', 'value']
    
    for comp in comparisons:
        # Check required fields
        if not all(field in comp for field in required_fields):
            return False, "Each comparison must contain criterion_a, criterion_b and value"
            
        # Check criteria exist
        if comp['criterion_a'] not in criteria_ids or comp['criterion_b'] not in criteria_ids:
            return False, "Invalid criterion ID in comparison"
            
        # Check value is positive
        if comp['value'] <= 0:
            return False, "Comparison values must be positive numbers greater than 0"
            
    return True, ""

# Criteria API Endpoints (Step 01)
@app.route('/api/criteria', methods=['GET', 'POST'])
def manage_criteria():
    if request.method == 'GET':
        criteria = list(criteria_col.find({}, {'_id': 1, 'name': 1, 'description': 1, 'created_at': 1}))
        return jsonify([{
            '_id': str(c['_id']),
            'name': c['name'],
            'description': c.get('description', ''),
            'created_at': c.get('created_at', '')
        } for c in criteria])
    
    elif request.method == 'POST':
        data = request.json
        if not data.get('name'):
            return jsonify({'error': 'Name is required'}), 400
            
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
# Location API Endpoints (Step 02)
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
        if not data.get('name') or not data.get('address'):
            return jsonify({'error': 'Name and address are required'}), 400
            
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

# Pairwise Comparisons API Endpoints (Step 03)
@app.route('/api/pairwise', methods=['POST'])
def handle_pairwise_comparisons():
    try:
        data = request.json
        user_id = data.get('user_id')
        criteria_ids = data.get('criteria_ids', [])
        comparisons = data.get('comparisons', [])

        # Validate input
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        if not criteria_ids:
            return jsonify({'error': 'criteria_ids is required'}), 400
        if not comparisons:
            return jsonify({'error': 'comparisons are required'}), 400
            
        # Validate comparisons
        is_valid, error_msg = validate_comparisons(comparisons, criteria_ids)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

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

        # Store the comparison
        comparison_record = {
            'user_id': user_id,
            'criteria_ids': criteria_ids,
            'comparisons': comparisons,
            'weights': ahp_result['weights'],
            'consistency_ratio': ahp_result['consistency_ratio'],
            'is_consistent': ahp_result['is_consistent'],
            'pairwise_matrix': ahp_result['pairwise_matrix'],
            'normalized_matrix': ahp_result['normalized_matrix'],
            'lambda_max': ahp_result['lambda_max'],
            'consistency_index': ahp_result['consistency_index'],
            'created_at': datetime.utcnow()
        }
        result = pairwise_col.insert_one(comparison_record)
        print("Pairwise Matrix:")
        print(tabulate(matrix, headers=criteria_ids, showindex=criteria_ids, tablefmt="grid"))

        return jsonify({
            '_id': str(result.inserted_id),
            **ahp_result,
            'criteria_ids': criteria_ids
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Evaluation API Endpoints (step 04)
@app.route('/api/evaluate', methods=['POST'])
def evaluate_locations():
    try:
        data = request.json
        user_id = data.get('user_id')
        criteria_ids = data.get('criteria_ids', [])
        location_ids = data.get('location_ids', [])
        weights = data.get('weights', [])

        # Validate input
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        if not criteria_ids:
            return jsonify({'error': 'criteria_ids is required'}), 400
        if not location_ids:
            return jsonify({'error': 'location_ids is required'}), 400
        if len(weights) != len(criteria_ids):
            return jsonify({'error': 'Weights length must match criteria_ids length'}), 400

        # Get all locations with their scores
        locations = list(locations_col.find(
            {'_id': {'$in': [ObjectId(id) for id in location_ids]}},
            {'name': 1, 'address': 1, 'scores': 1}
        ))
        
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Evaluation Details API Endpoints (Step 05 - Get AHP details and evaluation results)
@app.route('/api/evaluation-details/<evaluation_id>', methods=['GET'])
def get_evaluation_details(evaluation_id):
    try:
        # Lấy kết quả đánh giá từ database
        evaluation = results_col.find_one({'_id': ObjectId(evaluation_id)})
        if not evaluation:
            return jsonify({'error': 'Evaluation not found'}), 404

        # Lấy thông tin AHP từ pairwise comparison gần nhất của user
        pairwise_data = pairwise_col.find_one(
            {'user_id': evaluation['user_id']},
            sort=[('created_at', -1)]
        )

        # Tạo response bao gồm cả kết quả đánh giá và thông tin AHP
        response_data = {
            'evaluation': {
                '_id': str(evaluation['_id']),
                'user_id': evaluation['user_id'],
                'criteria_ids': evaluation['criteria_ids'],
                'location_ids': evaluation['location_ids'],
                'weights': evaluation['weights'],
                'results': evaluation['results'],
                'created_at': evaluation['created_at'].strftime('%Y-%m-%d %H:%M:%S') if 'created_at' in evaluation else None
            },
            'ahp_details': {
                'pairwise_matrix': pairwise_data.get('pairwise_matrix') if pairwise_data else None,
                'normalized_matrix': pairwise_data.get('normalized_matrix') if pairwise_data else None,
                'weights': pairwise_data.get('weights') if pairwise_data else None,
                'lambda_max': pairwise_data.get('lambda_max') if pairwise_data else None,
                'consistency_index': pairwise_data.get('consistency_index') if pairwise_data else None,
                'consistency_ratio': pairwise_data.get('consistency_ratio') if pairwise_data else None,
                'is_consistent': pairwise_data.get('is_consistent') if pairwise_data else None
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# User History API Endpoints
@app.route('/api/user/<user_id>/history', methods=['GET'])
def get_user_history(user_id):
    try:
        comparisons = list(pairwise_col.find(
            {'user_id': user_id},
            {'criteria_ids': 1, 'weights': 1, 'consistency_ratio': 1, 'created_at': 1}
        ).sort('created_at', -1))
        
        evaluations = list(results_col.find(
            {'user_id': user_id},
            {'location_ids': 1, 'results': 1, 'created_at': 1}
        ).sort('created_at', -1))
        
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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