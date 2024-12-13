from flask import Flask, request, jsonify
from flask_cors import CORS  # Import flask_cors
from recommend_nlp import process_query, get_products

app = Flask(__name__)
CORS(app)  # Thêm CORS vào ứng dụng Flask

# Định nghĩa API
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data.get('query', '')
    top_n = data.get('top_n', 3)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    products = get_products()
    recommendations = process_query(query, products)
    return jsonify(recommendations)

# Get all products
@app.route('/products', methods=['GET'])
def get_products_api():
    products = get_products()
    return jsonify(products)

if __name__ == "__main__":
    app.run(port=2712, debug=True)
