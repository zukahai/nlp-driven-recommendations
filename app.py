import re
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Danh sách thông tin sản phẩm
products = [
    {"id": 1, "name": "Điện thoại iPhone 14 Pro Max", "description": "Smartphone cao cấp từ Apple."},
    {"id": 2, "name": "Laptop Dell XPS 13", "description": "Máy tính xách tay siêu mỏng, hiệu năng mạnh mẽ."},
    {"id": 3, "name": "Điện thoại Samsung Galaxy S23", "description": "Điện thoại thông minh hàng đầu của Samsung."},
    {"id": 4, "name": "Tai nghe Sony WH-1000XM5", "description": "Tai nghe chống ồn hàng đầu."},
    {"id": 5, "name": "MacBook Air M2", "description": "Laptop mỏng nhẹ với chip M2 từ Apple."},
    {"id": 6, "name": "Kính mát Ray-Ban Aviator", "description": "Kính mát phong cách với bán kính mắt 55mm."},
    {"id": 7, "name": "Kính bảo hộ chống bụi", "description": "Phù hợp cho môi trường làm việc nguy hiểm, bán kính mắt 50mm."}
]

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = unidecode.unidecode(text)  # Bỏ dấu tiếng Việt
    text = re.sub(r"[^\w\s]", "", text)  # Loại bỏ ký tự đặc biệt
    return text

# Hàm gợi ý sản phẩm dựa trên độ tương đồng văn bản
def suggest_products(query, products, top_n=3):
    # Chuẩn bị danh sách văn bản từ sản phẩm
    product_texts = [
        preprocess_text(product["name"] + " " + product["description"])
        for product in products
    ]
    
    # Tiền xử lý câu truy vấn
    query_processed = preprocess_text(query)
    
    # Tính TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(product_texts + [query_processed])
    
    # Tính độ tương đồng cosine
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    scores = cosine_sim.flatten()
    
    # Sắp xếp và chọn top N sản phẩm phù hợp
    recommended_indices = scores.argsort()[-top_n:][::-1]
    recommendations = [products[i] for i in recommended_indices]
    
    return recommendations

# Định nghĩa API
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data.get('query', '')
    top_n = data.get('top_n', 3)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    recommendations = suggest_products(query, products, top_n)
    return jsonify(recommendations)

# Get all products
@app.route('/products', methods=['GET'])
def get_products():
    return jsonify(products)

if __name__ == "__main__":
    app.run(debug=True)
