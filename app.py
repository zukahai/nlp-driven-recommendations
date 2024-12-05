import re
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Danh sách thông tin sản phẩm - kính mắt với trường giá
products = [
    {"id": 1, "name": "Kính mát Ray-Ban Aviator", "description": "Kính mát phong cách với bán kính mắt 55mm.", "price": 1500000},
    {"id": 2, "name": "Kính mát Oakley OO9208", "description": "Kính mát thể thao, thiết kế mạnh mẽ, bán kính mắt 60mm.", "price": 1800000},
    {"id": 3, "name": "Kính mát Gucci GG0061S", "description": "Kính mát thời trang cao cấp, bán kính mắt 58mm.", "price": 2500000},
    {"id": 4, "name": "Kính bảo hộ chống bụi", "description": "Phù hợp cho môi trường làm việc nguy hiểm, bán kính mắt 50mm.", "price": 800000},
    {"id": 5, "name": "Kính mát Prada PR 17SS", "description": "Kính mát thiết kế sang trọng, bán kính mắt 52mm.", "price": 2200000},
    {"id": 6, "name": "Kính mát Carrera 5001", "description": "Kính mát với kiểu dáng cổ điển, bán kính mắt 54mm.", "price": 1700000}
]

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = unidecode.unidecode(text)  # Bỏ dấu tiếng Việt
    text = re.sub(r"[^\w\s]", "", text)  # Loại bỏ ký tự đặc biệt
    return text

# Hàm để trích xuất giá từ câu truy vấn
def extract_price(query):
    # Sử dụng regex để tìm kiếm giá trong câu truy vấn (giả sử giá có dạng số với 6 chữ số)
    price_match = re.findall(r'\d{5,}', query)
    if price_match:
        return int(price_match[0])  # Trả về giá đầu tiên tìm thấy (nếu có)
    return None  # Nếu không tìm thấy giá

# Hàm gợi ý sản phẩm dựa trên độ tương đồng văn bản và lọc theo mức giá
def suggest_products(query, products, top_n=3):
    # Trích xuất giá từ câu truy vấn
    price = extract_price(query)
    if price:
        from_price = max(price * 0.75 - 50000, 0)  # Đặt khoảng giá xung quanh giá tìm thấy
        to_price = price * 1.25
    else:
        from_price = 0
        to_price = 10**30  # Giá lớn vô cùng

    print(f"Price range: {from_price} - {to_price}")
    # Lọc sản phẩm theo giá
    filtered_products = [
        product for product in products if from_price <= product["price"] <= to_price
    ]

    if not filtered_products:
        return []  # Trả về danh sách rỗng nếu không có sản phẩm nào phù hợp
    
    # Chuẩn bị danh sách văn bản từ sản phẩm
    product_texts = [
        preprocess_text(product["name"] + " " + product["description"])
        for product in filtered_products
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
    recommendations = [filtered_products[i] for i in recommended_indices]
    
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
