import re
import unidecode
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Danh sách thông tin sản phẩm - kính mắt với trường giá
product = []
url = "http://127.0.0.1:8000/api/product"
response = requests.get(url)
products = response.json()

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
    print(greet_or_bye(query))
    if greet_or_bye(query):
        print(greet_or_bye(query))
        return {"text": greet_or_bye(query)}
    # Trích xuất giá từ câu truy vấn
    price = extract_price(query)
    if price:
        from_price = max(price * 0.75 - 50000, 0)  # Đặt khoảng giá xung quanh giá tìm thấy
        to_price = price * 1.25
    else:
        from_price = 0
        to_price = 10**30  # Giá lớn vô cùng

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

    text = f"Có {len(recommendations)} sản phẩm phù hợp với yêu cầu của bạn: "

    # Thêm danh sách sản phẩm gợi ý và text
    for i, product in enumerate(recommendations):
        text += f"{i+1}. {product['name']} - {product['price']} VND " + "\n"

    if not recommendations:
        text = 'Chúng tôi không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.'

    return {
        "text": text,
        "recommendations": recommendations
    }

# Hàm nhận diện câu chào và tạm biệt
def greet_or_bye(query):
    greetings = ["chào", "xin chào", "hello", "hi", "chào bạn"]
    farewells = ["tạm biệt", "bye", "hẹn gặp lại", "chúc bạn một ngày tốt", "chào tạm biệt"]

    query = query.lower()
    print( query)
    
    # Kiểm tra câu chào
    for greeting in greetings:
        if greeting in query:
            return "Chào bạn! Bạn cần tôi giúp gì không?"

    # Kiểm tra câu tạm biệt
    for farewell in farewells:
        if farewell in query:
            return "Tạm biệt! Hẹn gặp lại bạn sau nhé!"
    
    return None  # Trả về None nếu không nhận diện được câu chào hoặc tạm biệt
