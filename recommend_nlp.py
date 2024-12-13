import re
import unidecode
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Danh sách thông tin sản phẩm - kính mắt với trường giá
product = []

def get_products():
    url = "http://127.0.0.1:8000/api/product"
    response = requests.get(url)
    products = response.json()
    return products

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = unidecode.unidecode(text)  # Bỏ dấu tiếng Việt
    text = re.sub(r"[^\w\s]", "", text)  # Loại bỏ ký tự đặc biệt
    return text

# Hàm để trích xuất giá từ câu truy vấn
def extract_price(query):
    price_match = re.findall(r'\d{4,}', query)
    if price_match:
        return int(price_match[0])  # Trả về giá đầu tiên tìm thấy (nếu có)
    return None  # Nếu không tìm thấy giá

# Nhận diện câu chào và tạm biệt mở rộng
def greet_or_bye(query):
    query = query.lower()
    greeting_responses = [
        "Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?",
        "Xin chào! Bạn cần tư vấn sản phẩm nào?",
        "Hi! Bạn đang tìm sản phẩm gì thế?"
    ]

    farewell_responses = [
        "Tạm biệt nhé! Chúc bạn một ngày tốt lành!",
        "Hẹn gặp lại bạn lần sau!",
        "Bye bye! Hy vọng gặp lại bạn sớm!"
    ]

    greetings = ["chào", "xin chào", "hello", "hi", "alo", "mình chào bạn", "bạn khỏe không"]
    farewells = ["tạm biệt", "bye", "hẹn gặp lại", "chúc bạn một ngày tốt", "chào tạm biệt", "hẹn dịp khác"]

    query = preprocess_text(query)

    for greeting in greetings:
        if preprocess_text(greeting) in query:
            return random.choice(greeting_responses)

    for farewell in farewells:
        if preprocess_text(farewell) in query:
            return random.choice(farewell_responses)

    return None

# Hàm gợi ý sản phẩm dựa trên độ tương đồng văn bản và lọc theo mức giá
def suggest_products(query, products, top_n=3):
    products = get_products()
    if greet_or_bye(query):
        return {"text": greet_or_bye(query)}

    if not products:
        return {"text": "Không có sản phẩm nào trong cơ sở dữ liệu."}

    price = extract_price(query)
    if price:
        from_price = max(price * 0.75 - 50000, 0)  # Đặt khoảng giá xung quanh giá tìm thấy
        to_price = price * 1.25
    else:
        from_price = 0
        to_price = 10**30  # Giá lớn vô cùng

    filtered_products = [
        product for product in products if from_price <= product["price"] <= to_price
    ]

    if not filtered_products:
        return {"text": "Chúng tôi không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."}

    product_texts = [
        preprocess_text(product["name"] + " " + product["description"])
        for product in filtered_products
    ]

    query_processed = preprocess_text(query)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(product_texts + [query_processed])

    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    scores = cosine_sim.flatten()

    recommended_indices = scores.argsort()[-top_n:][::-1]
    recommendations = [filtered_products[i] for i in recommended_indices]

    text = f"Có {len(recommendations)} sản phẩm phù hợp với yêu cầu của bạn:\n"

    for i, product in enumerate(recommendations):
        text += f"{i+1}. {product['name']} - {product['price']} VND\n"

    return {
        "text": text,
        "recommendations": recommendations
    }

# Phát hiện ý định
def detect_intent(query):
    if "giá" in query or re.search(r'\d{4,}', query):
        return "price"
    if any(word in query for word in ["mua", "kính"]):
        return "question"
    if any(word in query for word in ["làm sao", "như thế nào", "tại sao"]):
        return "general_question"
    if greet_or_bye(query):
        return "greeting_or_farewell"
    return "unknown"

# Xử lý ý định
def process_query(query, products):
    intent = detect_intent(query)

    if intent == "question":
        return suggest_products(query, products)
    elif intent == "general_question":
        return {"text": "Bạn có thể đặt câu hỏi cụ thể hơn được không? Ví dụ: 'Kính bảo hộ dùng khi nào?'"}
    elif intent == "greeting_or_farewell":
        return {"text": greet_or_bye(query)}
    else:
        return {"text": "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể thử hỏi cách khác được không?"}
