import pandas as pd

# 📥 Đọc dữ liệu từ file
ratings = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest\ml-latest\ratings.csv")

# 🔧 Khoảng số lượt đánh giá mong muốn
min_ratings = 50
max_ratings = 150

# 🚫 Loại bỏ các dòng có NaN trong cột quan trọng
ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])

# 📊 Đếm số lượt đánh giá của mỗi user
user_counts = ratings['userId'].value_counts()

# 🔍 Giữ lại những user có lượt đánh giá trong khoảng [100, 300]
valid_users = user_counts[(user_counts >= min_ratings) & (user_counts <= max_ratings)].index

# 📌 Lọc dữ liệu giữ lại những user hợp lệ
filtered_ratings = ratings[ratings['userId'].isin(valid_users)]

# 💾 Xuất ra file CSV
filtered_ratings.to_csv("filtered_ratings_50_to_150.csv", index=False)

print(f"✅ Đã lọc và lưu dữ liệu với người dùng có từ {min_ratings} đến {max_ratings} đánh giá.")
