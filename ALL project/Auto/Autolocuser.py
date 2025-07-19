import pandas as pd

# Đọc file movies đã được làm sạch (không có IMAX đơn lẻ, không có 'no genres listed')
movies_cleaned = pd.read_csv('movies_cleaned.csv')

# Đọc file ratings gốc
ratings = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\ratings.csv")

# Lấy danh sách movieId hợp lệ
valid_movie_ids = set(movies_cleaned['movieId'])

# Giữ lại các dòng ratings có movieId tồn tại trong movies_cleaned
ratings_cleaned = ratings[ratings['movieId'].isin(valid_movie_ids)]

# Thông tin thống kê
print(f"Tổng số dòng ratings ban đầu: {len(ratings)}")
print(f"Số dòng ratings còn lại sau khi lọc: {len(ratings_cleaned)}")

# Lưu ra file mới
ratings_cleaned.to_csv('ratings_cleaned.csv', index=False)
