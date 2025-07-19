import pandas as pd

# Đọc file ratings.csv
ratings = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\ratings_cleaned.csv")

# Lấy danh sách userId duy nhất
unique_users = ratings['userId'].unique()

# Tạo DataFrame mới với cột userId và password (mặc định là "1")
users_df = pd.DataFrame({
    'userId': unique_users,
    'password': ['1'] * len(unique_users)
})

# Ghi ra file mới users.csv
users_df.to_csv("users.csv", index=False)

print("✅ Đã tạo file users.csv với cột userId và password.")
