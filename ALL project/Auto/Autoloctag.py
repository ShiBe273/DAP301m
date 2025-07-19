import pandas as pd

# Đọc file tag và file filtered_ratings đã lọc
tags = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest\ml-latest\tags.csv")
filtered_ratings = pd.read_csv("filtered_ratings_50_to_150.csv")

# Lấy danh sách userId hợp lệ
valid_user_ids = filtered_ratings['userId'].unique()

# Lọc tag theo các user hợp lệ
filtered_tags = tags[tags['userId'].isin(valid_user_ids)]

# Xuất ra file mới
filtered_tags.to_csv("filtered_tags_50_to_150.csv", index=False)

print("✅ Đã lọc và lưu tag tương ứng với user từ 50 đến 150 lượt đánh giá.")
