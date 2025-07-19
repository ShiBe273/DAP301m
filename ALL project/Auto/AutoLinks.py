import pandas as pd

# Đường dẫn đến file
movies_path = r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\movies_cleaned.csv"
links_path = r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\links.csv"
output_path = "links_cleaned.csv"

# Đọc dữ liệu
movies_df = pd.read_csv(movies_path)
links_df = pd.read_csv(links_path)

# Lọc các movieId hợp lệ (chỉ lấy những movieId có trong movies.csv)
valid_movie_ids = set(movies_df['movieId'])

# Giữ lại những dòng có movieId nằm trong danh sách hợp lệ
filtered_links_df = links_df[links_df['movieId'].isin(valid_movie_ids)]

# Xuất ra file mới
filtered_links_df.to_csv(output_path, index=False)

print(f"Đã lưu {len(filtered_links_df)} dòng hợp lệ vào '{output_path}'")
