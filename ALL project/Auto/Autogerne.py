import pandas as pd

# Đọc file movies.csv
movies = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\movies.csv")

# Bước 1: Loại bỏ phim có genres là 'no genres listed'
movies = movies[movies['genres'] != '(no genres listed)']

# Bước 2: Hàm xử lý thể loại
def remove_imax(genres):
    genre_list = genres.split('|')
    # Nếu chỉ có IMAX duy nhất → trả về None để loại bỏ phim
    if genre_list == ['IMAX']:
        return None
    # Nếu có nhiều thể loại → loại bỏ IMAX nếu có
    genre_list = [g for g in genre_list if g != 'IMAX']
    return '|'.join(genre_list)

# Bước 3: Áp dụng hàm xử lý
movies['genres'] = movies['genres'].apply(remove_imax)

# Bước 4: Loại bỏ các phim bị trả về None
movies = movies.dropna(subset=['genres'])

# Kết quả
print(f"Số lượng phim sau khi xử lý IMAX và 'no genres listed': {len(movies)}")

# (Tùy chọn) Lưu ra file mới
movies.to_csv('movies_cleaned.csv', index=False)
