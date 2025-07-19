import pandas as pd
import random
import csv
from collections import defaultdict

def auto():
    # --- Configuration ---
    INPUT_FILE = r'C:\Users\ACER\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.12\dulieuphim1.csv'
    OUTPUT_FILE = 'ratings.csv'
    NUM_USERS = 1000
    RATINGS_PER_USER = 100  # Số lượt đánh giá cố định mỗi user
    MIN_COUNT_PER_MOVIE = 5
    RATING_CHOICES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    df_movies = pd.read_csv(INPUT_FILE)
    movie_ids = df_movies['id'].dropna().unique().tolist()

    ratings = []
    movie_rating_counts = defaultdict(int)
    user_rated_movies = defaultdict(set)
    user_pool = list(range(1, NUM_USERS + 1))

    # Step 1: Đảm bảo mỗi phim được đánh giá ít nhất 1 lần (seed)
    for movie_id in movie_ids:
        user_id = random.choice(user_pool)
        rating = random.choice(RATING_CHOICES)
        ratings.append([user_id, movie_id, rating])
        movie_rating_counts[movie_id] += 1
        user_rated_movies[user_id].add(movie_id)

    # Step 2: Mỗi user đánh giá đúng RATINGS_PER_USER phim
    for user_id in user_pool:
        available_movies = [m for m in movie_ids if m not in user_rated_movies[user_id]]
        sampled_movies = random.sample(available_movies, min(RATINGS_PER_USER, len(available_movies)))
        for movie_id in sampled_movies:
            rating = random.choice(RATING_CHOICES)
            ratings.append([user_id, movie_id, rating])
            movie_rating_counts[movie_id] += 1
            user_rated_movies[user_id].add(movie_id)

    # Step 3: Đảm bảo mỗi phim có ít nhất MIN_COUNT_PER_MOVIE lượt đánh giá
    for movie_id in movie_ids:
        while movie_rating_counts[movie_id] < MIN_COUNT_PER_MOVIE:
            user_id = random.choice(user_pool)
            if movie_id not in user_rated_movies[user_id]:
                rating = random.choice(RATING_CHOICES)
                ratings.append([user_id, movie_id, rating])
                movie_rating_counts[movie_id] += 1
                user_rated_movies[user_id].add(movie_id)

    # Step 4: Ghi toàn bộ dữ liệu ra file
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['userId', 'id', 'rating'])
        writer.writerows(ratings)

    print(f"✅ Đã tạo {len(ratings):,} lượt đánh giá và lưu vào '{OUTPUT_FILE}'")

def count():
    ratings_df = pd.read_csv('ratings.csv')
    rating_counts = ratings_df['id'].value_counts().reset_index()
    rating_counts.columns = ['id', 'count']
    rating_counts.to_csv('movie_rating_counts.csv', index=False)
    print("✅ Đã lưu file 'movie_rating_counts.csv' gồm số lượt đánh giá cho từng phim.")

def main():
    print("Chương trình tự động tạo đánh giá phim")
    print("1. Tạo đánh giá phim")
    print("2. Đếm số lượt đánh giá cho từng phim")
    print("exit: Thoát chương trình")
    while True:
        choice = input("Chọn chức năng (1 hoặc 2): ")
        if choice == '1':
            auto()
        elif choice == '2':
            count()
        elif choice.lower() == "exit":
            break

if __name__ == "__main__":
    main()
