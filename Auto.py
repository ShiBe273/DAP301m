import pandas as pd
import random
import csv

# --- Configuration ---
INPUT_FILE = r'C:\Users\ACER\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.12\dulieuphim1.csv'               # Input CSV file with movie IDs
OUTPUT_FILE = 'ratings2.csv'   # Output ratings file
NUM_USERS = 1000                       # Number of fake users
MIN_RATINGS = 3                         # Each user rates at least 3 movies
MAX_RATINGS = 20                        # Each user rates at most 10 movies
RATING_CHOICES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# --- Load movie IDs ---
df_movies = pd.read_csv(INPUT_FILE)
movie_ids = df_movies['id'].dropna().unique().tolist()  # or 'movieId' if your column is named that

# --- Generate random ratings ---
ratings = []

for user_id in range(1, NUM_USERS + 1):
    num_ratings = random.randint(MIN_RATINGS, MAX_RATINGS)
    sampled_movies = random.sample(movie_ids, min(num_ratings, len(movie_ids)))
    
    for movie_id in sampled_movies:
        rating = random.choice(RATING_CHOICES)
        ratings.append([user_id, movie_id, rating])

# --- Write to CSV ---
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['userId', 'id', 'rating'])
    writer.writerows(ratings)

print(f"✅ Generated {len(ratings)} random ratings in '{OUTPUT_FILE}'")
