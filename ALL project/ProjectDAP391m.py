import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.sparse import load_npz
import plotly.express as px
import tabulate as tabulate
# === Load dữ liệu & mô hình ===
@st.cache_resource
def load_all():
    model_cf =  joblib.load(r"C:\Code\Kỳ 4\DAP391m\movie_rating_system\svdpp_best_model_small.pkl")
    movies = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\movies_cleaned.csv")
    ratings = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\ratings_cleaned.csv")
    cosine_sim = np.load(r"C:\Code\Kỳ 4\DAP391m\movie_rating_system\models\cosine_sim.npy")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    links = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\links_cleaned.csv")
    links['link'] = links['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{int(x):07d}" if pd.notnull(x) else "")  
    movies = pd.merge(movies, links[['movieId', 'link']], on="movieId", how="left")


    return model_cf, movies, ratings, cosine_sim, indices

model_cf, movies, ratings_df, cosine_sim, indices = load_all()

# === Khởi tạo session ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# === Trang đăng nhập / đăng ký ===
def login_page():
    # Giao diện đăng nhập
    st.markdown("""
        <style>
            .logo-container {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: -20px;
            }
            .logo-container img {
                max-width: 120px;
            }

            .block-container {
                padding-top: 1rem;
            }

            input[type="text"], input[type="password"] {
                width: 250px !important;
            }
        </style>

        <div class="logo-container">
            <img src="https://i.imgur.com/B5bO2zC.png" alt="Logo">
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="centered">', unsafe_allow_html=True)

    st.image(r"C:\Code\Kỳ 4\DAP391m\OIG1.webp", width=150)
    st.markdown("## 🎬 Movie Recommender System")
    st.markdown('</div>', unsafe_allow_html=True)

    user_id_input = st.text_input("Nhập User ID (số):", key="user_id_input")
    password_input = st.text_input("Nhập mật khẩu:", type="password", key="password_input")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Đăng nhập"):
            handle_login(user_id_input, password_input)
    with col2:
        if st.button("🆕 Đăng ký"):
            handle_register(user_id_input, password_input)

    st.markdown('</div>', unsafe_allow_html=True)


USERS_CSV = r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\users.csv"

def handle_login(user_input, password_input):
    if not user_input.isdigit():
        st.error("User ID phải là số.")
        return
    user_id = int(user_input)

    if os.path.exists(USERS_CSV):
        users_df = pd.read_csv(USERS_CSV)
        users_df['userId'] = users_df['userId'].astype(int)
    else:
        st.error("⚠️ Chưa có người dùng nào. Vui lòng đăng ký.")
        return

    if user_id in users_df['userId'].values:
        stored_password = users_df.loc[users_df['userId'] == user_id, 'password'].values[0]
        
        if str(password_input) == str(stored_password):
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.success(f"✅ Đăng nhập thành công với User {user_id}")
        else:
            st.error(f"❌ Sai mật khẩu.")
    else:
        st.error("❌ User ID chưa tồn tại. Vui lòng đăng ký.")



def handle_register(user_input, password_input):
    if not user_input.isdigit():
        st.error("User ID phải là số.")
        return
    if not password_input:
        st.error("Mật khẩu không được để trống.")
        return

    user_id = int(user_input)

    if os.path.exists(USERS_CSV):
        users_df = pd.read_csv(USERS_CSV)
        users_df['userId'] = users_df['userId'].astype(int)
    else:
        users_df = pd.DataFrame(columns=["userId", "password"])

    if user_id in users_df['userId'].values:
        st.warning("⚠️ User ID đã tồn tại. Vui lòng đăng nhập.")
    else:
        new_user = pd.DataFrame([[user_id, password_input]], columns=["userId", "password"])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_CSV, index=False)
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.success(f"✅ Đăng ký thành công với User {user_id}")



# === Gợi ý phim CF ===
def cf_recommend(user_id, model, ratings, movies_df, n):
    seen = set(ratings[ratings['userId'] == user_id]['movieId'])
    unseen = list(set(movies_df['movieId']) - seen)
    preds = [model.predict(user_id, movie_id) for movie_id in unseen]
    top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:n]
    return pd.DataFrame([{
        "movieId": p.iid,
        "title": movies_df[movies_df['movieId'] == int(p.iid)]['title'].values[0],
        "score": p.est,
        "link": movies_df[movies_df['movieId'] == int(p.iid)]['link'].values[0]  # thêm dòng này
    } for p in top_preds])


# === Gợi ý phim CBF ===
def cbf_recommend(user_id, ratings, movies_df, indices, cosine_sim, top_n=10):
    liked = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]
    if liked.empty: return pd.DataFrame()
    agg, count = {}, {}
    for movie_id in liked['movieId']:
        row = movies_df[movies_df['movieId'] == movie_id]
        if row.empty: continue
        title = row['title'].values[0]
        if title not in indices: continue
        idx = indices[title]
        sims = list(enumerate(cosine_sim[idx]))
        for i, s in sims:
            target_id = movies_df.iloc[i]['movieId']
            if target_id == movie_id: continue
            agg[i] = agg.get(i, 0) + s
            count[i] = count.get(i, 0) + 1
    scores = [(i, agg[i]/count[i]) for i in agg]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame([{
        "movieId": movies_df.iloc[i]['movieId'],
        "title": movies_df.iloc[i]['title'],
        "score": s,
        "link": movies_df.iloc[i]['link']
    } for i, s in top])


# === Gợi ý Hybrid ===
def hybrid_recommend(user_id, model, ratings, movies_df, indices, cosine_sim, alpha=0.7, top_n=10):
    liked = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]
    if liked.empty: return pd.DataFrame()
    try:
        title = movies_df[movies_df['movieId'] == liked.iloc[0]['movieId']]['title'].values[0]
    except: return pd.DataFrame()
    seen = set(ratings[ratings['userId'] == user_id]['movieId'])
    hybrid = []
    for movie_id in movies_df['movieId']:
        if movie_id in seen: continue
        try:
            cf_score = model.predict(user_id, movie_id).est
            idx1 = indices.get(title)
            idx2 = movies_df[movies_df['movieId'] == movie_id].index[0]
            cbf_score = cosine_sim[idx1, idx2]
            score = alpha * cf_score + (1 - alpha) * cbf_score
            link = movies_df.loc[idx2, 'link'] if 'link' in movies_df.columns else ''
            hybrid.append((movie_id, movies_df.loc[idx2, 'title'], score, link))
        except: continue
    top = sorted(hybrid, key=lambda x: x[2], reverse=True)[:top_n]
    return pd.DataFrame(hybrid, columns=["movieId", "title", "score", "link"])


# === Lưu lịch sử gợi ý ===
def save_history(user_id, df):
    os.makedirs("history", exist_ok=True)
    path = f"history/{user_id}.csv"
    df['timestamp'] = pd.Timestamp.now()
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)

# === Biểu đồ ===
def show_chart(df):
    fig = px.bar(
        df.sort_values("score"),
        x="score", y="title", orientation="h",
        color="score", color_continuous_scale="viridis"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# === Giao diện chính ===
def main_page():
    st.title(f"🎥 Gợi ý phim cho User {st.session_state.user_id}")
    
    st.subheader("📌 Gợi ý phim theo thể loại")

    # Lấy danh sách tất cả thể loại
    all_genres = set()
    for g in movies['genres'].dropna():
        all_genres.update(g.split('|'))
    all_genres = sorted(all_genres)

    # Cho phép chọn 1 hoặc nhiều thể loại
    selected_genres = st.multiselect("🎞️ Chọn thể loại yêu thích:", all_genres)

    if selected_genres:
        # Tìm phim chứa tất cả thể loại đã chọn
        filtered_movies = movies[movies['genres'].apply(lambda x: all(genre in x.split('|') for genre in selected_genres))]
        
        if not filtered_movies.empty:
            st.success(f"✅ Tìm thấy {len(filtered_movies)} phim thuộc thể loại: {', '.join(selected_genres)}")
            st.dataframe(filtered_movies[['title', 'genres']])
        else:
            st.warning("⚠️ Không tìm thấy phim nào khớp với các thể loại đã chọn.")

    # Đánh giá phim
    st.subheader("📌 Đánh giá phim")
    with st.form("rate_form"):
        search_query = st.text_input("🔍 Nhập tên phim để tìm kiếm").strip()
        filtered_movies = movies[movies['title'].str.lower().str.contains(search_query.lower(), na=False)] if search_query else pd.DataFrame()

        if not filtered_movies.empty:
            movie_title = st.selectbox("🎬 Chọn phim từ kết quả tìm được:", filtered_movies['title'].unique())
            rating = st.slider("⭐ Đánh giá", 1.0, 5.0, 3.0, 0.5)

        submitted = st.form_submit_button("📥 Lưu đánh giá")

        if submitted:
            if not filtered_movies.empty:
                movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
                new_row = {
                    "userId": st.session_state.user_id,
                    "movieId": movie_id,
                    "rating": rating,
                    "timestamp": pd.Timestamp.now().timestamp()
                }
                ratings_path = r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\ratings.csv"
                df_ratings = pd.read_csv(ratings_path)
                if not ((df_ratings['userId'] == new_row['userId']) & (df_ratings['movieId'] == new_row['movieId'])).any():
                    df_ratings.loc[len(df_ratings)] = new_row
                    df_ratings.to_csv(ratings_path, index=False)
                    st.success("✅ Đã lưu đánh giá!")
                else:
                    st.warning("⚠️ Bạn đã đánh giá phim này trước đó.")
            else:
                st.error("❌ Không tìm thấy phim nào khớp với tên đã nhập.")


    # Gợi ý
    st.subheader("📥 Gợi ý phim")
    method = st.radio("Chọn phương pháp", ["CF", "CBF", "Hybrid"])
    top_n = st.slider("Số phim", 5, 20, 10)

    if st.button("🎯 Tạo gợi ý"):
        ratings = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\ratings.csv")
        if method == "CF":
            df = cf_recommend(st.session_state.user_id, model_cf, ratings, movies, top_n)
        elif method == "CBF":
            df = cbf_recommend(st.session_state.user_id, ratings, movies, indices, cosine_sim, top_n)
        else:
            df = hybrid_recommend(st.session_state.user_id, model_cf, ratings, movies, indices, cosine_sim, top_n=top_n)
        if df.empty:
            st.warning("Không thể gợi ý.")
        else:
            df['link'] = df['link'].apply(lambda url: f"[🔗 Xem phim]({url})" if pd.notnull(url) and url != '' else "")
            st.markdown(df[['title', 'score', 'link']].to_markdown(index=False), unsafe_allow_html=True)

            show_chart(df)
            save_history(st.session_state.user_id, df)

    # Lịch sử gợi ý
    st.subheader("🕘 Lịch sử gợi ý")
    history_file = f"history/{st.session_state.user_id}.csv"
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        st.dataframe(history[['title', 'score', 'timestamp']].sort_values("timestamp", ascending=False))
    else:
        st.info("Chưa có lịch sử gợi ý.")

    if st.button("🔓 Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.rerun()

# === Routing ===
if st.session_state.logged_in:
    main_page()
else:
    login_page()
