import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.sparse import load_npz
import plotly.express as px
import tabulate as tabulate
import base64
from PIL import Image
# === Load dữ liệu & mô hình ===
@st.cache_resource
def load_all():
    model_cf =  joblib.load(r"C:\Code\Kỳ 4\DAP391m\ALL project\svdpp_best_model_small.pkl")
    movies = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\movies_cleaned.csv")
    ratings = pd.read_csv(r"C:\Users\ACER\Desktop\Dataset\DATA\Movie\Moive lens\ml-latest-small\ml-latest-small\ratings_cleaned.csv")
    cosine_sim = np.load(r"C:\Code\Kỳ 4\DAP391m\ALL project\models\cosine_sim.npy")
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
if "change_pass_mode" not in st.session_state:
    st.session_state.change_pass_mode = False  # 👈 Thêm dòng này để tránh lỗi

# === Trang đăng nhập / đăng ký ===


def login_page():
    # Hàm chuyển ảnh sang base64
    def img_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # === Load ảnh logo chính + 3 ảnh góc phải dưới ===
    logo_base64 = img_to_base64(r"C:\Users\ACER\Downloads\Black Grey Minimalist Modern Cinema Logo.png")
    logo1 = img_to_base64(r"C:\Users\ACER\Downloads\z6823275587371_82f3a11ec0863ab2f3c0cf3bfa11d550.jpg")
    logo2 = img_to_base64(r"C:\Users\ACER\Downloads\z6823275587377_885a8a0f294bb5630759d020edc49d35.jpg")
    logo3 = img_to_base64(r"C:\Users\ACER\Downloads\z6823275587396_e89114528f3772d56fa4d22973963bb5.jpg")

    # === CSS giao diện ===
    st.markdown(f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {{
                font-family: 'Roboto', sans-serif;
                background: linear-gradient(to right, #e0f7fa, #e1f5fe);
                background-image: url('https://www.transparenttextures.com/patterns/white-wall-3.png');
                background-repeat: repeat;
            }}

            .centered-box {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin-top: 30px;
            }}

            .login-logo {{
                width: 160px;
                height: 160px;
                border-radius: 50%;
                box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
                margin-bottom: 20px;
                object-fit: cover;
            }}

            .login-container {{
                background-color: rgba(255, 255, 255, 0.85);
                border-radius: 16px;
                padding: 2.5rem 2rem 2rem;
                box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 420px;
                margin: auto;
            }}

            .stTextInput > div > div > input {{
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #d0d0d0;
                font-size: 16px;
            }}

            .stTextInput label {{
                font-weight: bold;
            }}

            .stButton > button {{
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 16px;
                background-color: #4a90e2;
                color: white;
                border: none;
                transition: background-color 0.3s ease;
            }}

            .stButton > button:hover {{
                background-color: #357ABD;
            }}

            .login-title {{
                text-align: center;
                margin-bottom: 2rem;
                font-size: 1.5rem;
                font-weight: bold;
                color: #263238;
            }}
        </style>
    """, unsafe_allow_html=True)

    # === Layout chính giữa ===
    st.markdown('<div class="centered-box">', unsafe_allow_html=True)

    # Logo chính
    st.markdown(f'<img src="data:image/png;base64,{logo_base64}" class="login-logo" />', unsafe_allow_html=True)

    # Tiêu đề
    st.markdown('<div class="login-title">🎬 MOVIE RECOMMENDATION SYSTEM</div>', unsafe_allow_html=True)

    # === Nhập thông tin đăng nhập ===
    user_id_input = st.text_input("🔑 Nhập User ID (số):", key="user_id_input")
    password_input = st.text_input("🔒 Nhập mật khẩu:", type="password", key="password_input")

    col1, col2, col3= st.columns(3)
    with col1:
        if st.button("Đăng nhập"):
            handle_login(user_id_input, password_input)
    with col2:
        if st.button("Đăng ký"):
            handle_register(user_id_input, password_input)
    with col3:
        if st.button("🔐 Đổi mật khẩu"):
            st.session_state.change_pass_mode = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # đóng centered-box
    
    # === 3 logo nhỏ ở góc phải dưới ===
    st.markdown(f"""
        <div style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 12px;
            z-index: 9999;
        ">
            <img src="data:image/png;base64,{logo1}" style="width:70px; height:70px; border-radius:50%; box-shadow:0 0 4px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{logo2}" style="width:70px; height:70px; border-radius:50%; box-shadow:0 0 4px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{logo3}" style="width:70px; height:70px; border-radius:50%; box-shadow:0 0 4px rgba(0,0,0,0.1);">
        </div>
    """, unsafe_allow_html=True)

def change_password_page():
    st.markdown('<div class="centered-box">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 ĐỔI MẬT KHẨU</div>', unsafe_allow_html=True)

    with st.form("change_pass_form"):
        user_input = st.text_input("🔑 Nhập User ID (số):", key="change_user_id")
        old_pass = st.text_input("🔒 Mật khẩu hiện tại:", type="password", key="old_pass")
        new_pass = st.text_input("🆕 Mật khẩu mới:", type="password", key="new_pass")
        confirm_pass = st.text_input("✅ Xác nhận mật khẩu mới:", type="password", key="confirm_pass")

        submitted = st.form_submit_button("Cập nhật mật khẩu")

        if submitted:
            if not user_input.isdigit():
                st.error("User ID phải là số.")
                return
            if new_pass != confirm_pass:
                st.error("Mật khẩu mới không khớp.")
                return
            user_id = int(user_input)

            if os.path.exists(USERS_CSV):
                users_df = pd.read_csv(USERS_CSV)
                users_df['userId'] = users_df['userId'].astype(int)
            else:
                st.error("Chưa có người dùng nào.")
                return

            if user_id in users_df['userId'].values:
                stored_password = users_df.loc[users_df['userId'] == user_id, 'password'].values[0]
                if str(old_pass) == str(stored_password):
                    users_df.loc[users_df['userId'] == user_id, 'password'] = new_pass
                    users_df.to_csv(USERS_CSV, index=False)
                    st.success("✅ Mật khẩu đã được cập nhật!")
                else:
                    st.error("❌ Mật khẩu hiện tại không đúng.")
            else:
                st.error("User ID không tồn tại.")

    # Nút quay lại login
    if st.button("🔙 Quay lại đăng nhập"):
        st.session_state.change_pass_mode = False
        st.rerun()

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
        st.error("Chưa có người dùng nào. Vui lòng đăng ký.")
        return

    if user_id in users_df['userId'].values:
        stored_password = users_df.loc[users_df['userId'] == user_id, 'password'].values[0]
        
        if str(password_input) == str(stored_password):
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.success(f"Đăng nhập thành công với User {user_id}")
        else:
            st.error(f"Sai mật khẩu.")
    else:
        st.error("User ID chưa tồn tại. Vui lòng đăng ký.")



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
        st.warning("User ID đã tồn tại. Vui lòng đăng nhập.")
    else:
        new_user = pd.DataFrame([[user_id, password_input]], columns=["userId", "password"])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_CSV, index=False)
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.success(f"Đăng ký thành công với User {user_id}✅ ")




# === Gợi ý phim CF ===
# def cf_recommend(user_id, model, ratings, movies_df, n):
#     seen = set(ratings[ratings['userId'] == user_id]['movieId'])
#     unseen = list(set(movies_df['movieId']) - seen)
#     preds = [model.predict(user_id, movie_id) for movie_id in unseen]
#     top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:n]
#     return pd.DataFrame([{
#         "movieId": p.iid,
#         "title": movies_df[movies_df['movieId'] == int(p.iid)]['title'].values[0],
#         "score": p.est,
#         "link": movies_df[movies_df['movieId'] == int(p.iid)]['link'].values[0]  # thêm dòng này
#     } for p in top_preds])

def cf_recommend(user_id, model, ratings_df, movies_df, top_n=10):
    # Tạo ánh xạ nhanh để lấy title và link
    movieId_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    movieId_to_link = dict(zip(movies_df['movieId'], movies_df['link']))

    # Phim user đã xem
    seen_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])

    # Phim chưa xem
    unseen_movie_ids = [mid for mid in movies_df['movieId'] if mid not in seen_movie_ids]

    # Dự đoán điểm cho các phim chưa xem
    predictions = [
        (mid, model.predict(user_id, mid).est)
        for mid in unseen_movie_ids
    ]

    # Lấy top N phim có điểm cao nhất
    top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    # Tạo DataFrame kết quả
    return pd.DataFrame([{
        "movieId": mid,
        "title": movieId_to_title.get(mid, "Unknown"),
        "score": score,
        "link": movieId_to_link.get(mid, "")
    } for mid, score in top_preds])

# === Gợi ý phim CBF ===
def cbf_recommend(user_id, ratings, movies_df, indices, cosine_sim, top_n=10):
    # Phim đã xem
    seen = set(ratings[ratings['userId'] == user_id]['movieId'])

    # Phim user đánh giá cao (làm gốc so sánh)
    liked = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]
    if liked.empty:
        return pd.DataFrame()

    # Tạo ánh xạ movieId -> title / link để truy xuất nhanh
    movieId_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    movieId_to_link  = dict(zip(movies_df['movieId'], movies_df['link']))
    movieId_to_index = dict(zip(movies_df['movieId'], movies_df.index))

    agg_scores = {}
    count_scores = {}

    for movie_id in liked['movieId']:
        title = movieId_to_title.get(movie_id)
        if not title or title not in indices:
            continue

        try:
            idx = indices[title]
            sims = list(enumerate(cosine_sim[idx]))
        except:
            continue  # bỏ qua nếu lỗi chỉ số

        for i, sim in sims:
            target_id = movies_df.iloc[i]['movieId']
            if target_id in seen or target_id == movie_id:
                continue

            agg_scores[target_id] = agg_scores.get(target_id, 0) + sim
            count_scores[target_id] = count_scores.get(target_id, 0) + 1

    # Tính điểm trung bình và sắp xếp
    scores = [(mid, agg_scores[mid] / count_scores[mid]) for mid in agg_scores]
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    return pd.DataFrame([{
        "movieId": mid,
        "title": movieId_to_title.get(mid, "Unknown"),
        "score": score,
        "link": movieId_to_link.get(mid, "")
    } for mid, score in top_scores])



# === Gợi ý Hybrid ===
def hybrid_recommend(user_id, model, ratings, movies_df, indices, cosine_sim, alpha=0.7, top_n=10):
    liked = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]
    if liked.empty: return pd.DataFrame()

    liked_movies = liked['movieId'].tolist()
    seen = set(ratings[ratings['userId'] == user_id]['movieId'])
    hybrid = []

    for movie_id in movies_df['movieId']:
        if movie_id in seen: continue
        try:
            # === CF ===
            cf_score = model.predict(user_id, movie_id).est
            cf_score = (cf_score - 1.0) / 4.0  # normalize về [0, 1]

            # === CBF ===
            similarities = []
            for liked_id in liked_movies:
                liked_title = movies_df[movies_df['movieId'] == liked_id]['title'].values[0]
                idx1 = indices.get(liked_title)
                idx2 = movies_df[movies_df['movieId'] == movie_id].index[0]
                sim = cosine_sim[idx1, idx2]
                similarities.append(sim)
            cbf_score = np.mean(similarities) if similarities else 0

            # === Kết hợp ===
            score = alpha * cf_score + (1 - alpha) * cbf_score
            link = movies_df.loc[idx2, 'link'] if 'link' in movies_df.columns else ''
            hybrid.append((movie_id, movies_df.loc[idx2, 'title'], score, link))
        except:
            continue

    top = sorted(hybrid, key=lambda x: x[2], reverse=True)[:top_n]
    return pd.DataFrame(top, columns=["movieId", "title", "score", "link"])


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

def main_page():
    # CSS cải tiến giao diện tổng thể + background + logo
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                background: linear-gradient(to right, #f2f6f9, #dfe9f3);
                background-image: url('https://www.transparenttextures.com/patterns/white-wall-3.png');
                background-repeat: repeat;

                font-family: 'Segoe UI', sans-serif;
            }

            .title-style {
                font-size: 36px;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                margin-top: 10px;
            }

            .section-box {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 25px 30px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
                margin-bottom: 30px;
            }

            .section-title {
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #34495e;
            }

            .stButton > button {
                background-color: #3498db;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                border: none;
                transition: background-color 0.3s ease;
            }

            .stButton > button:hover {
                background-color: #2980b9;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="title-style">🎥 Gợi ý phim cho User {st.session_state.user_id}</div>', unsafe_allow_html=True)

    # --- GỢI Ý THEO THỂ LOẠI ---
    with st.container():
        # st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📂 Gợi ý phim theo thể loại</div>', unsafe_allow_html=True)

        all_genres = set()
        for g in movies['genres'].dropna():
            all_genres.update(g.split('|'))
        all_genres = sorted(all_genres)

        selected_genres = st.multiselect("🎞️ Chọn thể loại yêu thích:", all_genres)

        if selected_genres:
            filtered_movies = movies[movies['genres'].apply(lambda x: all(genre in x.split('|') for genre in selected_genres))]

            if not filtered_movies.empty:
                st.success(f"Tìm thấy {len(filtered_movies)} phim thuộc thể loại: {', '.join(selected_genres)}")
                st.dataframe(
                    filtered_movies[['title', 'genres']],
                    height=300,  # Có thể chỉnh cao hơn nếu muốn
                    use_container_width=True
                )

            else:
                st.warning("Không tìm thấy phim nào khớp với các thể loại đã chọn.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ĐÁNH GIÁ PHIM ---
    with st.container():
        # st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⭐ Đánh giá phim</div>', unsafe_allow_html=True)

        with st.form("rate_form"):
            search_query = st.text_input("🔍 Nhập tên phim để tìm kiếm").strip()
            filtered_movies = movies[movies['title'].str.lower().str.contains(search_query.lower(), na=False)] if search_query else pd.DataFrame()

            if not filtered_movies.empty:
                movie_title = st.selectbox("🎬 Chọn phim từ kết quả tìm được:", filtered_movies['title'].unique())
                rating = st.slider("⭐ Đánh giá", 1.0, 5.0, 3.0, 0.5)

            submitted = st.form_submit_button("Lưu đánh giá")

            if submitted:
                if not filtered_movies.empty:
                    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
                    new_row = {
                        "userId": st.session_state.user_id,
                        "movieId": movie_id,
                        "rating": rating,
                        "timestamp": pd.Timestamp.now().timestamp()
                    }
                    ratings_path = r"C:\\Users\\ACER\\Desktop\\Dataset\\DATA\\Movie\\Moive lens\\ml-latest-small\\ml-latest-small\\ratings.csv"
                    df_ratings = pd.read_csv(ratings_path)
                    if not ((df_ratings['userId'] == new_row['userId']) & (df_ratings['movieId'] == new_row['movieId'])).any():
                        df_ratings.loc[len(df_ratings)] = new_row
                        df_ratings.to_csv(ratings_path, index=False)
                        st.success("Đã lưu đánh giá!✅")
                    else:
                        st.warning("Bạn đã đánh giá phim này trước đó.")
                else:
                    st.error("Không tìm thấy phim nào khớp với tên đã nhập.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- GỢI Ý PHIM ---
    with st.container():
        # st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Gợi ý phim</div>', unsafe_allow_html=True)

        max_possible = len(movies)
        top_n = st.number_input("Nhập số lượng phim muốn gợi ý:", min_value=1, step=1, value=10)

        if top_n > max_possible:
            st.error(f"Số lượng phim gợi ý vượt quá số phim có trong hệ thống ({max_possible}).")
            st.stop()

        col1, col2, col3 = st.columns(3)
        method = None
        with col1:
            if st.button("CF (Collaborative)"):
                method = "CF"
        with col2:
            if st.button("CBF (Content-based)"):
                method = "CBF"
        with col3:
            if st.button("Hybrid"):
                method = "Hybrid"

        if method:
            ratings = pd.read_csv(r"C:\\Users\\ACER\\Desktop\\Dataset\\DATA\\Movie\\Moive lens\\ml-latest-small\\ml-latest-small\\ratings.csv")

            if method == "CF":
                df = cf_recommend(st.session_state.user_id, model_cf, ratings, movies, top_n)
            elif method == "CBF":
                df = cbf_recommend(st.session_state.user_id, ratings, movies, indices, cosine_sim, top_n)
            else:
                df = hybrid_recommend(st.session_state.user_id, model_cf, ratings, movies, indices, cosine_sim, top_n=top_n)

            if df.empty:
                st.warning("Không thể tạo gợi ý.")
            else:
                df['link'] = df['link'].apply(lambda url: f"[🔗 Xem phim]({url})" if pd.notnull(url) and url != '' else "")
                st.markdown(df[['title', 'score', 'link']].to_markdown(index=False), unsafe_allow_html=True)
                show_chart(df)
                save_history(st.session_state.user_id, df)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- LỊCH SỬ GỢI Ý ---
    with st.container():
        # st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📜 Lịch sử gợi ý</div>', unsafe_allow_html=True)

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
        st.markdown('</div>', unsafe_allow_html=True)


# === Routing ===
if st.session_state.logged_in:
    main_page()
elif st.session_state.change_pass_mode:
    change_password_page()
else:
    login_page()
