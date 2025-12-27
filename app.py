from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)   # FIXED


# ================= LOAD MODELS / DATA =====================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")


# ================= DATABASE CONFIG ========================
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()


# ================= DATABASE MODELS ========================
class Signup(db.Model):
    __tablename__ = 'signup'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# ================= UTILITY ================================
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text


# ================= RECOMMENDATION ENGINE ===================
def content_based_recommendations(train_data, item_name, top_n=10):

    if item_name not in train_data['Name'].values:
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    item_index = train_data[train_data['Name'] == item_name].index[0]

    similar_items = sorted(
        list(enumerate(cosine_sim[item_index])),
        key=lambda x: x[1],
        reverse=True
    )

    top_similar = similar_items[1:top_n + 1]
    recommended_indices = [i[0] for i in top_similar]

    return train_data.iloc[recommended_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]


# ================= IMAGE BANK =============================
random_image_urls = [
    "static/img/img_1.png", "static/img/img_2.png", "static/img/img_3.png",
    "static/img/img_4.png", "static/img/img_5.png", "static/img/img_6.png",
    "static/img/img_7.png", "static/img/img_8.png",
]


# ================= ROUTES ================================
@app.route("/")
def index():
    random_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_urls,
        random_price=random.choice(prices)
    )


@app.route("/main")
def main():
    return render_template('main.html')


@app.route("/index")
def indexredirect():
    random_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_urls,
        random_price=random.choice(prices)
    )


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        random_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

        return render_template(
            'index.html',
            trending_products=trending_products.head(8),
            truncate=truncate,
            random_product_image_urls=random_urls,
            random_price=random.choice(prices),
            signup_message='User signed up successfully!'
        )

    return render_template('index.html')


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        new_signin = Signin(username=username, password=password)
        db.session.add(new_signin)
        db.session.commit()

        random_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

        return render_template(
            'index.html',
            trending_products=trending_products.head(8),
            truncate=truncate,
            random_product_image_urls=random_urls,
            random_price=random.choice(prices),
            signup_message='User signed in successfully!'
        )

    return render_template('index.html')


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            return render_template('main.html', message="No recommendations available for this product.")

        random_prices = [random.randint(30, 200) for _ in range(len(content_based_rec))]
        random_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]

        return render_template(
            'main.html',
            content_based_rec=content_based_rec,
            truncate=truncate,
            random_product_image_urls=random_urls,
            random_prices=random_prices
        )

    return render_template('main.html')


# ================= RUN SERVER =============================
if __name__ == '__main__':   # FIXED
    app.run(debug=True)
