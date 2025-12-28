from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# ================= CONFIG ========================
app.secret_key = "your-secret-key-change-this-in-production"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ================= LOAD DATA =====================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# ================= DATABASE MODELS ========================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    cart_items = db.relationship('Cart', backref='user', lazy=True, cascade='all, delete-orphan')
    wishlist_items = db.relationship('Wishlist', backref='user', lazy=True, cascade='all, delete-orphan')
    orders = db.relationship('Order', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Cart(db.Model):
    __tablename__ = 'cart'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    product_image = db.Column(db.String(500))
    product_brand = db.Column(db.String(100))
    product_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, default=1)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)


class Wishlist(db.Model):
    __tablename__ = 'wishlist'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    product_image = db.Column(db.String(500))
    product_brand = db.Column(db.String(100))
    added_at = db.Column(db.DateTime, default=datetime.utcnow)


class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), default='Pending')  # Pending, Confirmed, Shipped, Delivered
    shipping_address = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    order_items = db.relationship('OrderItem', backref='order', lazy=True, cascade='all, delete-orphan')


class OrderItem(db.Model):
    __tablename__ = 'order_items'
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    product_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

class ProductRating(db.Model):
    __tablename__ = 'product_ratings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    review = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# Create tables
with app.app_context():
    db.create_all()


# ================= DECORATORS ========================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# ================= UTILITY ================================
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text


def get_current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None


from difflib import get_close_matches

def content_based_recommendations(train_data, item_name, top_n=10):
    # Try exact match first
    if item_name not in train_data['Name'].values:
        # Try partial matching
        all_product_names = train_data['Name'].str.lower().tolist()
        search_term = item_name.lower()
        
        # Find products containing the search term
        matching_products = train_data[train_data['Name'].str.lower().str.contains(search_term, na=False)]
        
        if matching_products.empty:
            # Try fuzzy matching as last resort
            close_matches = get_close_matches(search_term, all_product_names, n=1, cutoff=0.3)
            if close_matches:
                item_name = train_data[train_data['Name'].str.lower() == close_matches[0]]['Name'].iloc[0]
            else:
                return pd.DataFrame()
        else:
            # Use the first matching product
            item_name = matching_products['Name'].iloc[0]

    # Rest of the function remains the same
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    
    return train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

def collaborative_recommendations(user_id, top_n=10):
    """
    Collaborative filtering based on user's order history
    """
    user_orders = Order.query.filter_by(user_id=user_id).all()
    
    if not user_orders:
        return trending_products.head(top_n)
    
    # Get products from orders
    user_products = []
    for order in user_orders:
        for item in order.order_items:
            user_products.append(item.product_name)
    
    # Find similar products
    all_recommendations = pd.DataFrame()
    for product in user_products[-3:]:  # Last 3 products
        recs = content_based_recommendations(train_data, product, top_n=5)
        all_recommendations = pd.concat([all_recommendations, recs])
    
    # Remove duplicates and return
    return all_recommendations.drop_duplicates().head(top_n)


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
    current_user = get_current_user()
    
    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_urls,
        random_price=random.choice(prices),
        current_user=current_user
    )

@app.route("/signup", methods=['POST'])
def signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Validation
    if not username or not email or not password:
        flash('All fields are required!', 'danger')
        return redirect(url_for('index'))
    
    # Check if user already exists
    if User.query.filter_by(username=username).first():
        flash('Username already exists!', 'danger')
        return redirect(url_for('index'))
    
    if User.query.filter_by(email=email).first():
        flash('Email already registered!', 'danger')
        return redirect(url_for('index'))
    
    # Create new user
    new_user = User(username=username, email=email)
    new_user.set_password(password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        
        # Auto-login after signup
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        
        flash(f'Welcome {username}! Your account has been created successfully!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        db.session.rollback()
        flash('An error occurred. Please try again.', 'danger')
        return redirect(url_for('index'))

@app.route('/signin', methods=['POST'])
def signin():
    username = request.form.get('signinUsername')
    password = request.form.get('signinPassword')
    
    user = User.query.filter_by(username=username).first()
    
    if user and user.check_password(password):
        session['user_id'] = user.id
        session['username'] = user.username
        flash(f'Welcome back, {user.username}!', 'success')
    else:
        flash('Invalid username or password!', 'danger')
    
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route("/main")
def main():
    current_user = get_current_user()
    return render_template('main.html', current_user=current_user)


@app.route("/recommendations", methods=['POST'])
def recommendations():
    prod = request.form.get('prod')
    nbr = int(request.form.get('nbr', 10))
    
    content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
    
    if content_based_rec.empty:
        flash("No recommendations available for this product.", 'warning')
        return redirect(url_for('main'))
    
    random_prices = [random.randint(30, 200) for _ in range(len(content_based_rec))]
    random_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
    current_user = get_current_user()
    
    return render_template(
        'main.html',
        content_based_rec=content_based_rec,
        truncate=truncate,
        random_product_image_urls=random_urls,
        random_prices=random_prices,
        current_user=current_user
    )

@app.route('/add-to-cart', methods=['POST'])
@login_required
def add_to_cart():
    try:
        product_name = request.form.get('product_name')
        product_image = request.form.get('product_image')
        product_brand = request.form.get('product_brand')
        product_price = float(request.form.get('product_price', 0))
        
        # Check if item already in cart
        existing_item = Cart.query.filter_by(
            user_id=session['user_id'],
            product_name=product_name
        ).first()
        
        if existing_item:
            existing_item.quantity += 1
            flash('Product quantity updated in cart!', 'success')
        else:
            cart_item = Cart(
                user_id=session['user_id'],
                product_name=product_name,
                product_image=product_image,
                product_brand=product_brand,
                product_price=product_price
            )
            db.session.add(cart_item)
            flash('Product added to cart!', 'success')
        
        db.session.commit()
        return redirect(request.referrer or url_for('index'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding to cart: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('index'))
    
@app.route('/cart')
@login_required
def view_cart():
    cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
    total = sum(item.product_price * item.quantity for item in cart_items)
    current_user = get_current_user()
    
    return render_template('cart.html', cart_items=cart_items, total=total, current_user=current_user)


@app.route('/remove-from-cart/<int:item_id>')
@login_required
def remove_from_cart(item_id):
    cart_item = Cart.query.get_or_404(item_id)
    
    if cart_item.user_id != session['user_id']:
        flash('Unauthorized action!', 'danger')
        return redirect(url_for('view_cart'))
    
    db.session.delete(cart_item)
    db.session.commit()
    flash('Item removed from cart!', 'success')
    
    return redirect(url_for('view_cart'))


@app.route('/update-cart/<int:item_id>', methods=['POST'])
@login_required
def update_cart(item_id):
    cart_item = Cart.query.get_or_404(item_id)
    
    if cart_item.user_id != session['user_id']:
        flash('Unauthorized action!', 'danger')
        return redirect(url_for('view_cart'))
    
    new_quantity = int(request.form.get('quantity', 1))
    
    if new_quantity > 0:
        cart_item.quantity = new_quantity
        db.session.commit()
        flash('Cart updated!', 'success')
    else:
        db.session.delete(cart_item)
        db.session.commit()
        flash('Item removed from cart!', 'success')
    
    return redirect(url_for('view_cart'))


@app.route('/add-to-wishlist', methods=['POST'])
@login_required
def add_to_wishlist():
    product_name = request.form.get('product_name')
    product_image = request.form.get('product_image')
    product_brand = request.form.get('product_brand')
    
    # Check if already in wishlist
    existing = Wishlist.query.filter_by(
        user_id=session['user_id'],
        product_name=product_name
    ).first()
    
    if existing:
        flash('Product already in wishlist!', 'info')
    else:
        wishlist_item = Wishlist(
            user_id=session['user_id'],
            product_name=product_name,
            product_image=product_image,
            product_brand=product_brand
        )
        db.session.add(wishlist_item)
        db.session.commit()
        flash('Product added to wishlist!', 'success')
    
    return redirect(request.referrer or url_for('index'))


@app.route('/wishlist')
@login_required
def view_wishlist():
    wishlist_items = Wishlist.query.filter_by(user_id=session['user_id']).all()
    current_user = get_current_user()
    return render_template('wishlist.html', wishlist_items=wishlist_items, current_user=current_user)


@app.route('/profile')
@login_required
def profile():
    user = get_current_user()
    orders = Order.query.filter_by(user_id=session['user_id']).order_by(Order.created_at.desc()).all()
    return render_template('profile.html', user=user, orders=orders, current_user=user)


@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    if request.method == 'POST':
        # Get form data
        shipping_address = request.form.get('address')
        city = request.form.get('city')
        pincode = request.form.get('pincode')
        
        full_address = f"{shipping_address}, {city}, {pincode}"
        
        # Get cart items
        cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
        
        if not cart_items:
            flash('Your cart is empty!', 'warning')
            return redirect(url_for('view_cart'))
        
        # Calculate total
        subtotal = sum(item.product_price * item.quantity for item in cart_items)
        tax = subtotal * 0.1
        shipping = 5.0
        total = subtotal + tax + shipping
        
        # Create order
        new_order = Order(
            user_id=session['user_id'],
            total_amount=total,
            shipping_address=full_address,
            status='Pending'
        )
        db.session.add(new_order)
        db.session.flush()  # Get order ID
        
        # Create order items
        for item in cart_items:
            order_item = OrderItem(
                order_id=new_order.id,
                product_name=item.product_name,
                product_price=item.product_price,
                quantity=item.quantity
            )
            db.session.add(order_item)
        
        # Clear cart
        Cart.query.filter_by(user_id=session['user_id']).delete()
        
        db.session.commit()
        
        flash(f'Order placed successfully! Order ID: {new_order.id}', 'success')
        return redirect(url_for('profile'))
    
    # GET request
    cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
    
    if not cart_items:
        flash('Your cart is empty!', 'warning')
        return redirect(url_for('view_cart'))
    
    subtotal = sum(item.product_price * item.quantity for item in cart_items)
    current_user = get_current_user()
    
    return render_template('checkout.html', cart_items=cart_items, subtotal=subtotal, current_user=current_user)


@app.route("/search", methods=['GET'])
def search():
    query = request.args.get('q', '')
    min_price = request.args.get('min_price', 0, type=float)
    max_price = request.args.get('max_price', 1000, type=float)
    min_rating = request.args.get('min_rating', 0, type=float)
    
    # Filter products
    results = train_data[
        (train_data['Name'].str.contains(query, case=False, na=False)) &
        (train_data['Rating'] >= min_rating)
    ]
    
    current_user = get_current_user()
    return render_template('search_results.html', results=results, query=query, current_user=current_user)


# ================= ERROR HANDLERS ========================
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

@app.route('/remove-wishlist/<int:item_id>')
@login_required
def remove_from_wishlist(item_id):
    wishlist_item = Wishlist.query.get_or_404(item_id)
    
    if wishlist_item.user_id != session['user_id']:
        flash('Unauthorized action!', 'danger')
        return redirect(url_for('view_wishlist'))
    
    db.session.delete(wishlist_item)
    db.session.commit()
    flash('Item removed from wishlist!', 'success')
    
    return redirect(url_for('view_wishlist'))

@app.route('/search-suggestions', methods=['POST'])
def search_suggestions():
    term = request.form.get('term', '').lower()
    
    # Get matching product names
    matching_products = train_data[
        train_data['Name'].str.lower().str.contains(term, na=False)
    ]['Name'].head(10).tolist()
    
    return jsonify({'suggestions': matching_products})
@app.route('/rate-product', methods=['POST'])
@login_required
def rate_product():
    try:
        product_name = request.form.get('product_name')
        rating = float(request.form.get('rating'))
        review = request.form.get('review', '')
        
        # Check if user already rated this product
        existing_rating = ProductRating.query.filter_by(
            user_id=session['user_id'],
            product_name=product_name
        ).first()
        
        if existing_rating:
            existing_rating.rating = rating
            existing_rating.review = review
            flash('Your rating has been updated!', 'success')
        else:
            new_rating = ProductRating(
                user_id=session['user_id'],
                product_name=product_name,
                rating=rating,
                review=review
            )
            db.session.add(new_rating)
            flash('Thank you for your rating!', 'success')
        
        db.session.commit()
        return redirect(request.referrer or url_for('index'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error submitting rating: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('index'))



# ================= RUN SERVER =============================
if __name__ == '__main__':
    app.run(debug=True)