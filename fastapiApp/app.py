import datetime
import bcrypt
from bson.objectid import ObjectId
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt as jose_jwt
from pydantic import BaseModel
from typing import Optional, List, Any

# ──────────────────────────────────────────────
# TensorFlow / Sentiment Model (loaded once at startup)
# ──────────────────────────────────────────────
import numpy as np

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow import keras
    _sentiment_model = keras.models.load_model(
        "sentimentAnalysis.h5",
        custom_objects={"KerasLayer": hub.KerasLayer},
    )
    print("[INFO] Sentiment model loaded successfully.")
except Exception as _e:
    print(f"[WARN] Sentiment model could not be loaded: {_e}")
    _sentiment_model = None

# ──────────────────────────────────────────────
# App & Config
# ──────────────────────────────────────────────
app = FastAPI(title="E-Commerce API", version="2.0.0")

SECRET_KEY = "your_secret_key"   # Replace with a strong secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# In-memory "databases"
# ──────────────────────────────────────────────
users_db: dict = {}
products_db: dict = {}
admins_db: dict = {}


# ──────────────────────────────────────────────
# JWT helper
# ──────────────────────────────────────────────
def create_access_token(identity: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": identity, "exp": expire}
    return jose_jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ──────────────────────────────────────────────
# Pydantic request models
# ──────────────────────────────────────────────
class UserRegisterRequest(BaseModel):
    email: str
    username: str
    phone: str
    password: str
    cpassword: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class AuthTokenRequest(BaseModel):
    auth: str


class AddToCartRequest(BaseModel):
    uid: str
    product: dict


class RemoveFromCartRequest(BaseModel):
    uid: str
    cid: str


class DeleteOrderRequest(BaseModel):
    uid: str
    oid: str


class UserOrderRequest(BaseModel):
    uid: str
    order: dict


class AddCommentRequest(BaseModel):
    comment: str
    uid: str
    pid: str


class AddRatingRequest(BaseModel):
    rating: float
    pid: str


class AdminRegisterRequest(BaseModel):
    email: str
    companyName: str
    phone: str
    password: str
    cpassword: str


class AdminLoginRequest(BaseModel):
    email: str
    password: str


class AddProductRequest(BaseModel):
    adminId: str
    productName: str
    productPrice: float
    productUrl: str
    productCategory: str


# ══════════════════════════════════════════════
# USER ENDPOINTS
# ══════════════════════════════════════════════

@app.post("/userRegister", status_code=status.HTTP_201_CREATED)
def user_register(data: UserRegisterRequest):
    email = data.email
    username = data.username
    phone = data.phone
    password = data.password
    cpassword = data.cpassword

    if email in users_db:
        raise HTTPException(status_code=401, detail="Email already exists")
    if any(u["username"] == username for u in users_db.values()):
        raise HTTPException(status_code=401, detail="Username already exists")
    if any(u["phone"] == phone for u in users_db.values()):
        raise HTTPException(status_code=401, detail="Phone Number already exists")
    if password != cpassword:
        raise HTTPException(status_code=401, detail="Password Not Matching!")

    hashpw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    access_token = create_access_token(identity=email)

    users_db[email] = {
        "email": email,
        "password": hashpw,
        "username": username,
        "phone": phone,
        "tokens": [{"token": access_token}],
        "cartProducts": [],
        "orders": [],
    }

    return {"token": access_token}


@app.post("/userLogin", status_code=status.HTTP_201_CREATED)
def user_login(data: UserLoginRequest):
    user = users_db.get(data.email)
    if user and bcrypt.checkpw(
        data.password.encode("utf-8"), user["password"].encode("utf-8")
    ):
        access_token = create_access_token(identity=data.email)
        user["tokens"].append({"token": access_token})
        return {"token": access_token}

    raise HTTPException(status_code=401, detail="Invalid Username/Password")


@app.post("/getUserData", status_code=status.HTTP_201_CREATED)
def get_user_data(data: AuthTokenRequest):
    auth_token = data.auth
    for user in users_db.values():
        if any(t["token"] == auth_token for t in user["tokens"]):
            user_data = user.copy()
            user_data.pop("password", None)
            return user_data

    raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.put("/addtoCart", status_code=status.HTTP_201_CREATED)
def add_to_cart(data: AddToCartRequest):
    user = users_db.get(data.uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found!")
    if any(p["pid"] == data.product["pid"] for p in user["cartProducts"]):
        raise HTTPException(status_code=401, detail="Product Already Added!")

    user["cartProducts"].append(data.product)
    return {"message": "Product Added Successfully!"}


@app.put("/removefromCart", status_code=status.HTTP_201_CREATED)
def remove_from_cart(data: RemoveFromCartRequest):
    user = users_db.get(data.uid)
    if not user:
        raise HTTPException(status_code=401, detail="Something went wrong!")

    user["cartProducts"] = [
        p for p in user["cartProducts"] if str(p.get("_id")) != data.cid
    ]
    return {"message": "Product Removed Successfully!"}


@app.put("/deleteOrder", status_code=status.HTTP_201_CREATED)
def delete_order(data: DeleteOrderRequest):
    user = users_db.get(data.uid)
    if not user:
        raise HTTPException(status_code=401, detail="Something went wrong!")

    user["orders"] = [
        o for o in user["orders"] if str(o.get("_id")) != data.oid
    ]
    return {"message": "Order Cancelled Successfully!"}


@app.put("/userOrders", status_code=status.HTTP_201_CREATED)
def user_orders(data: UserOrderRequest):
    user = users_db.get(data.uid)
    if not user:
        raise HTTPException(status_code=401, detail="Something went wrong!")
    if any(o["pid"] == data.order["pid"] for o in user["orders"]):
        raise HTTPException(status_code=401, detail="Order already Placed")

    user["orders"].append(data.order)
    return {"message": "Order Placed Successfully!"}


@app.post("/logoutUser", status_code=status.HTTP_201_CREATED)
def logout_user(data: AuthTokenRequest):
    auth_token = data.auth
    for user in users_db.values():
        if any(t["token"] == auth_token for t in user["tokens"]):
            user["tokens"] = [t for t in user["tokens"] if t["token"] != auth_token]
            return {"message": "Logout Successfully!"}

    raise HTTPException(status_code=401, detail="Invalid or expired token!")


# ══════════════════════════════════════════════
# PRODUCT ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/getAllProducts", status_code=status.HTTP_201_CREATED)
def get_all_products():
    return list(products_db.values())


@app.post("/addComments", status_code=status.HTTP_201_CREATED)
def add_comments(data: AddCommentRequest):
    if _sentiment_model is None:
        raise HTTPException(status_code=500, detail="Sentiment model not available")

    product = products_db.get(data.pid)
    if not product:
        raise HTTPException(status_code=401, detail="Product not found!")

    user = users_db.get(data.uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found!")

    try:
        pred = _sentiment_model.predict([data.comment])[0][0]
        sentiment = 1 if float(pred) >= 0.5 else 0

        product.setdefault("comments", []).append(
            {
                "_id": str(ObjectId()),
                "uid": data.uid,
                "username": user["username"],
                "comment": data.comment,
                "sentiment": sentiment,
                "date": str(datetime.datetime.now()),
            }
        )
        return {"message": "Thanks for your Feedback!"}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail="Something went Wrong!")


@app.post("/addRating", status_code=status.HTTP_201_CREATED)
def add_rating(data: AddRatingRequest):
    product = products_db.get(data.pid)
    if not product:
        raise HTTPException(status_code=401, detail="Something went wrong!")

    prev_rating = product.get("rating", 0)
    product["rating"] = round((prev_rating + data.rating) / 2, 1)
    return {"message": "Thanks for Rating!"}


# ══════════════════════════════════════════════
# ADMIN ENDPOINTS
# ══════════════════════════════════════════════

@app.post("/adminRegister", status_code=status.HTTP_201_CREATED)
def admin_register(data: AdminRegisterRequest):
    email = data.email
    companyName = data.companyName
    phone = data.phone
    password = data.password
    cpassword = data.cpassword

    if email in admins_db:
        raise HTTPException(status_code=401, detail="Email already exists")
    if any(a["companyName"] == companyName for a in admins_db.values()):
        raise HTTPException(status_code=401, detail="Company Name already exists")
    if any(a["phone"] == phone for a in admins_db.values()):
        raise HTTPException(status_code=401, detail="Phone Number already exists")
    if password != cpassword:
        raise HTTPException(status_code=401, detail="Password Not Matching!")

    hashpw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    access_token = create_access_token(identity=email)

    admins_db[email] = {
        "email": email,
        "companyName": companyName,
        "phone": phone,
        "password": hashpw,
        "tokens": [{"token": access_token}],
        "products": [],
    }

    return {"token": access_token}


@app.post("/adminLogin", status_code=status.HTTP_201_CREATED)
def admin_login(data: AdminLoginRequest):
    admin = admins_db.get(data.email)
    if admin and bcrypt.checkpw(
        data.password.encode("utf-8"), admin["password"].encode("utf-8")
    ):
        access_token = create_access_token(identity=data.email)
        admin["tokens"].append({"token": access_token})
        return {"token": access_token}

    raise HTTPException(status_code=401, detail="Invalid Username/Password")


@app.post("/getAdminData")
def get_admin_data(data: AuthTokenRequest):
    auth_token = data.auth
    for admin in admins_db.values():
        if any(t["token"] == auth_token for t in admin["tokens"]):
            admin_data = admin.copy()
            admin_data.pop("tokens", None)
            admin_data.pop("password", None)
            return admin_data

    raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.post("/addProduct", status_code=status.HTTP_201_CREATED)
def add_product(data: AddProductRequest):
    admin = admins_db.get(data.adminId)
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found!")

    product_id = str(ObjectId())
    new_product = {
        "_id": product_id,
        "productName": data.productName,
        "productPrice": data.productPrice,
        "productUrl": data.productUrl,
        "productCategory": data.productCategory,
        "comments": [],
        "rating": 0,
    }

    products_db[product_id] = new_product
    admin.setdefault("products", []).append(new_product)

    return {"message": "Product Added Successfully!"}


# ──────────────────────────────────────────────
# Entry point (for running directly)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
