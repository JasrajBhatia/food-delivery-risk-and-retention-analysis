import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

print("Setting up configuration...")

# ─────────────────────────────────────────
# GEOGRAPHY
# ─────────────────────────────────────────
CITIES = {
    "Dubai":     0.60,
    "Abu Dhabi": 0.20,
    "Sharjah":   0.15,
    "Ajman":     0.05,
}

AREAS = {
    "Dubai": [
        "Marina", "JLT", "Downtown", "Business Bay", "DIFC",
        "Jumeirah", "Deira", "Al Barsha", "Mirdif", "Silicon Oasis",
        "Palm Jumeirah", "Dubai Hills", "Arabian Ranches", "Motor City",
        "Sports City", "Discovery Gardens", "International City",
        "Al Quoz", "Bur Dubai", "Karama", "Satwa", "Oud Metha",
        "Garhoud", "Festival City", "Creek Harbour", "Nad Al Sheba",
        "Al Furjan", "Jumeirah Village Circle", "Jumeirah Village Triangle",
        "Dubai South", "Al Nahda Dubai", "Muhaisnah",
    ],
    "Abu Dhabi": [
        "Corniche", "Al Reem Island", "Khalidiyah", "Yas Island",
        "Saadiyat Island", "Al Mushrif", "Al Khalidiyah", "Al Karamah",
        "Mohammed Bin Zayed City", "Shahama", "Al Reef", "Masdar City",
        "Al Shamkha", "Khalifa City", "Al Raha Beach", "Tourist Club Area",
        "Electra Street", "Hamdan Street", "Al Muroor", "Al Zaab",
    ],
    "Sharjah": ["Al Nahda", "Al Majaz", "Al Taawun", "Muwaileh"],
    "Ajman":   ["Al Rashidiya", "Al Nuaimiya", "Al Jurf"],
}

AREA_WEIGHTS = {
    "Dubai": [
        0.07, 0.06, 0.07, 0.06, 0.04,
        0.06, 0.05, 0.05, 0.03, 0.02,
        0.04, 0.04, 0.02, 0.02,
        0.02, 0.03, 0.03,
        0.02, 0.04, 0.03, 0.02, 0.02,
        0.02, 0.02, 0.02, 0.01,
        0.02, 0.02, 0.02,
        0.01, 0.01, 0.01,
    ],
    "Abu Dhabi": [
        0.09, 0.07, 0.07, 0.06,
        0.05, 0.05, 0.05, 0.05,
        0.05, 0.04, 0.05, 0.03,
        0.04, 0.06, 0.05, 0.04,
        0.03, 0.04, 0.04, 0.04,
    ],
    "Sharjah": [0.30, 0.25, 0.25, 0.20],
    "Ajman":   [0.40, 0.35, 0.25],
}

# ─────────────────────────────────────────
# CUISINES WITH DEMOGRAPHIC WEIGHTS
# ─────────────────────────────────────────
CUISINES = {
    "Indian":    0.28,
    "Pakistani": 0.10,
    "Arabic":    0.12,
    "Lebanese":  0.08,
    "Filipino":  0.08,
    "Chinese":   0.08,
    "American":  0.10,
    "Italian":   0.07,
    "Japanese":  0.05,
    "Thai":      0.02,
    "Mexican":   0.01,
    "Sri Lankan":0.01,
}

# ─────────────────────────────────────────
# RESTAURANT NAMES PER CUISINE
# ─────────────────────────────────────────
RESTAURANT_NAMES = {
    "Indian": [
        "Spice Route", "Bombay Kitchen", "Delhi Darbar", "Curry Leaf", "Maharaja Palace",
        "Punjab Grill", "Tandoor House", "Masala Junction", "Royal Biryani House", "Saffron Table",
        "Mumbai Street Food", "Bollywood Bites", "Golden Curry", "Namaste Kitchen", "Chai & Spice"
    ],
    "Pakistani": [
        "Karachi Darbar", "Lahori Chaska", "BBQ Tonight", "Islamabad Grill", "Peshawar Wala",
        "Al Hamra Pakistani", "Desi Zaika", "Nihari House", "Haleem Express", "Green Valley Pakistani"
    ],
    "Arabic": [
        "Al Fanar", "Arabian Nights Kitchen", "Emirati Table", "Khaleeji Bites", "Al Safadi",
        "Shawarma Station", "Mandi House", "Al Dawaar Grill", "Bedouin Kitchen", "Suhoor Lounge"
    ],
    "Lebanese": [
        "Zaatar w Zeit", "Babel Beirut", "Tawlet", "Leila Restaurant", "Al Waha Lebanese",
        "Beirut Express", "Cedar Tree", "Byblos on the Sea", "Fattoush Kitchen", "Mezze House"
    ],
    "Filipino": [
        "Ihaw Ihaw", "Jollibee Express", "Manila Garden", "Pinoy Kitchen", "Lutong Bahay",
        "Adobo House", "Sisig Station", "Pampanga Kitchen", "Tagalog Tastes", "Sarap Dito"
    ],
    "Chinese": [
        "Wok to Walk", "Dim Sum Palace", "Golden Dragon", "Jade Garden", "Peking House",
        "Noodle Bar Shanghai", "Lucky Wok", "Dynasty Kitchen", "Chao Fan Express", "Dumpling Republic"
    ],
    "American": [
        "Burger District", "Smash & Stack", "The Wing Co", "Liberty Grill", "All American Diner",
        "Shake Shack Express", "BBQ Smokehouse", "Five Guys Kitchen", "Nashville Hot", "Patty Perfect"
    ],
    "Italian": [
        "Bella Napoli", "Trattoria Roma", "Pizza Express Dubai", "La Cucina", "Il Pomodoro",
        "Pasta Fresca", "Vesuvio Kitchen", "Amalfi Table", "Nonna Rosa", "Capricciosa"
    ],
    "Japanese": [
        "Sakura Sushi", "Ramen Republic", "Tokyo Kitchen", "Nobu Express", "Izakaya House",
        "Edo Japanese", "Wasabi Kitchen", "Sashimi Bar", "Yakitori Yoshi", "Katsu Corner"
    ],
    "Thai": [
        "Thai Orchid", "Bangkok Bites", "Pad Thai House", "Sawadee Kitchen", "Lotus Thai"
    ],
    "Mexican": [
        "Taco Loco", "El Rancho Mexican", "Burrito Bandido"
    ],
    "Sri Lankan": [
        "Colombo Kitchen", "Ceylon Spice", "Lanka Bites"
    ],
}

# ─────────────────────────────────────────
# MENU ITEMS PER CUISINE
# ─────────────────────────────────────────
MENU_ITEMS = {
    "Indian":    ["Butter Chicken", "Chicken Biryani", "Dal Makhani", "Garlic Naan", "Palak Paneer",
                  "Lamb Rogan Josh", "Samosa (2pcs)", "Mango Lassi", "Chicken Tikka Masala", "Gulab Jamun"],
    "Pakistani": ["Nihari", "Haleem", "Karahi Chicken", "Seekh Kebab", "Mutton Biryani",
                  "Aloo Gosht", "Chapli Kebab", "Paya Soup", "Daal Chawal", "Sheer Khurma"],
    "Arabic":    ["Chicken Shawarma", "Lamb Mandi", "Kabsa", "Falafel Wrap", "Grilled Hammour",
                  "Hummus Platter", "Mixed Grill", "Machboos", "Jareesh", "Dates & Karak"],
    "Lebanese":  ["Chicken Tawook", "Fattoush Salad", "Mezze Platter", "Kafta Wrap", "Lamb Kibbeh",
                  "Baba Ghanoush", "Manakeesh Zaatar", "Warak Dawali", "Shanklish Salad", "Knafeh"],
    "Filipino":  ["Chicken Adobo", "Sinigang na Baboy", "Kare Kare", "Lechon Kawali", "Pancit Canton",
                  "Sisig Rice Bowl", "Crispy Pata", "Halo Halo", "Beef Caldereta", "Longganisa Rice"],
    "Chinese":   ["Kung Pao Chicken", "Dim Sum Basket", "Beef Chow Mein", "Fried Rice Special",
                  "Sweet & Sour Pork", "Mapo Tofu", "Spring Rolls (4pcs)", "Wonton Soup",
                  "Peking Duck Wrap", "Steamed Dumplings"],
    "American":  ["Classic Smash Burger", "BBQ Chicken Wings", "Mac & Cheese", "Loaded Fries",
                  "Crispy Chicken Sandwich", "Beef Hot Dog", "Onion Rings", "Philly Cheesesteak",
                  "Nashville Hot Tenders", "Chocolate Milkshake"],
    "Italian":   ["Margherita Pizza", "Pasta Carbonara", "Chicken Parmigiana", "Tiramisu",
                  "Risotto ai Funghi", "Lasagne al Forno", "Bruschetta", "Penne Arrabbiata",
                  "Cannoli Siciliani", "Caprese Salad"],
    "Japanese":  ["Salmon Sushi Roll", "Tonkotsu Ramen", "Chicken Katsu Curry", "Edamame",
                  "Beef Gyoza (6pcs)", "Dragon Roll", "Miso Soup", "Teriyaki Chicken Bowl",
                  "Matcha Ice Cream", "Yakitori Skewers"],
    "Thai":      ["Pad Thai", "Green Curry Chicken", "Tom Yum Soup", "Mango Sticky Rice", "Thai Basil Fried Rice"],
    "Mexican":   ["Beef Tacos (3pcs)", "Chicken Burrito", "Loaded Nachos"],
    "Sri Lankan":["Rice & Curry", "Kottu Roti", "Fish Ambul Thiyal"],
}

# ─────────────────────────────────────────
# PRICE RANGES PER CUISINE AND AREA TIER
# ─────────────────────────────────────────
PRICE_TIERS = {
    "budget":   {"Indian": (20,45),  "Pakistani": (18,40), "Arabic": (22,50),
                 "Lebanese": (20,45),"Filipino": (18,38),  "Chinese": (20,42),
                 "American": (22,48),"Italian": (25,55),   "Japanese": (28,60),
                 "Thai": (20,40),    "Mexican": (22,45),   "Sri Lankan": (18,35)},
    "mid":      {"Indian": (45,90),  "Pakistani": (40,80), "Arabic": (50,100),
                 "Lebanese": (45,90),"Filipino": (38,75),  "Chinese": (42,85),
                 "American": (48,95),"Italian": (55,110),  "Japanese": (60,120),
                 "Thai": (40,80),    "Mexican": (45,85),   "Sri Lankan": (35,70)},
    "premium":  {"Indian": (90,200), "Pakistani": (80,180),"Arabic": (100,250),
                 "Lebanese": (90,200),"Filipino": (75,160),"Chinese": (85,190),
                 "American": (95,210),"Italian": (110,280),"Japanese": (120,300),
                 "Thai": (80,170),   "Mexican": (85,180),  "Sri Lankan": (70,150)},
}

AREA_PRICE_TIER = {
    # Dubai premium
    "Marina": "premium", "DIFC": "premium", "Downtown": "premium",
    "Palm Jumeirah": "premium", "Dubai Hills": "premium", "Creek Harbour": "premium",
    "Festival City": "premium",
    # Dubai mid
    "Business Bay": "mid", "JLT": "mid", "Jumeirah": "mid",
    "Al Barsha": "mid", "Mirdif": "mid", "Garhoud": "mid",
    "Jumeirah Village Circle": "mid", "Jumeirah Village Triangle": "mid",
    "Motor City": "mid", "Sports City": "mid", "Al Furjan": "mid",
    "Nad Al Sheba": "mid", "Arabian Ranches": "mid", "Oud Metha": "mid",
    # Dubai budget
    "Deira": "budget", "Silicon Oasis": "budget", "Al Nahda Dubai": "budget",
    "Discovery Gardens": "budget", "International City": "budget",
    "Al Quoz": "budget", "Bur Dubai": "budget", "Karama": "budget",
    "Satwa": "budget", "Dubai South": "budget", "Muhaisnah": "budget",
    # Abu Dhabi premium
    "Corniche": "premium", "Yas Island": "premium", "Saadiyat Island": "premium",
    "Al Raha Beach": "premium",
    # Abu Dhabi mid
    "Al Reem Island": "mid", "Khalidiyah": "mid", "Al Khalidiyah": "mid",
    "Al Mushrif": "mid", "Al Karamah": "mid", "Khalifa City": "mid",
    "Tourist Club Area": "mid", "Electra Street": "mid", "Hamdan Street": "mid",
    "Masdar City": "mid",
    # Abu Dhabi budget
    "Mohammed Bin Zayed City": "budget", "Shahama": "budget", "Al Reef": "budget",
    "Al Shamkha": "budget", "Al Muroor": "budget", "Al Zaab": "budget",
    # Sharjah and Ajman
    "Al Nahda": "budget", "Al Majaz": "budget", "Al Taawun": "budget",
    "Muwaileh": "budget", "Al Rashidiya": "budget", "Al Nuaimiya": "budget",
    "Al Jurf": "budget",
}

# ─────────────────────────────────────────
# TRAFFIC AND TIMING
# ─────────────────────────────────────────
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Very High"]
TRAFFIC_WEIGHTS = [0.25, 0.40, 0.25, 0.10]

DRIVER_VEHICLES = ["Motorcycle", "Bicycle", "Car"]
DRIVER_VEHICLE_WEIGHTS = [0.65, 0.10, 0.25]

PAYMENT_METHODS = ["Credit Card", "Cash", "Apple Pay", "Careem Pay", "Talabat Pay"]
PAYMENT_WEIGHTS = [0.35, 0.20, 0.20, 0.10, 0.15]

ORDER_STATUSES = ["Delivered", "Cancelled", "In Transit"]

# ─────────────────────────────────────────
# DATE RANGE (1 YEAR OF DATA)
# ─────────────────────────────────────────
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days

# Ramadan 2024: March 11 to April 9
RAMADAN_START = datetime(2024, 3, 11)
RAMADAN_END   = datetime(2024, 4, 9)

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_order_hour(cuisine, date):
    """Returns a realistic order hour based on cuisine and whether it is Ramadan"""
    is_ramadan = RAMADAN_START <= date <= RAMADAN_END
    is_weekend = date.weekday() in [4, 5]  # Friday and Saturday

    if is_ramadan:
        # Iftar rush 7-9pm, Suhoor orders 12am-2am, full 24 hours
        p = [0.04,0.03,0.02,0.01,0.01,0.01,
             0.01,0.02,0.02,0.02,0.02,0.02,
             0.03,0.03,0.03,0.03,0.03,0.04,
             0.08,0.12,0.12,0.10,0.08,0.06]
        p = [x / sum(p) for x in p]
        hour = np.random.choice(range(24), p=p)
    elif cuisine == "Arabic":
        p = [0.01,0.01,0.00,0.00,0.00,0.01,
             0.02,0.03,0.04,0.04,0.03,0.04,
             0.06,0.06,0.05,0.04,0.05,0.06,
             0.10,0.12,0.10,0.08,0.03,0.02]
        p = [x / sum(p) for x in p]
        hour = np.random.choice(range(24), p=p)
    else:
        # General: lunch and dinner peaks
        p = [0.01,0.01,0.00,0.00,0.00,0.01,
             0.02,0.03,0.04,0.04,0.04,0.04,
             0.07,0.08,0.06,0.05,0.05,0.05,
             0.08,0.10,0.10,0.07,0.06,0.02]
        if is_weekend:
            p = [0.01,0.01,0.01,0.00,0.00,0.01,
                 0.01,0.02,0.03,0.04,0.04,0.04,
                 0.06,0.07,0.06,0.05,0.05,0.06,
                 0.09,0.11,0.10,0.08,0.05,0.01]
        p = [x / sum(p) for x in p]
        hour = np.random.choice(range(24), p=p)
    return hour


def get_delivery_duration(distance_km, traffic, vehicle):
    """Returns delivery duration in minutes based on distance, traffic and vehicle"""
    base_speed = {"Motorcycle": 25, "Bicycle": 12, "Car": 20}
    traffic_multiplier = {"Low": 1.0, "Medium": 1.3, "High": 1.6, "Very High": 2.0}
    speed = base_speed[vehicle] * (1 / traffic_multiplier[traffic])
    base_minutes = (distance_km / speed) * 60
    prep_time = np.random.uniform(8, 20)
    noise = np.random.normal(0, 3)
    duration = max(10, base_minutes + prep_time + noise)
    return round(min(duration, 90), 1)


def get_order_status(restaurant_health, traffic, user_cancel_rate):
    """Returns order status based on restaurant health, traffic and user history"""
    cancel_prob = 0.05
    cancel_prob += (1 - restaurant_health) * 0.12
    if traffic == "Very High":
        cancel_prob += 0.06
    elif traffic == "High":
        cancel_prob += 0.03
    cancel_prob += user_cancel_rate * 0.10
    cancel_prob = min(cancel_prob, 0.35)

    in_transit_prob = 0.04
    delivered_prob = 1 - cancel_prob - in_transit_prob

    return np.random.choice(
        ["Delivered", "Cancelled", "In Transit"],
        p=[delivered_prob, cancel_prob, in_transit_prob]
    )


def get_order_quality_risk(duration, distance, traffic, vehicle, restaurant_health):
    """Returns a 0 to 1 risk score for order quality"""
    expected_duration = get_delivery_duration(distance, "Low", vehicle)
    duration_ratio = duration / max(expected_duration, 1)

    traffic_score = {"Low": 0.0, "Medium": 0.2, "High": 0.5, "Very High": 0.9}
    vehicle_score = {"Motorcycle": 0.1, "Bicycle": 0.3, "Car": 0.15}

    risk = (
        0.35 * min(duration_ratio / 2, 1.0) +
        0.25 * traffic_score[traffic] +
        0.15 * vehicle_score[vehicle] +
        0.25 * (1 - restaurant_health)
    )
    return round(min(max(risk + np.random.normal(0, 0.03), 0), 1), 3)


# ─────────────────────────────────────────
# STEP 1 - BUILD USER POOL (25,000 users)
# ─────────────────────────────────────────
print("Building user pool...")

N_USERS = 25000
cuisine_list = list(CUISINES.keys())
cuisine_probs = list(CUISINES.values())

city_list = list(CITIES.keys())
city_probs = list(CITIES.values())

users = []
for i in range(N_USERS):
    city = np.random.choice(city_list, p=city_probs)
    area = np.random.choice(AREAS[city], p=AREA_WEIGHTS[city])
    cuisine = np.random.choice(cuisine_list, p=cuisine_probs)
    payment = np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
    subscribed = np.random.choice([1, 0], p=[0.62, 0.38])
    cancel_rate = np.random.beta(1.5, 10)  # Most users rarely cancel

    users.append({
        "user_id": f"USR{str(i+1).zfill(6)}",
        "city": city,
        "area": area,
        "preferred_cuisine": cuisine,
        "preferred_payment": payment,
        "is_subscribed": subscribed,
        "base_cancel_rate": cancel_rate,
    })

users_df = pd.DataFrame(users)

# ─────────────────────────────────────────
# STEP 2 - BUILD RESTAURANT POOL (2,000 restaurants)
# ─────────────────────────────────────────
print("Building restaurant pool...")

N_RESTAURANTS = 2000
restaurants = []

for i in range(N_RESTAURANTS):
    cuisine = np.random.choice(cuisine_list, p=cuisine_probs)
    city = np.random.choice(city_list, p=city_probs)
    area = np.random.choice(AREAS[city], p=AREA_WEIGHTS[city])
    name_pool = RESTAURANT_NAMES[cuisine]
    name = random.choice(name_pool) + f" {area}"
    health_score = np.random.beta(5, 2)  # Most restaurants are decent, skewed high

    restaurants.append({
        "restaurant_id": f"RST{str(i+1).zfill(5)}",
        "restaurant_name": name,
        "cuisine": cuisine,
        "city": city,
        "area": area,
        "restaurant_health_score": round(health_score, 3),
    })

restaurants_df = pd.DataFrame(restaurants)

# ─────────────────────────────────────────
# STEP 3 - GENERATE 750,000 ORDERS
# ─────────────────────────────────────────
print("Generating 750,000 orders (this will take a few minutes)...")

N_ORDERS = 750000
BATCH_SIZE = 50000

all_orders = []

users_arr = users_df.to_dict("records")
restaurants_arr = restaurants_df.to_dict("records")

# Pre-sample user and restaurant indices for speed
user_indices = np.random.randint(0, N_USERS, N_ORDERS)
restaurant_indices = np.random.randint(0, N_RESTAURANTS, N_ORDERS)

# Pre-sample dates
random_days = np.random.randint(0, TOTAL_DAYS, N_ORDERS)
dates = [START_DATE + timedelta(days=int(d)) for d in random_days]

# Weekend orders get a 30% boost so we oversample weekends
# Already handled via date sampling, we accept natural distribution

traffic_arr = np.random.choice(TRAFFIC_LEVELS, N_ORDERS, p=TRAFFIC_WEIGHTS)
vehicle_arr = np.random.choice(DRIVER_VEHICLES, N_ORDERS, p=DRIVER_VEHICLE_WEIGHTS)
distance_arr = np.round(np.random.uniform(0.5, 15.0, N_ORDERS), 2)

driver_ids = [f"DRV{str(np.random.randint(1, 3001)).zfill(5)}" for _ in range(N_ORDERS)]

order_counter = 1

for idx in range(N_ORDERS):
    if idx % 50000 == 0:
        print(f"  Generated {idx:,} / {N_ORDERS:,} orders...")

    user = users_arr[user_indices[idx]]
    restaurant = restaurants_arr[restaurant_indices[idx]]
    date = dates[idx]
    traffic = traffic_arr[idx]
    vehicle = vehicle_arr[idx]
    distance = distance_arr[idx]

    cuisine = restaurant["cuisine"]
    hour = get_order_hour(cuisine, date)
    order_time = date.replace(hour=hour, minute=np.random.randint(0, 60))

    duration = get_delivery_duration(distance, traffic, vehicle)
    status = get_order_status(
        restaurant["restaurant_health_score"],
        traffic,
        user["base_cancel_rate"]
    )

    area = restaurant["area"]
    price_tier = AREA_PRICE_TIER.get(area, "mid")
    price_range = PRICE_TIERS[price_tier][cuisine]
    item = random.choice(MENU_ITEMS[cuisine])
    quantity = np.random.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05])
    unit_price = round(np.random.uniform(*price_range), 2)
    delivery_fee = round(np.random.choice([0, 3, 5, 7], p=[0.30, 0.25, 0.30, 0.15]), 2)
    total_price = round(unit_price * quantity + delivery_fee, 2)

    quality_risk = get_order_quality_risk(
        duration, distance, traffic, vehicle,
        restaurant["restaurant_health_score"]
    )

    driver_avail = np.random.choice(
        ["Available", "Busy", "Unavailable"],
        p=[0.65, 0.28, 0.07]
    )

    is_ramadan = 1 if RAMADAN_START <= date <= RAMADAN_END else 0
    is_weekend = 1 if date.weekday() in [4, 5] else 0

    all_orders.append({
        "order_id":                f"ORD{str(order_counter).zfill(8)}",
        "user_id":                 user["user_id"],
        "restaurant_id":           restaurant["restaurant_id"],
        "restaurant_name":         restaurant["restaurant_name"],
        "driver_id":               driver_ids[idx],
        "order_time":              order_time.strftime("%Y-%m-%d %H:%M:%S"),
        "order_date":              date.strftime("%Y-%m-%d"),
        "order_hour":              hour,
        "is_weekend":              is_weekend,
        "is_ramadan_period":       is_ramadan,
        "city":                    restaurant["city"],
        "area":                    area,
        "cuisine":                 cuisine,
        "item_name":               item,
        "quantity":                quantity,
        "unit_price_aed":          unit_price,
        "delivery_fee_aed":        delivery_fee,
        "total_price_aed":         total_price,
        "payment_method":          user["preferred_payment"],
        "delivery_distance_km":    distance,
        "traffic_level":           traffic,
        "driver_vehicle":          vehicle,
        "driver_availability":     driver_avail,
        "delivery_duration_mins":  duration,
        "order_status":            status,
        "user_subscription":       user["is_subscribed"],
        "restaurant_health_score": restaurant["restaurant_health_score"],
        "order_quality_risk_score":quality_risk,
    })

    order_counter += 1

print("Building final dataframe...")
orders_df = pd.DataFrame(all_orders)

# ─────────────────────────────────────────
# STEP 4 - ENGINEER CHURN LABEL
# ─────────────────────────────────────────
print("Engineering churn labels...")

orders_df["order_time_dt"] = pd.to_datetime(orders_df["order_time"])

# For each user, look at last 30 days vs prior 30 days order frequency
# Use a reference date of 2024-12-01 as the evaluation point
EVAL_DATE = datetime(2024, 12, 1)
WINDOW_END = EVAL_DATE
WINDOW_MID = EVAL_DATE - timedelta(days=30)
WINDOW_START = EVAL_DATE - timedelta(days=60)

recent = orders_df[(orders_df["order_time_dt"] >= WINDOW_MID) &
                   (orders_df["order_time_dt"] < WINDOW_END)]
prior  = orders_df[(orders_df["order_time_dt"] >= WINDOW_START) &
                   (orders_df["order_time_dt"] < WINDOW_MID)]

recent_counts = recent.groupby("user_id").size().rename("recent_orders")
prior_counts  = prior.groupby("user_id").size().rename("prior_orders")

# Recent cancellations
recent_cancels = (
    recent[recent["order_status"] == "Cancelled"]
    .groupby("user_id").size().rename("recent_cancels")
)

churn_df = pd.concat([recent_counts, prior_counts, recent_cancels], axis=1).fillna(0)

def churn_label(row):
    if row["prior_orders"] > 0:
        drop = (row["prior_orders"] - row["recent_orders"]) / row["prior_orders"]
        if drop >= 0.40:
            return 1
    if row["recent_cancels"] >= 2:
        return 1
    return 0

churn_df["churn_risk"] = churn_df.apply(churn_label, axis=1)
churn_map = churn_df["churn_risk"].to_dict()

orders_df["churn_risk"] = orders_df["user_id"].map(churn_map).fillna(0).astype(int)

# ─────────────────────────────────────────
# STEP 5 - FINALISE AND SAVE
# ─────────────────────────────────────────
print("Saving dataset...")

orders_df = orders_df.drop(columns=["order_time_dt"])

output_path = "/mnt/user-data/outputs/uae_food_delivery_750k.csv"
orders_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"Total rows:        {len(orders_df):,}")
print(f"Total columns:     {len(orders_df.columns)}")
print(f"Unique users:      {orders_df['user_id'].nunique():,}")
print(f"Unique restaurants:{orders_df['restaurant_id'].nunique():,}")
print(f"Unique drivers:    {orders_df['driver_id'].nunique():,}")
print(f"Date range:        {orders_df['order_date'].min()} to {orders_df['order_date'].max()}")
print(f"\nOrder status split:")
print(orders_df["order_status"].value_counts(normalize=True).round(3).to_string())
print(f"\nCuisine split:")
print(orders_df["cuisine"].value_counts(normalize=True).round(3).to_string())
print(f"\nCity split:")
print(orders_df["city"].value_counts(normalize=True).round(3).to_string())
print(f"\nChurn risk rate:   {orders_df['churn_risk'].mean():.1%}")
print(f"Avg quality risk:  {orders_df['order_quality_risk_score'].mean():.3f}")
print(f"Avg total price:   {orders_df['total_price_aed'].mean():.2f} AED")
print(f"\nSaved to: {output_path}")
