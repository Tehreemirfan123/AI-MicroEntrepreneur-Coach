# Automatic Data Generation
import pandas as pd
import random

# 30 business types
business_types = [
    "Chai Stall","Shawarma Stand","Juice Bar","Dahi Bhallay Stall","Street Clothing Vendor",
    "Small Tuition Center","Mobile Repair Shop","Bakery Stall","Pakora Stall","Ice Cream Cart",
    "Stationery Shop","Fruit Stall","Tea & Snacks Stall","Laundry Shop","Flower Stall",
    "Coffee Cart","Samosa Stall","Veggie Juice Stand","Street Photography","Mini Grocery",
    "Mobile Accessories","Repair Electronics","Fitness Trainer","Online Tutoring","Handmade Jewelry",
    "Candle Stall","Fresh Milk Seller","Local Snacks Delivery","Stationery Kiosk","Fruit Smoothie Stall"
]

# Cities
cities = ["Lahore", "Karachi", "Islamabad", "Multan", "Peshawar", "Faisalabad"]

# Marketing channels
marketing_channels = ["Flyers", "Social Media", "Word of Mouth", "Local Advertising"]

# Generate 500 entries
data_list = []
for i in range(500):
    business = random.choice(business_types)
    city = random.choice(cities)
    
    # For simplicity, set product/service same as business
    product_service = business  

    # Random numeric values
    startup_cost = random.randint(5000, 50000)
    cost_per_unit = random.randint(5, 100)
    price_per_unit = cost_per_unit + random.randint(10, 100)
    failure_risk = round(random.uniform(0.1, 0.8), 2)
    marketing = random.choice(marketing_channels)

    data_list.append({
        "Business": business,
        "City": city,
        "Product/Service": product_service,
        "Startup_Cost_PKR": startup_cost,
        "Cost_per_Unit": cost_per_unit,
        "Price_per_Unit": price_per_unit,
        "Marketing_Channel": marketing,
        "Failure_Risk": failure_risk
    })

# Create DataFrame
df = pd.DataFrame(data_list)

# Save CSV
df.to_csv("micro_business_dataset_500.csv", index=False)

print("500-entry synthetic micro-business dataset created successfully!")
