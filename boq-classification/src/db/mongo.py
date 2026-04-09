from pymongo import MongoClient

def get_collection():
    client = MongoClient("mongodb://localhost:27017/")  # OK fixed
    db = client["boq_database"]
    return db["boq_results"]

def save_to_mongo(data):
    collection = get_collection()
    result = collection.insert_one(data)
    print(f"[OK] Saved to MongoDB -> ID: {result.inserted_id}")