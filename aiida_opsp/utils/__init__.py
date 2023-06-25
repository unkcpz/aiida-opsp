
import hashlib
import json

def hash_dict(d: dict):
    # dict to JSON string representation
    json_str = json.dumps(d, sort_keys=True)
    
    # Hash the JSON using SHA-256
    hash_object = hashlib.sha256(json_str.encode())
    
    return hash_object.hexdigest()