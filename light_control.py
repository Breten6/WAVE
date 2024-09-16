import requests
import json

url = "http://192.168.0.198/api/p1csEbKFRYbAqmqSNNDs93MaLvZv6OXi6DhXBjW8/lights/1/state"

light_on = {"on":True,"bri":1}
light_off = {"on":False}

# r = requests.put(url, json.dumps(light_off), timeout=5)

try:
    r = requests.put(url, json.dumps(light_off), timeout=5)
    r.raise_for_status()  # Raise an exception for HTTP errors
    print("Request successful")
except requests.exceptions.RequestException as e:
    print("Request failed:", e)

# p1csEbKFRYbAqmqSNNDs93MaLvZv6OXi6DhXBjW8 #
