import json
import random

user_behavior = {}
for i in range(1, 6):
    user_id = f"user_{i}"
    watched_videos = [f"vid_{random.randint(1, 200):03d}" for _ in range(3)]
    preferred_tags = random.sample(["funny", "cat", "compilation", "football", "sports", "goals",
                                    "baking", "cake", "chocolate", "yoga", "fitness", "beginner",
                                    "tech", "gadgets", "review", "travel", "vlog", "paris",
                                    "python", "tutorial", "coding", "makeup", "beauty",
                                    "diy", "home", "decor", "fails"], 3)
    user_behavior[user_id] = {
        "watched_videos": watched_videos,
        "preferred_tags": preferred_tags
    }

with open('user_behavior.json', 'w', encoding='utf-8') as f:
    json.dump(user_behavior, f, indent=2)
