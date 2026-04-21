import csv
import random

titles = [
    "Funny Cat Compilation", "Top 10 Football Goals 2023", "How to Bake a Chocolate Cake",
    "Yoga for Beginners", "Latest Tech Gadgets Review", "Travel Vlog: Paris",
    "Learn Python in 10 Minutes", "Makeup Tutorial for Beginners", "DIY Home Decor Ideas",
    "Epic Fails Compilation"
]

descriptions = [
    "A compilation of funny cat videos.", "The best football goals of the year.",
    "Step-by-step guide to baking a chocolate cake.", "A beginner's guide to yoga.",
    "Reviewing the latest tech gadgets.", "Exploring the beautiful city of Paris.",
    "Quick tutorial on Python basics.", "Makeup tips for beginners.",
    "Creative DIY home decor ideas.", "A collection of epic fail moments."
]

tags_list = [
    ["funny", "cat", "compilation"], ["football", "sports", "goals"],
    ["baking", "cake", "chocolate"], ["yoga", "fitness", "beginner"],
    ["tech", "gadgets", "review"], ["travel", "vlog", "paris"],
    ["python", "tutorial", "coding"], ["makeup", "beauty", "tutorial"],
    ["diy", "home", "decor"], ["fails", "funny", "compilation"]
]

labels_list = [
    ["Animals", "Comedy"], ["Sports"], ["Cooking"], ["Health & Fitness"],
    ["Technology"], ["Travel"], ["Education"], ["Beauty"], ["Lifestyle"], ["Comedy"]
]

with open('mini_youtube8m.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["video_id", "title", "description", "tags", "labels"])
    for i in range(200):
        idx = i % len(titles)
        video_id = f"vid_{i+1:03d}"
        writer.writerow([
            video_id,
            titles[idx],
            descriptions[idx],
            tags_list[idx],
            labels_list[idx]
        ])
