import os
import django

# Set up Django settings before importing models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")  # Change 'myproject' to your actual project name
django.setup()

import pandas as pd
from accounts.models import UserSongInteraction
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import pickle

# Function to fetch user listening data
def get_music_data():
    interactions = UserSongInteraction.objects.all()
    data = []
    
    for interaction in interactions:
        data.append([interaction.user.id, interaction.song.id])
    
    df = pd.DataFrame(data, columns=["user_id", "song_id"])
    return df

# Train the AI model
df = get_music_data()
reader = Reader(rating_scale=(1, 5))  # Define a rating scale (fake rating)
data = Dataset.load_from_df(df[['user_id', 'song_id']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# Save trained AI model
with open("music_recommendation_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ AI Model Training Completed & Saved!")
