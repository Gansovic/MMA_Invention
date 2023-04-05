import os
import sqlite3
import numpy as np
import feature_extraction as ft
import glob
import argparse

def save_mfcc_to_db(db_file, video_file, mfcc_features):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS mfcc_features (video_file TEXT, mfcc BLOB)''')
    cursor.execute('''INSERT INTO mfcc_features (video_file, mfcc) VALUES (?, ?)''', (video_file, sqlite3.Binary(mfcc_features.tobytes())))
    conn.commit()
    conn.close()

video_list = []
for type_ in video_types:
    files = args.training_set + '/' +  type_
    video_list.extend(glob.glob(files))

db_file = "mfcc_feature.db"
# Process each video file
for video_file in video_files:
    # Extract audio
    audio_file = os.path.splitext(video_file)[0] + '.wav'
    video_to_audio(video_file, audio_file)

    # Compute MFCC features
    mfcc_features = compute_mfcc(audio_file)

    # Save MFCC features to the database
    save_mfcc_to_db(db_file, video_file, mfcc_features)

    # Clean up the intermediate audio file
    os.remove(audio_file)

print("MFCC features saved to the database.")