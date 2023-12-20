import googleapiclient.discovery
import pickle
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import matplotlib.pyplot as plt

MAX_LEN = 250
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16


def encode_text(text,tokenizer):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [tokenizer.word_index[word] if word in tokenizer.word_index and tokenizer.word_index[word] < 1000 else 0 for word in tokens]
    return pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

def load_model():
    recent_path = os.getcwd()
    file_name = "sentiment_analysis_model.h5"
    if os.path.exists(os.path.join(recent_path,file_name)):
        new_model = tf.keras.models.load_model(os.path.join(recent_path,file_name))

    tokenizer_name = "tokenizer.pickle"
    with open(os.path.join(recent_path,tokenizer_name),'rb') as file:
        tokenizer = pickle.load(file)

    return (new_model,tokenizer)

def find_youtube_id(api_key,channel_name):
    youtube = googleapiclient.discovery.build("youtube","v3",developerKey = api_key)

    search_response = youtube.search().list(

        q = channel_name,
        part = 'id',
        type = 'channel'
    ).execute()

    if 'items' in search_response:
        channel_id = search_response['items'][0]['id']['channelId']
        return channel_id
    else:
        print(f"Channel with name '{channel_name}' not found.")
        return None

def get_latest_videos(api_key, channel_id, max_results=10):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    # Get the playlist ID of the uploads playlist for the channel
    channels_response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Get the latest videos from the uploads playlist
    playlist_items_response = youtube.playlistItems().list(
        part="snippet",
        playlistId=playlist_id,
        maxResults=max_results
    ).execute()

    videos = []
    for item in playlist_items_response["items"]:
        video = {
            "title": item["snippet"]["title"],
            "video_id": item["snippet"]["resourceId"]["videoId"],
            "published_at": item["snippet"]["publishedAt"]
        }
        videos.append(video)

    return videos

def get_video_comments(api_key, video_id, max_results=1000):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# Get comments for the specified video
    comments_response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        order="relevance",  # You can change the order if needed
        maxResults=max_results
    ).execute()

    comments = []
    for item in comments_response["items"]:
        comment = {
            "text": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
        }
        comments.append(comment)

    return comments

# Replace with your API key and channel ID
api_key = "AIzaSyD25q0HUw7sAa6xnc1mf-fVp0wlAMRAaIQ"

channel = input("Introduce el nombe del canal: \n")
channel_id = find_youtube_id(api_key,  channel)

latest_videos = get_latest_videos(api_key, channel_id)
(model,tokenizer) = load_model()
Information = []
review = [0] * 3
for video in latest_videos:
    print(f"Title: {video['title']}")
    print(f"Video ID: {video['video_id']}")
    ## Top 1000 first comments
    try:
        comments_sections = get_video_comments(api_key,video['video_id'])
    except:
        continue

    positiveCounter = 0
    negativeCounter = 0
    neuterCounter = 0
    
    for comment in comments_sections:
        ## Tokenize and clean the data
        data = encode_text(comment['text'],tokenizer)
        prediction = np.argmax(model.predict(data))

        if prediction == 0:
            negativeCounter+=1
        elif prediction == 1:
            neuterCounter+=1
        else:
            positiveCounter+=1
    
    total = positiveCounter + negativeCounter + neuterCounter

    PositivePercentage = (positiveCounter/total) * 100
    NegativePercentage = (negativeCounter/total) * 100
    NeuterPercentage = (neuterCounter/total) * 100

    review[0] += PositivePercentage
    review[1] += NegativePercentage
    review[2] += NeuterPercentage

    videoInformation = {
        "Title": video['title'],
        "positivePercentage" : PositivePercentage,
        "negativePercentage" : NegativePercentage,
        "neuterPercentage" : NeuterPercentage
    }
    Information.append(videoInformation)

prediction = np.argmax(review)

if prediction == 0:
    print("positive channel")
elif prediction == 1:
    print("negative channel")
else:
    print("neuter channel")


positive_bar = []
negative_bar = []
neuter_bar = []
Categories = [] 
for video_info in Information:
    Categories.append(video_info["Title"])
    positive_bar.append(video_info["positivePercentage"])
    negative_bar.append(video_info["negativePercentage"])
    neuter_bar.append(video_info["neuterPercentage"])

plt.rc('font', weight='bold')
 
# Heights of bars1 + bars2
bars = np.add(positive_bar, negative_bar).tolist()
 
# The position of the bars on the x-axis
r = [i+5 for i,v in enumerate(Categories)]
 
# Names of group and bar width

barWidth = 1
 
plt.bar(r, positive_bar, color='yellow', edgecolor='white', width=barWidth)
plt.bar(r, negative_bar, bottom=positive_bar, color='black', edgecolor='white', width=barWidth)
plt.bar(r, neuter_bar, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)
 
# Custom X axis
plt.xticks(r, Categories, fontweight='bold', rotation = 45)
plt.xlabel("group")
# Show graphic
plt.show()