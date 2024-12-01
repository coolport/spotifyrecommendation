from flask import Flask, render_template, request, session, redirect, url_for
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd
import numpy as np
import re
import itertools
from createFeaturesSet import create_feature_set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from create_necessary_outputs import (
    create_necessary_outputs_function,
    generate_playlist_feature,
)
from generate_recoSystem_function import generate_playlist_recos
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from preprocess import preprocess
from filter_artists import filter_artists


app = Flask(__name__)
app.secret_key = "your_secret_key"  # Add a secret key for session management

# Spotify API credentials
SPOTIPY_CLIENT_ID = "c8cab740d73646199056df6fde510b05"
SPOTIPY_CLIENT_SECRET = "37ffc1d79eea480d8565f1da8ae51a6e"
SPOTIPY_REDIRECT_URI = "http://localhost:5000/callback"
SCOPE = "playlist-modify-public user-library-read user-read-playback-state user-read-recently-played user-read-private user-read-email"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE,
)


@app.route("/", methods=["GET"])
def index():
    if not session.get("token_info"):
        return redirect(url_for("login"))
    token_info = session.get("token_info")
    print(session.get("token_info"))
    sp = spotipy.Spotify(auth=token_info["access_token"])
    id_name = get_playlists(sp)
    session["id_name"] = id_name
    return render_template("index.html", playlists=id_name, recommendations=None)


@app.route("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    return render_template("login.html", auth_url=auth_url)


@app.route("/callback")
def callback():
    code = request.args.get("code")
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    return redirect(url_for("index"))


@app.route("/recommendations", methods=["POST"])
def recommendations():
    token_info = session.get("token_info")
    sp = spotipy.Spotify(auth=token_info["access_token"])
    id_name = session.get("id_name")

    # Retrieve the selected playlist from the form
    selected_playlist_id = request.form.get("playlist")
    selected_playlist_name = [
        name for name, id in id_name.items() if id == selected_playlist_id
    ][0]

    # Preprocessing
    spotify_df = pd.read_csv("tracks.csv")
    data_w_genre = filter_artists(
        pd.read_csv("artists.csv"),
        target_genres=[
            "classical",
            "jazz",
            "reggae",
            "rock",
            "pop",
            "electronics",
            "hip_hop",
            "hip hop",
            "rap",
        ],
    )
    spotify_df, float_cols = preprocess(spotify_df, data_w_genre)

    # Feature Extraction (check nb)
    complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)

    # extracts the selected playlist
    playlist_EDM = create_necessary_outputs_function(
        selected_playlist_name, id_name, spotify_df, sp
    )

    # Summarize playlist into a single vector (check nb)
    complete_feature_set_playlist_vector_EDM, complete_feature_set_nonplaylist_EDM = (
        generate_playlist_feature(complete_feature_set, playlist_EDM, 1.09)
    )

    # Generate Recommendation
    edm_top40 = generate_playlist_recos(
        spotify_df,
        complete_feature_set_playlist_vector_EDM,
        complete_feature_set_nonplaylist_EDM,
        sp,
    )

    # for visualizing purposes
    recommendations = edm_top40.to_dict(orient="records")

    # Check if the user wants to save the recommendations as a playlist
    save_playlist = request.form.get("save_playlist")
    if save_playlist:
        playlist_name = request.form.get("playlist_name", "Recommended Playlist")
        create_and_add_to_playlist(sp, edm_top40, playlist_name)

    # Render the main page with recommendations
    return render_template(
        "index.html", playlists=id_name, recommendations=recommendations
    )


def get_playlists(sp):
    try:
        playlists = sp.current_user_playlists()
        print(f"Playlists response: {playlists}")
        # Filter out any None items
        id_name = {
            item["name"]: item["id"] for item in playlists["items"] if item is not None
        }
        return id_name
    except Exception as e:
        print(f"Error in get_playlists: {e}")
        return {}


def create_and_add_to_playlist(sp, edm_top40, playlist_name):
    user_id = sp.me()["id"]
    playlist = sp.user_playlist_create(user_id, name=playlist_name, public=True)
    track_ids = edm_top40["id"].tolist()
    sp.user_playlist_add_tracks(user_id, playlist["id"], track_ids)


if __name__ == "__main__":
    app.run(debug=True)
