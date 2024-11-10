from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# Decision tree data (X) and corresponding games (y)
X = np.array([
    [0, 0, 0, 0],  # singleplayer, first person, survival, pc
    [1, 0, 0, 0],  # multiplayer, first person, survival, pc
    [0, 1, 0, 0],  # singleplayer, third person, survival, pc
    [0, 0, 1, 0],  # singleplayer, first person, action, pc
    [0, 0, 0, 1],  # singleplayer, first person, survival, console
    [1, 1, 0, 0],  # multiplayer, third person, survival, pc
    [1, 0, 1, 0],  # multiplayer, first person, action, pc
    [1, 1, 1, 0],  # multiplayer, third person, action, pc
    [0, 1, 1, 0],  # singleplayer, third person, action, pc
    [1, 1, 0, 1],  # multiplayer, third person, survival, console
    [0, 1, 0, 1],  # singleplayer, third person, survival, console
    [1, 0, 0, 1],  # multiplayer, first person, survival, console
    [1, 0, 1, 1],  # multiplayer, first person, action, console
    [0, 0, 1, 1],  # singleplayer, first person, action, console
    [0, 1, 1, 1],   # singleplayer, third person, action, console
    [1, 1, 1, 1],   # multiplayer, third person, action, console
])

y = np.array([
    "Sons Of The Forest",  # singleplayer, first person, survival, pc
    "Rust",  # multiplayer, first person, survival, pc
    "Valheim",  # singleplayer, third person, survival, pc
    "DOOM Eternal",  # singleplayer, first person, action, pc
    "Subnautica",  # singleplayer, first person, survival, console
    "Astroneer",  # multiplayer, third person, survival, pc
    "Rainbow Six Siege",  # multiplayer, first person, action, pc
    "Gears 5",  # multiplayer, third person, action, pc
    "The Witcher 3",  # singleplayer, third person, action, pc
    "Conan Exiles",  # multiplayer, third person, survival, console
    "Horizon Zero Dawn",  # singleplayer, third person, survival, console
    "DayZ",  # multiplayer, first person, survival, console
    "Borderlands 3",  # multiplayer, first person, action, console
    "Metro Exodus",  # singleplayer, first person, action, console
    "Ghost of Tsushima",  # singleplayer, third person, action, console
    "Fortnite",  # multiplayer, third person, action, console
])

# Game cover images
cover = {
    "Sons Of The Forest": "sotf.jpg",
    "Rust": "rust.jpg",
    "Valheim": "valheim.png",
    "DOOM Eternal": "doom.jpeg",
    "Subnautica": "sub.png",
    "Astroneer": "astro.jpg",
    "Rainbow Six Siege": "siege.png",
    "Gears 5": "gears.png",
    "Conan Exiles": "conan.jpg",
    "Horizon Zero Dawn": "horizon.jpg",
    "DayZ": "dayz.jpg",
    "Borderlands 3": "bl3.jpg",
    "Metro Exodus": "metro.png",
    "Ghost of Tsushima": "tsushima.jpg",
    "The Witcher 3": "witcher3.jpg",
    "Fortnite": "fortnite.png",
}

# Train the decision tree model
clf = DecisionTreeClassifier()
clf.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Collect user answers
    answers = [
        int(request.form['question1']),
        int(request.form['question2']),
        int(request.form['question3']),
        int(request.form['question4']),
    ]
    
    # Predict the game
    predicted_game = clf.predict([answers])[0]
    
    # Display result
    return render_template('result.html', game_title=predicted_game, image_file=cover[predicted_game])

if __name__ == '__main__':
    app.run(debug=True)
