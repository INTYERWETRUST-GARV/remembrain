

# Remembrain

AI memory assistant for people living with dementia and elderly people 

Remembrain is a simple prototype that helps users recognize people around them. It uses a webcam to detect faces, identify known individuals, and show contextual reminders such as the person’s name, relationship, and last interaction.

The goal is to reduce confusion during social interactions and help dementia patients reconnect with people in their daily lives.

---

# Why this project exists

Dementia and Alzheimer’s disease affect millions of people worldwide. One of the most difficult moments for patients is failing to recognize someone they know.

Even a small prompt like:

"This is Rohit. He is your nephew."

can remove a lot of stress from that interaction.

Remembrain explores whether computer vision can assist memory in a simple and non-intrusive way.

---

# What the prototype does

The current prototype runs on a laptop webcam and performs the following steps:

1. Detects faces from the webcam feed
2. Generates facial embeddings using a face recognition model
3. Compares the detected face with stored known faces
4. Displays information about the recognized person
5. Optionally reads the reminder aloud using text-to-speech

Example reminder shown on screen:

Name: Rohit
Relationship: Nephew
Last seen: Yesterday at lunch

---

# How it works

Each face is converted into a numerical embedding.

An embedding is a vector representation of facial features:

f(x) = (e₁, e₂, e₃, ..., eₙ)

To identify a person, the system compares the embedding from the webcam with stored embeddings.

Similarity is measured using Euclidean distance:

d(a,b) = √ Σ (aᵢ − bᵢ)²

If the distance is below a threshold, the system assumes both faces belong to the same person.

---

# Features

• Real-time webcam face detection
• Face recognition using stored face embeddings
• Contextual reminder display
• Optional voice reminders
• Optional Google Maps powered precise place updates
• Local storage of known people
• Diagnostic tool for environment verification

---

# Project structure

```
remembrain/
|
+- main.py                 # Desktop GUI orchestrator
+- diagnostics.py          # Environment and hardware checks
+- requirements.txt        # Project dependencies
+- MVP_PLAN.md             # 4-week MVP gate and acceptance criteria
|
+- services/
|  +- recognition.py       # Recognition and fallback pipeline
|  +- memory.py            # people.json persistence and memory updates
|  +- voice.py             # Text-to-speech service
|  +- location.py          # Google Maps geolocation + reverse geocoding
|
+- faces/                  # Face images for known people
|
'- data/
   '- people.json          # Information about known individuals
```

---

# Installation

This project works best with **Python 3.10 or 3.11**.

Create a virtual environment:

```
py -3.11 -m venv .venv
```

Activate it:

```
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```
python -m pip install -r requirements.txt
```

If `dlib` fails to install, install CMake and Visual Studio Build Tools.

```
winget install Kitware.CMake
```

Then retry installing requirements.

---

# Running diagnostics

Before launching the app you can run a system check:

```
python diagnostics.py
```

This verifies:

• Python compatibility
• Dependency imports
• Webcam access
• Database integrity
• Face encoding pipeline
• Text-to-speech availability

Optional voice test:

```
python diagnostics.py --speak-test
```

---

# Running the prototype

Start the application:

```
python main.py
```

If `REMEMBRAIN_GOOGLE_MAPS_API_KEY` is configured, the app auto-refreshes the current place on startup and you can click **Use Google Maps Location** anytime.

---

# Google Maps setup (optional)

To make last-marked locations precise (instead of vague manual text), configure a Google Maps API key.

1. Create or use a Google Cloud project.
2. Enable these APIs:
  - Geolocation API
  - Geocoding API
3. Create an API key and set it in your environment.

PowerShell (current terminal session):

```
$env:REMEMBRAIN_GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"
```

PowerShell (persist for future terminals):

```
setx REMEMBRAIN_GOOGLE_MAPS_API_KEY "YOUR_API_KEY"
```

After setting the key, restart the app.



# Adding people to the system

Add a face image to the `faces` folder.

Example:


faces/rohit.jpg


Then create an entry in `data/people.json`.

Example:


{
  "rohit": {
    "name": "Rohit",
    "relationship": "Nephew",
    "last_seen": "Yesterday at lunch",
    "image": "faces/rohit.jpg"
  }
}


When the person appears in front of the camera, the system will identify them and display the reminder.



# Future ideas

This prototype is intentionally simple. Some possible improvements:

• Integration with smart glasses
• Automatic tracking of last interactions
• Emotion detection during conversations
• Secure encrypted memory storage
• Voice interaction for reminders


