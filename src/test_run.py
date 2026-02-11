import requests
from pathlib import Path

URL = "http://127.0.0.1:7860/"

def test_tts_to_audio():


    text = """In the frozen north of The Elder Scrolls 5: Skyrim, the land itself is breaking—civil war at its throat, old gods whispering in buried stone, and the Empire’s grip slipping in the snow.

    You are no hero—just another name on a death list—until the sky tears open and a dragon’s shadow returns to a world that swore they were gone.

    Now, with prophecy waking and kingdoms burning, one soul stands between The Empire of Tamriel and ruin… and the ancient power rising in the dark.

    This winter… the north remembers. And Sovngarde is calling you."""

    payload = {
        "text": text,
        "speaker_wav": "malecommoner",
        "language": "en",
        "accent": "",
        "save_path": "output",
        "override": False,
        "format": "wav",
    }

    resp = requests.post(URL+"tts_to_audio/", json=payload, timeout=1000)
    resp.raise_for_status()

    # data = resp.json()
    # print("Response JSON:", data)
    print("Jobs Done")

def test_create_and_store_latents():

    data = {
        "speaker_name": "malecommoner",
        "language": "en",
    }

    ROOT_DIR = Path(__file__).resolve().parents[1]
    SPEAKER_DIRECTORY = ROOT_DIR / "test_speakers"
    SAMPLE_SPEAKER = SPEAKER_DIRECTORY / "malecommoner.wav"

    with open(SAMPLE_SPEAKER, "rb") as fp:
        file = {
            "wav_file": (str(SAMPLE_SPEAKER), fp, "audio/wav"),
        }

        resp = requests.post("http://127.0.0.1:7860/create_and_store_latents/", data=data, files=file)
        resp.raise_for_status()

    print(resp.status_code)
    print(resp.text)

# test_create_and_store_latents()
test_tts_to_audio()