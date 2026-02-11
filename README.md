# Setup
1 - You should pip install all the requirements 

2 - To setup the server you need to download the weights from the huggingface repo. Go here: https://huggingface.co/kyutai/pocket-tts/tree/main

and download the .safetensors file and place it in the .venv/Lib/site-packages/pocket_tts/config/ directory
Next copy the skyrimnet.yaml from the weights directory and also paste it in the above config directory

You will need to have a huggingface account and need to accept the terms and conditions of the repo before you can download the file.

# Running the server
run the skyrimnet_api.py file to start the server. 

