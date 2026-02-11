# Setup
1 - You need to install python 3.10, 3.11, 3.12, 3.13 or 3.14. You can do that here: https://www.python.org/downloads/

2 - Now you need to go to https://huggingface.co/kyutai/pocket-tts/tree/main and download the safetensors file (`tts_b6369a24.safetensors` at time of writing this). You will need to have a huggingface account and accept kyutai's terms of service. Save the file in the weights directory and name it `tts_skyrimnet.safetensors`. Its important to name it exactly as written here, the script requires it to have that name.

3 - You can now run the setup.bat

# Running the server
run runserver.bat to start the server


# Server features
Pocket-tts runs only on the CPU, so no need to worry about GPU versions or VRAM budgets. It also has 0 shot voice cloning despite taking very little RAM. 
