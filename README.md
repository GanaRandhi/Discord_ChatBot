# Discord_ChatBot
Creating a Personal Chatbot for Discord to test the basic chatbot skills for a specific category.

## Prerequisites
Install dependencies present in requirements.txt <br>
``` Python
python -m pip install -r requirements.txt
```
 *Also install if any missing*
## Getting started
please update your config files <br>
"ibot" to run llama2 <br>
```
    "cogs": [
        "ibot"
    ]
```
"ibotGemini" to run Google AI <br>
```
    "cogs": [
        "ibotGemini"
    ]
```

Open your CMD and type 
```cmd
ollama
```
(after installing Ollama setup)<br>
then type
``` CMD
ollama run llama2
```
## Run the script
``` Python
python bot.py
```
