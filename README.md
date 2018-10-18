# Voogle
Voogle is an audio search engine that uses vocal imitations of the desired sound as the search query.

## Installation
### Server
Voogle backend dependencies are installed with

```
pip install -r requirements.txt
```

**Note:** Windows and Linux users must have [FFmpeg](https://www.ffmpeg.org/) installed.

### Interface
Voogle frontend dependencies are installed with npm:

```
npm install
```

**Note:** You must have [Node.js](https://nodejs.org/en/) installed.

## Setup
After installing the dependencies, the Voogle app can be deployed.

### Deploying Locally
1. Start the server process by executing `python server/main.py`.
2. Start the interface by executing `npm build`.
3. Navigate to `localhost:8080` in your browser.

From there, please follow the directions on the website. Enjoy!
