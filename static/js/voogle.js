import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import AudioFiles from './audiofiles.js'
import AWS from 'aws-sdk'
import React from 'react';
import Recorder from './recorder.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js'
import WavEncoder from 'wav-encoder';
import WaveSurfer from 'wavesurfer.js';
import '../css/voogle.css';

class Voogle extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            matchDivHeight: 64,
            hasRecorded: false,
            loadedMatch: null,
            matches: [],
            playMatchText: 'Play',
            playRecordingText: 'Play',
            playingMatch: false,
            playingRecording: false,
            recordButtonText: 'Record',
            recording: false,
            textInput: ''
        }

        // A handle for the periodically drawing the waveform while recording
        this.drawIntervalId = null;

        // A handle for stopping recording when the maximum recording length
        // has been reached
        this.recordingTimerId = null;

        // Create references to DOM nodes to place the waveforms
        this.recordingWaveform = React.createRef();
        this.playbackWaveform = React.createRef();

        // Create reference to div holding instructions and textbox in order
        // to match the height in the match div.
        this.resizeTopDiv = React.createRef();
        this.resizeBottomDiv = React.createRef();
        this.matchesBox = React.createRef();

        // The start and end sample indices of the query within the recording
        this.start = null;
        this.end = null;

        // Position to start playback in seconds
        this.recordingPlaybackStart = 0;

        // Connect to the AWS bucket storing audio files
        AWS.config.update({
            region: 'us-east-2',
            credentials: new AWS.CognitoIdentityCredentials({
                IdentityPoolId: 'us-east-2:be4dd070-23b0-4a6b-ade4-99bb48caaf24',
            })
        });
        this.bucket = new AWS.S3({
          apiVersion: '2006-03-01',
          params: {Bucket: 'voogle'}
        });
    }

    componentDidMount() {
        // Construct the waveform display
        this.wavesurfer = WaveSurfer.create({
            container: this.recordingWaveform.current,
            cursorColor: '#242A36',
            hideScrollbar: true,
            pixelRatio: 1,
            plugins: [RegionsPlugin.create()],
            progressColor: '#3D7FB3',
            responsive: true,
            waveColor: '#4A99D8',
        });

        this.matchWavesurfer = WaveSurfer.create({
            container: this.playbackWaveform.current,
            cursorColor: '#242A36',
            hideScrollbar: true,
            pixelRatio: 1,
            plugins: [RegionsPlugin.create()],
            progressColor: '#8519A1',
            responsive: true,
            waveColor: '#A51FC7',
        });

        // Reset the cursor when the audio is done playing
        this.wavesurfer.on('finish', () => {
            this.wavesurfer.stop();
            this.setState({
                playingRecording: false,
                playRecordingText: 'Play'
            });
        });

        this.matchWavesurfer.on('finish', () => {
            this.matchWavesurfer.stop();
            this.setState({
                playingMatch: false,
                playMatchText: 'Play'
            });
        });

        this.matchWavesurfer.on('ready',  () => {
            this.matchWavesurfer.play();
            this.setState({
                playingMatch: true,
                playMatchText: 'Pause'
            });
        });

        // Grab the audio routing graph
        this.audioContext = this.wavesurfer.backend.getAudioContext();

        // Get the sampling rate at which audio processing occurs
        this.samplingRate = this.audioContext.sampleRate;

        // Request mic access
        navigator.mediaDevices.getUserMedia({audio: true, video: false}).then(
            (stream) => {
                // Plug the user's mic into the graph
                this.audioStream = this.audioContext.createMediaStreamSource(
                    stream);

                // Plug mic into recorder and recorder into waveform
                this.recorder = new Recorder(
                    this.audioStream, { numChannels: 1});

            }
        ).catch(
            (error) => console.log(error)
        );

        // Update matches box size when the window is created or resized
        this.resizeMatches();
        window.addEventListener('resize', this.resizeMatches);
    }

    componentDidUpdate(prevProps, prevState) {
        // If the update was the user starting or stopping the recording,
        // send the update to the recorder.
        if (this.state.recording != prevState.recording) {
            if (this.state.recording) {
                // Stop playback
                if (this.state.playingRecording) {
                    this.setState({
                        playingRecording: false,
                        playRecordingText: 'Play'
                    })
                }

                // Reset the waveform
                this.clearRecording();

                // Start recording
                this.recorder.record();

                // Periodically draw the waveform while recording
                this.drawIntervalId = setInterval(
                    this.draw, this.props.drawingRate);

                // Stop recording after the maximum allowed recording length
                // has been reached
                this.recordingTimerId = setTimeout(
                    () => this.setState({
                        recording: false,
                        recordButtonText: 'Record'
                    }),
                    this.props.maxRecordingLength * 1000);
            } else {
                // Stop recording
                this.recorder.stop();

                // Indicate that a query is available
                this.setState({ hasRecorded: true });

                // Stop periodically drawing the waveform while recording
                clearInterval(this.drawIntervalId);

                // Stop the recording timer
                clearTimeout(this.recordingTimerId);

                // Find the user's audio via level detection
                this.drawRegion();
            }
        }

        // If the user pressed the play/pause button, signal wavesurfer to play
        // the recorded audio
        if (this.state.playingRecording != prevState.playingRecording) {
            if (this.state.playingRecording) {
                this.wavesurfer.play(this.recordingPlaybackStart);
            } else {
                let currentTime = this.wavesurfer.getCurrentTime();
                if (currentTime > this.recordingPlaybackStart) {
                    this.recordingPlaybackStart = currentTime;
                }
                this.wavesurfer.pause();
            }
        }

        // If the user pressed the play/pause button, signal wavesurfer to play
        // the recorded audio
        if (this.state.playingMatch != prevState.playingMatch) {
            if (this.state.playingMatch) {
                this.matchWavesurfer.play();
            } else {
                this.matchWavesurfer.pause();
            }
        }
    }

    clearRecording = () => {
        // Erase the recorded audio
        this.queryBuffer = null;
        this.recorder.clear();
        this.wavesurfer.empty();
        this.wavesurfer.clearRegions();
        this.setState({
            hasRecorded: false,
            playingRecording: false,
            playRecordingText: 'Play'
        });
    }

    clearMatch = () => {
        this.matchWavesurfer.empty();
        this.matchWavesurfer.clearRegions();
        this.setState({
            loadedMatch: null,
            playingMatch: false,
            playMatchText: 'Play'
        });
    }

    download = () => {
        // Don't download if we have no audio loaded
        if (!this.state.loadedMatch) {
            return;
        }

        // Encode the audio as a WAV file
        WavEncoder.encode({
            sampleRate: this.matchWavesurfer.backend.ac.sampleRate,
            channelData: [this.matchWavesurfer.backend.buffer.getChannelData(0)]
        }).then((buffer) => {
            let blob = new Blob([buffer], {type: 'audio/wav'});
            let filename = this.state.loadedMatch.slice(
                this.state.loadedMatch.lastIndexOf('/') + 1);

            // Download hack: create a ghost element with a download link and
            // click it
            let link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = filename;
            link.click();
        });
    }

    draw = () => {
        // Update the waveform with the new audio
        this.recorder.exportWAV((blob) => {
            this.wavesurfer.loadBlob(blob);
        });
    }

    drawRegion = () => {
        // Grab the audio buffer
        let buffer = this.wavesurfer.backend.buffer.getChannelData(0);

        // Find the first location at which the audio exceeds the threshold
        // level
        let start = buffer.findIndex((x) => {
            return Math.abs(x) > this.props.regionStartThreshold;
        });

        // Find the last location at which the audio exceeds the threshold level
        let end = buffer.length - buffer.reverse().findIndex((x) => {
            return Math.abs(x) > this.props.regionEndThreshold;
        });

        // This is the actual array buffer--not a copy. Undo our reversal.
        buffer.reverse();

        // If audio never exceeded either threshold, set the entire buffer as
        // the region
        if (start == -1 || end == -1) {
            start = 0;
            end = buffer.length;
        }

        // Convert to seconds and grab the surrounding audio
        start = start / this.samplingRate - this.props.regionTolerance;
        end = end / this.samplingRate + this.props.regionTolerance;

        // Clip the audio to the bounds of the buffer
        start = Math.max(0, start);
        end = Math.min(this.wavesurfer.getDuration(), end);

        // Save buffer indices for sending query
        this.start = start * this.samplingRate;
        this.end = end * this.samplingRate;

        // Add the region
        this.wavesurfer.addRegion({
            id: 'queryRegion',
            start: start,
            end: end,
            color: 'rgb(36,42,54,0.1)'
        });

        // Start playback at region start
        this.recordingPlaybackStart = start;

        let region = this.wavesurfer.regions.list.queryRegion;

        // Stop playback when region bound is passed
        region.on('out', () => {
            this.wavesurfer.stop();
            this.recordingPlaybackStart = start;
            this.setState({
                playingRecording: false,
                playRecordingText: 'Play'
            });
        });

        // Change the bounds of the query when the region is resized
        region.on('update-end', () => {
            this.start = region.start * this.samplingRate;
            this.end = region.end * this.samplingRate;
        });
    }

    handleTextInput = (event) => {
        this.setState({textInput: event.target.value});
    }

    loadAudio = (key) => {
        // Don't retrieve the audio if we already have it
        if (key === this.state.loadedMatch) {
            this.matchWavesurfer.seekTo(0);
            this.setState({playingMatch: true, playMatchText: 'Pause'});
            return;
        }

        // Grab the file from the S3 instance
        this.bucket.getSignedUrl('getObject', {Key: key}, (err, url) => {
            if (err) {
                console.log(err);
            } else {
                this.matchWavesurfer.load(url);
                this.setState({loadedMatch: key});
            }
        });
    }

    render() {
        return (
            <div className='container'>
              <div className='mt-4 ml-1'>
                <h1 className='text-off-white'>
                  Voogle
                  <small className='text-muted'>
                    &nbsp;&nbsp;A Vocal-Imitation Search Engine
                  </small>
                </h1>
              </div>
              <div className='row'>
                <div className='col-md-6 mb-3'>
                  <div className='text-off-white gray rounded px-0 pb-1 my-4'>
                    <div className='card btn btn-all blue mb-2 instructions' ref={this.matchesBox}>
                      Instructions
                    </div>
                    <div ref={this.resizeTopDiv}>
                        <ol className='big-text'>
                          <li> Press <mark className='rounded btn-all red'> &nbsp;Record&nbsp; </mark> </li>
                          <li> Imitate your desired sound with your voice </li>
                          <li> Press <mark className='rounded btn-all red'> &nbsp;Stop Recording&nbsp; </mark> </li>
                          <li> Press <mark className='rounded btn-all purple'> &nbsp;Play&nbsp; </mark>/<mark className='rounded btn-all purple'>Pause</mark> to review your recording </li>
                          <li> (Optional) Fit the region bounds to your imitation </li>
                          <li> (Optional) Enter a text description of your sound </li>
                          <li> Press <mark className='rounded btn-all blue'> &nbsp;Search&nbsp; </mark> </li>
                          <li> Click on an audio file in <mark className='rounded btn-all purple'> &nbsp;Matches&nbsp;</mark> to hear the match </li>
                          <li> Press <mark className='rounded btn-all blue'> &nbsp;Download&nbsp;</mark> to download the audio file </li>
                        </ol>
                    </div>
                  </div>
                  <div className="form-group form-group-lg my-4 " ref={this.resizeBottomDiv}>
                    <input type="text" className="form-control" placeholder="Enter a text description of your sound (Optional)" aria-describedby="inputGroup-sizing-sm" value={this.state.textInput} onChange={this.handleTextInput} onKeyPress={this.submit}/>
                  </div>
                  <div className='my-4'>
                    <div className='waveform' ref={this.recordingWaveform}/>
                  </div>
                  <div className='btn-group w-100'>
                    <button className='btn btn-all btn-red' onClick={this.toggleRecording}>
                      {this.state.recordButtonText}
                    </button>
                    <button className='btn btn-all btn-purple' onClick={this.togglePlayRecording}>
                      {this.state.playRecordingText}
                    </button>
                    <button className='btn btn-all btn-blue' onClick={this.search}>
                      Search
                    </button>
                    <button className='btn btn-all btn-green' onClick={this.clearRecording}>
                      Clear
                    </button>
                  </div>
                </div>
                <div className='col-md-6 mb-2'>
                  <div className='text-off-white gray rounded px-0 my-4'>
                    <div className="card btn btn-all purple mb-2 instructions">
                      Matches
                    </div>
                      <div className='scrollbox m-2' style={{height: this.state.matchDivHeight}}>
                        <div className='pb-2 pt-1'>
                          <AudioFiles files={this.state.matches} loader={this.loadAudio}/>
                      </div>
                    </div>
                  </div>
                  <div className='my-4'>
                    <div className='waveform' ref={this.playbackWaveform}/>
                  </div>
                  <div className='btn-group w-100'>
                    <button className='btn btn-all btn-purple' onClick={this.togglePlayMatch}>
                      {this.state.playMatchText}
                    </button>
                    <button className='btn btn-all btn-blue' onClick={this.download}>
                      Download
                    </button>
                    <button className='btn btn-all btn-green' onClick={this.clearMatch}>
                      Clear
                    </button>
                  </div>
                </div>
              </div>
            </div>
        )
    }

    resizeMatches = () => {
        const top = this.resizeTopDiv.current.getBoundingClientRect().top;
        const btm = this.resizeBottomDiv.current.getBoundingClientRect().bottom;
        this.setState({matchDivHeight: btm - top});
    }

    search = () => {
        // Event handler for the search button
        if (this.state.hasRecorded) {
            // Get the full recording
            let buffer = this.wavesurfer.backend.buffer.getChannelData(0);

            // Grab the segment containing the query
            let query = buffer.slice(this.start, this.end)

            // Send the underlying data as a bytestream
            this.sendQuery(new Blob([query.buffer]));
        }
    }

    submit = (event) => {
        if (event.key == 'Enter') {
            this.search();
        }
    }

    sendQuery = (query) => {
        // Don't send search request if no recording exists
        if (!this.state.hasRecorded) {
            return;
        }

        let formData = new FormData;
        formData.append('query', query);
        formData.append('sampling_rate', this.samplingRate);
        formData.append('text_input', this.state.textInput);

        fetch('/search', {
            method: 'POST',
            body: formData
        }).then(response => {
            response.json().then(results => {
                let newMatches = [];
                for (let i = 0; i < results.matches.length; i++) {
                    newMatches.push({
                        rank: i,
                        filename: results.matches[i],
                        textMatch: results.text_matches[i],
                        similarityScore: results.similarity_scores[i]
                    })
                }
                this.setState({ matches: newMatches });
            });
        });
    }

    togglePlayRecording = () => {
        // Event handler for the play/pause button
        this.setState(state => {
            if (!state.playingRecording && !state.recording && state.hasRecorded) {
                return {
                    playingRecording: true,
                    playRecordingText: 'Pause'
                };
            } else {
                return {
                    playingRecording: false,
                    playRecordingText: 'Play'
                }
            }
        });
    }

    togglePlayMatch = () => {
        // Event handler for the play/pause button
        this.setState(state => {
            if (!state.playingMatch && state.loadedMatch) {
                return {
                    playingMatch: true,
                    playMatchText: 'Pause'
                };
            } else {
                return {
                    playingMatch: false,
                    playMatchText: 'Play'
                }
            }
        });
    }

    toggleRecording = () => {
        // Event handler for the recording button
        this.setState(state => {
            if (state.recording) {
                return {
                    recording: false,
                    recordButtonText: 'Record'
                };
            } else {
                return {
                    recording: true,
                    recordButtonText: 'Stop Recording'
                }
            }
        });
    }
}

Voogle.defaultProps = {
    // The time (in milliseconds) between waveform updates
    drawingRate: 500,

    // The maximum duration (in seconds) of a user's recording
    maxRecordingLength: 10,

    // The minimum audio buffer value above which automatic region placement
    // will begin
    regionStartThreshold: 0.10,

    // The level below which the automatically placed region will end
    regionEndThreshold: 0.05,

    // The amount of time (in seconds) to add to either side
    regionTolerance: 0.25
};

export default Voogle;