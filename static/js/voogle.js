import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import AudioFiles from './audiofiles.js';
import AWS from 'aws-sdk/global';
import CircularProgressbar from 'react-circular-progressbar';
import React from 'react';
import Recorder from './recorder.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js'
import S3 from 'aws-sdk/clients/s3';
import WavEncoder from 'wav-encoder';
import WaveSurfer from 'wavesurfer.js';
import 'react-circular-progressbar/dist/styles.css'
import '../css/voogle.css';

/* Match list for testing
{ filename: 'a.wav', textMatch: true, similarityScore: 1.0},
{ filename: 'a.wav', textMatch: false, similarityScore: 0.9},
{ filename: 'a.wav', textMatch: true, similarityScore: 0.8},
{ filename: 'a.wav', textMatch: false, similarityScore: 0.7},
{ filename: 'a.wav', textMatch: true, similarityScore: 0.6},
{ filename: 'a.wav', textMatch: false, similarityScore: 0.5},
{ filename: 'a.wav', textMatch: true, similarityScore: 0.4},
{ filename: 'a.wav', textMatch: false, similarityScore: 0.3},
{ filename: 'a.wav', textMatch: true, similarityScore: 0.2},
{ filename: 'a.wav', textMatch: false, similarityScore: 0.1},
*/

class Voogle extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            matchDivHeight: 64,
            matchDivWidth: 64,
            hasRecorded: false,
            loadedMatch: null,
            matches: [],
            playMatchText: 'Play',
            playRecordingText: 'Play',
            playingMatch: false,
            playingRecording: false,
            recordButtonText: 'Record',
            recording: false,
            recordingProgress: 0.0,
            searching: false,
            searchTime: 0,
            textInput: ''
        }

        // A handle for the periodically drawing the waveform while recording
        this.drawIntervalId = null;

        // A handle for stopping recording when the maximum recording length
        // has been reached
        this.timerAnimationId = null;
        this.recordingTimerId = null;
        // The time at which the recording timer was last initiated
        this.recordingStartTime = null;

        // A handle for search animation updates
        this.searchTimerId = null;

        // The time at which search began
        this.searchStartTime = null;

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
                IdentityPoolId: 'us-east-2:be4dd070-23b0-4a6b-ade4-99bb48caaf24'
            })
        });
        this.bucket = new S3({
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
                if (this.state.playingMatch) {
                    this.setState({
                        playingMatch: false,
                        playMatchText: 'Play'
                    });
                }

                // Reset the waveforms
                this.clearRecording();
                this.clearMatch();

                // Clear the existing matches
                this.setState({ matches: [] });

                // Start recording
                this.recorder.record();

                // Periodically draw the waveform while recording
                this.drawIntervalId = setInterval(
                    this.draw, this.props.drawingRate);

                // Update the timer animation every 100 ms
                this.recordingStartTime = (new Date()).getTime();
                this.timerAnimationId = setInterval(
                    () => {
                        let currentTime = (new Date()).getTime();
                        let elapsed = (currentTime - this.recordingStartTime) /
                            10;
                        let recordingProgress = elapsed /
                            this.props.maxRecordingLength;
                        this.setState({ recordingProgress: recordingProgress });
                    },
                    100
                );

                // Stop recording after the maximum allowed recording length
                // has been reached
                this.recordingTimerId = setTimeout(
                    () => {
                        clearInterval(this.timerAnimationId);
                        this.setState({
                            recording: false,
                            recordButtonText: 'Record'
                        })
                    },
                    this.props.maxRecordingLength * 1000
                );

            } else {
                // Stop recording
                this.recorder.stop();

                // Indicate that a query is available
                this.setState({ hasRecorded: true, recordingProgress: 0 });

                // Stop periodically drawing the waveform while recording
                clearInterval(this.drawIntervalId);

                // Stop updating the timer animation
                clearInterval(this.timerAnimationId);

                // Stop the recording timer
                clearTimeout(this.recordingTimerId);

                // Find the user's audio via level detection
                this.drawRegion();
            }
        }

        // If we start searching, being search animation
        if (this.state.searching != prevState.searching) {
            if (this.state.searching) {
                this.searchTimerId = setInterval(
                    () => {
                        let currentTime = (new Date()).getTime();
                        let elapsed = Math.floor(
                            (currentTime - this.searchStartTime) / 1000);
                        this.setState({ searchTime: elapsed });
                    },
                    1000
                );
            } else {
                clearInterval(this.searchTimerId);
                this.setState({ searchTime: 0 });
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
            color: 'rgb(36,42,54,0.25)'
        });

        // Start playback at region start
        this.recordingPlaybackStart = start;

        let region = this.wavesurfer.regions.list.queryRegion;

        // Stop playback when region bound is passed
        region.on('out', () => {
            if (this.wavesurfer.getCurrentTime() > region.end - 0.001) {
                this.wavesurfer.stop();
                this.recordingPlaybackStart = start;
                this.setState({
                    playingRecording: false,
                    playRecordingText: 'Play'
                });
            }
        });

        // Change the bounds of the query when the region is resized
        region.on('update-end', (event) => {
            let newRegion = this.wavesurfer.regions.list.queryRegion;
            this.recordingPlaybackStart = newRegion.start + 0.001;
            this.start = Math.ceil(newRegion.start * this.samplingRate);
            this.end = Math.floor(newRegion.end * this.samplingRate);
        });
    }

    getRecordingProgress = () => {
        return this.state.recordingProgress;
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

    matchesBoxContents = (recordingProgress) => {
        if (this.state.recording) {
            return (
                <div className='timer' style={{
                    width: Math.min(this.state.matchDivWidth, this.state.matchDivHeight) / 1.75,
                    paddingTop: this.state.matchDivHeight / 2 - Math.min(this.state.matchDivWidth, this.state.matchDivHeight) / 3.3}}>
                    <CircularProgressbar
                        percentage={recordingProgress}
                        strokeWidth={50}
                        styles={{
                            path: { strokeLinecap: 'butt', stroke: '#DD3C6D' },
                            text: { fill: '#E8EFF3' },
                            trail: { stroke: '#333C4D' },
                        }}
                        text={ (this.props.maxRecordingLength - Math.ceil(recordingProgress / 10)).toString() }
                    />
                </div>
            );
        } else if (this.state.searching) {
            const styles = {
                paddingLeft: this.state.matchDivWidth / 2.85,
                paddingTop: this.state.matchDivHeight / 2 - Math.min(this.state.matchDivWidth, this.state.matchDivHeight) / 10
            };
            return (
                <h2 style={styles}> Searching{'.'.repeat(this.state.searchTime % 4)} </h2>
            );
        } else if (this.state.matches) {
            return (
                <AudioFiles files={this.state.matches} loader={this.loadAudio}/>
            );
        } else {
            return null;
        }
    }

    render() {
        const recordingProgress = this.getRecordingProgress();
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
                          <li> Press&nbsp;
                            <mark className='rounded btn-all purple'>
                              &nbsp;Play&nbsp;
                            </mark>
                            &nbsp;/&nbsp;
                            <mark className='rounded btn-all purple'>
                              Pause
                            </mark>
                            &nbsp;to review your recording
                          </li>
                          <li> (Optional) Fit the region bounds to your imitation </li>
                          <li> (Optional) Enter a text description of your sound </li>
                          <li> Press <mark className='rounded btn-all blue'> &nbsp;Search&nbsp; </mark> </li>
                          <li> Click on an audio file in&nbsp;
                            <mark className='rounded btn-all purple'>
                              &nbsp;Matches&nbsp;
                            </mark>
                            &nbsp;to hear the match
                          </li>
                          <li> Press&nbsp;
                            <mark className='rounded btn-all blue'>
                              &nbsp;Download&nbsp;
                            </mark>
                            &nbsp;to download the audio file
                          </li>
                        </ol>
                    </div>
                  </div>
                  <div className="form-group form-group-lg my-4 " ref={this.resizeBottomDiv}>
                    <input type="text" className="form-control"
                      placeholder="Enter a text description of your sound (Optional)"
                      aria-describedby="inputGroup-sizing-sm"
                      value={this.state.textInput}
                      onChange={this.handleTextInput}
                      onKeyPress={this.submit}/>
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
                          {this.matchesBoxContents(recordingProgress)}
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
        const lft = this.resizeTopDiv.current.getBoundingClientRect().left;
        const rgt = this.resizeBottomDiv.current.getBoundingClientRect().right;
        this.setState({ matchDivHeight: btm - top, matchDivWidth: rgt - lft });
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

        this.setState({ searching: true });
        this.searchStartTime = (new Date()).getTime();
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
                this.setState({ matches: newMatches, searching: false });
            });
        });
    }

    togglePlayRecording = () => {
        // Event handler for the play/pause button
        this.setState(state => {
            if (!state.playingRecording &&
                !state.recording &&
                state.hasRecorded) {
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
