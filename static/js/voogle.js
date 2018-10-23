import 'bootstrap/dist/css/bootstrap.min.css';
import $ from 'jquery';
import AudioFiles from './audiofiles.js'
import Popper from 'popper.js';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import React from 'react';
import Recorder from './recorder.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js'
import WaveSurfer from 'wavesurfer.js';

import '../css/voogle.css';

class Voogle extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            hasLoadedMatch: false,
            hasRecorded: false,
            matches: [
              {rank: 1, filename: 'a.wav', textMatch: true},
              {rank: 1, filename: 'b.wav', textMatch: false}
            ],
            playMatchText: 'Play',
            playRecordingText: 'Play',
            playingMatch: false,
            playingRecording: false,
            recordButtonText: 'Record',
            recording: false,
            textInput: ''
        }

        // A handle for the periodic drawing event
        this.timerId = null;

        // Create references to DOM nodes to place the waveforms
        this.recordingWaveform = React.createRef();
        this.playbackWaveform = React.createRef();
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
            progressColor: '#3D7FB3',
            responsive: true,
            waveColor: '#4A99D8',
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
    }

    componentDidUpdate(prevProps, prevState) {
        // If the update was the user starting or stopping the recording,
        // send the update to the recorder.
        if (this.state.recording != prevState.recording) {
            if (this.state.recording) {
                // Clear the level-detected region on the waveform
                this.wavesurfer.clearRegions();

                // Start recording
                this.recorder.record();

                // Periodically draw the recorded waveform
                this.timerId = setInterval(this.draw, this.props.drawingRate);
            } else {
                // Stop recording
                this.recorder.stop();

                // Indicate that a query is available
                this.setState({ hasRecorded: true });

                // Stop drawing new audio
                clearInterval(this.timerId);

                // Find the user's audio via level detection
                this.drawRegion();
            }
        }

        // If the user pressed the play/pause button, signal wavesurfer to play
        // the recorded audio
        if (this.state.playingRecording != prevState.playingRecording) {
            if (this.state.playingRecording) {
                this.wavesurfer.play();
            } else {
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

        // If new matches for the target's query are available, render them
        if (this.state.matches != prevState.matches) {
            console.log(this.state.matches);
        }
    }

    clear = () => {
        // Erase the recorded audio
        this.recorder.clear();
        this.wavesurfer.empty();
        this.wavesurfer.clearRegions();
        this.setState({hasRecorded: false});
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

        // Add the region
        this.wavesurfer.addRegion({
            id: 'queryRegion',
            start: start,
            end: end,
            color: 'rgb(36,42,54,0.1)'
        });
    }

    handleTextInput = (event) => {
        this.setState({textInput: event.target.value});
    }

    loadAudio = (filename) => {
        fetch('/load', {
            method: 'POST',
            body: filename
        }).then(response => {
            // TODO: decode and load into wavesurfer
            this.setState({hasLoadedMatch: true});
        });
    }

    render() {
        return (
            <div className='container'>
              <div className='mt-4 ml-1 row'>
                <div className='col'>
                  <h1 className='text-off-white'>
                    Voogle
                    <small className='text-muted'>
                      &nbsp;&nbsp;A Vocal-Imitation Search Engine
                    </small>
                  </h1>
                </div>
              </div>
              <div className='row row-eq-height'>
                <div className='col-6 mb-2'>
                  <div className='col-md-12 mx-auto text-off-white gray rounded px-0 mt-3 h-100'>
                     <div className='card btn btn-all blue mb-2 instructions'>
                       Instructions
                     </div>
                     <ol className='big-text'>
                       <li> Press the <kbd>Record</kbd> button </li>
                       <li> Imitate your desired sound with your voice </li>
                       <li> Press the <kbd>Stop Recording</kbd> button </li>
                       <li> Press Play/Pause to review your recording </li>
                       <li> Enter a text description of your sound if applicable </li>
                       <li> <kbd> Search! </kbd> </li>
                     </ol>
                  </div>
                </div>
                <div className='col-6 mb-2'>
                  <div className='col-md-12 mx-auto text-off-white gray rounded px-0 mt-3 h-100'>
                    <div className="card btn btn-all green mb-2 instructions">
                      Matches
                    </div>
                    <div className='scrollbox mx-4 pt-1'>
                        <AudioFiles files={this.state.matches} loader={this.loadAudio}/>
                    </div>
                  </div>
                </div>
              </div>
              <div className='row'>
                <div className='col-6 mt-4 mb-1'>
                  <div className='waveform' ref={this.recordingWaveform}/>
                </div>
                <div className='col-6 mt-4 mb-1'>
                  <div className='waveform' ref={this.playbackWaveform}/>
                </div>
              </div>
              <div className='row'>
                <div className='col-6'>
                  <div className="form-group form-group-lg mb-3">
                    <span className="awesomplete mb-3">
                      <input type="text" className="form-control" placeholder="Enter Text Description of Sound (Optional)" aria-describedby="inputGroup-sizing-sm" value={this.state.textInput} onChange={this.handleTextInput}/>
                    </span>
                  </div>
                </div>
                <div className='col-6'>
                </div>
              </div>
              <div className='row'>
                <div className='col-6 btn-group'>
                  <button className='btn btn-all btn-red' onClick={this.toggleRecording}>
                    {this.state.recordButtonText}
                  </button>
                  <button className='btn btn-all btn-purple' onClick={this.togglePlayRecording}>
                    {this.state.playRecordingText}
                  </button>
                  <button className='btn btn-all btn-blue' onClick={this.search}>
                    Search
                  </button>
                  <button className='btn btn-all btn-green' onClick={this.clear}>
                    Clear
                  </button>
                </div>
                <div className='col-6 btn-group'>
                  <button className='btn btn-all btn-purple' onClick={this.togglePlayMatch}>
                    {this.state.playMatchText}
                  </button>
                  <button className='btn btn-all btn-blue' onClick={this.search}>
                    Download
                  </button>
                  <button className='btn btn-all btn-green' onClick={this.clear}>
                    Clear
                  </button>
                </div>
              </div>
            </div>
        )
    }

    search = () => {
        // Event handler for the search button
        if (this.state.hasRecorded) {
            this.recorder.exportWAV(this.sendQuery);
        }
    }

    sendQuery = (query) => {
        let start = this.wavesurfer.regions.list.queryRegion.start;
        let end = this.wavesurfer.regions.list.queryRegion.end;

        let formData = new FormData;
        formData.append('query', query);
        formData.append('start', start);
        formData.append('length', end - start);
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
                        textMatch: results.text_matches[i]
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
            if (!state.playingMatch && state.hasLoadedMatch) {
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

    // The minimum audio buffer value above which automatic region placement
    // will begin
    regionStartThreshold: 0.10,

    // The level below which the automatically placed region will end
    regionEndThreshold: 0.05,

    // The amount of time (in seconds) to add to either side
    regionTolerance: 0.25
};

export default Voogle;
