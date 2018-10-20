import 'bootstrap/dist/css/bootstrap.min.css';
import $ from 'jquery';
import Popper from 'popper.js';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import React from 'react';
import Recorder from './recorder.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js'
import WaveSurfer from 'wavesurfer.js';

import '../css/wavesurfer.css';

class Voogle extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            hasRecorded: false,
            playButtonText: 'Play',
            playing: false,
            recordButtonText: 'Record',
            recording: false,
            matches: null
        }

        // A handle for the periodic drawing event
        this.timerId = null;

        // Create a reference to a DOM node to place the waveform
        this.waveform = React.createRef();
    }

    componentDidMount() {
        console.log('mounted');
        // Construct the waveform display
        this.wavesurfer = WaveSurfer.create({
            container: this.waveform.current,
            cursorColor: 'black',
            hideScrollbar: true,
            pixelRatio: 1,
            plugins: [RegionsPlugin.create()],
            progressColor: 'purple',
            responsive: true,
            waveColor: 'violet',
        });

        // Reset the cursor when the audio is done playing
        this.wavesurfer.on('finish', () => {
            this.wavesurfer.stop();
            this.setState({
                playing: false,
                playButtonText: 'Play'
            });
        });

        // Grab the audio routing graph
        this.audioContext = this.wavesurfer.backend.getAudioContext();

        // Get the sampling rate at which audio processing occurs
        this.samplingRate = this.audioContext.sampleRate;

        console.log('requesting mic access');

        // Request mic access
        navigator.mediaDevices.getUserMedia({audio: true, video: false}).then(
            (stream) => {
                // Plug the user's mic into the graph
                this.audioStream = this.audioContext.createMediaStreamSource(
                    stream);

                console.log('constructing recorder');

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
        if (this.state.playing != prevState.playing) {
            if (this.state.playing) {
                this.wavesurfer.play();
            } else {
                this.wavesurfer.pause();
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
            color: 'rgb(238,130,238,0.1)'
        });
    }

    render() {
        // TODO: wavesurfer styling

        return (
            <div className='container'>
              <div className='page-header mt-3'>
                <h1>
                  Voogle
                  <small className='text-muted'>
                    &nbsp;A Vocal-Imitation Search Engine
                  </small>
                </h1>
              </div>
              <div className='jumbotron vertical-center'>
                <div className='card text-white bg-secondary mb-3'>
                <button className="btn btn-info instructions">
                  INSTRUCTIONS
                </button>
                  <div className='m-3'>
                    <ol className='big-text'>
                      <li> Press the <kbd>Start Recording</kbd> button </li>
                      <li> Try to imitate your desired sound as well as possible with your voice </li>
                      <li> Press the <kbd>Stop Recording</kbd> button </li>
                      <li> Press Play/Pause to review your recording </li>
                      <li> Enter a text description of your sound if applicable </li>
                      <li> <kbd> Search! </kbd> </li>
                    </ol>
                  </div>
                </div>
                <div className='waveform' ref={this.waveform}/>
                <div className='panel panel-default'>
                  <div className='panel-body'>
                    <button className='btn btn-lg btn-success' onClick={this.toggleRecording}>
                      {this.state.recordButtonText}
                    </button>
                    <button className='btn btn-lg btn-primary' onClick={this.togglePlayback}>
                      {this.state.playButtonText}
                    </button>
                    <button className='btn btn-lg btn-primary' onClick={this.search}>
                      Search
                    </button>
                    <button className='btn btn-lg btn-primary' onClick={this.clear}>
                      Clear
                    </button>
                  </div>
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

        fetch('/search', {
            method: 'POST',
            body: formData
        }).then(response => {
            response.text().then(text => {
                this.setState({ matches: text.split(',') });
            });
        });
    }

    togglePlayback = () => {
        // Event handler for the play/pause button
        this.setState(state => {
            if (!state.playing && !state.recording && state.hasRecorded) {
                return {
                    playing: true,
                    playButtonText: 'Pause'
                };
            } else {
                return {
                    playing: false,
                    playButtonText: 'Play'
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
