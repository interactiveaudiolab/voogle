import React from 'react';
import Recorder from './recorder.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js'
import WaveSurfer from 'wavesurfer.js';

import waveStyle from '../css/wavesurfer.css';

class Voogle extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            hasRecorded: false,
            playButtonText: 'Play',
            playing: false,
            recordButtonText: 'Record',
            recording: false,
        }

        // A handle for the periodic drawing event
        this.timerId = null;

        // Create a reference to a DOM node to place the waveform
        this.waveform = React.createRef();
    }

    componentDidMount() {
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
            color: 'purple'
        });
    }

    render() {
        return (
            <div className='Voogle'>
                <div className={waveStyle.waveform} ref={this.waveform}/>
                <button onClick={this.toggleRecording}>
                    {this.state.recordButtonText}
                </button>
                <button onClick={this.togglePlayback}>
                    {this.state.playButtonText}
                </button>
                <button onClick={this.search}>
                    Search
                </button>
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
        formData.append('end', end);
        formData.append('sampling_rate', this.samplingRate);

        fetch('/search', {
            method: 'POST',
            body: formData
        }).then(matches => this.setState({ matches: matches }));
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
