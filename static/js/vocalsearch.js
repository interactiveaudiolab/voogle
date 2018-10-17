import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import Recorder from './recorder.js';

class VocalSearch extends React.Component {
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
                this.recorder.record();

                // Periodically draw the recorded waveform
                this.timerId = setInterval(this.draw, this.props.drawingRate);
            } else {
                this.recorder.stop();
                this.setState({ hasRecorded: true });

                // Stop drawing new audio
                clearInterval(this.timerId);
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

    render() {
        return (
            <div className='vocalsearch'>
                <div className='waveform' ref={this.waveform}/>
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
            // TODO: clip audio to region
            this.recorder.exportWAV(this.sendQuery);
        }
    }

    sendQuery = (query) => {
        let formData = new FormData;
        formData.append('query', query);
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

VocalSearch.defaultProps = {
    // The time in milliseconds between waveform updates
    drawingRate: 500
};

export default VocalSearch;
