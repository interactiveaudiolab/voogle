import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import Recorder from './recorder.js';

class VocalSearch extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
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
            waveColor: 'violet',
            progressColor: 'purple',
            responsive: true
        });

        // Grab the audio routing graph
        this.audioContext = this.wavesurfer.backend.getAudioContext();

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

                // Stop drawing new audio
                clearInterval(this.timerId);
            }
        }
    }

    draw = () => {
        // Update the waveform with the new audio
        this.recorder.exportWAV((blob) => {
            this.wavesurfer.loadBlob(blob);
        });
    }

    toggleRecording = () => {
        this.setState(state => {
            return { recording: !state.recording };
        });
    }

    render() {
        return (
            <div className='vocalsearch'>
                <div className='waveform' ref={this.waveform}/>
                <button onClick={this.toggleRecording}>Record</button>
            </div>
        )
    }
}

VocalSearch.defaultProps = {
    // The time in milliseconds between waveform updates
    drawingRate: 500
};

export default VocalSearch;
