import React from 'react';
import WaveSurfer from 'wavesurfer.js';

export default class VocalSearch extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            audioStream: this.getAudioStream()
        };
        this.waveform = React.createRef();
    }

    componentDidMount() {
        this.wavesurfer = new WaveSurfer({ container: this.waveform.current });
    }

    /**
     * Requests user permission for audio recording.
     *
     * @returns {MediaStream} The user's audio stream.
     */
    getAudioStream = () => {
        const constraints = {audio: true, video: false};
        return (navigator.mediaDevices.getUserMedia(constraints)
            .catch((error) => this.handleStreamAccessError(error)));
    }

    handleStreamAccessError = (error) => {
        /* TODO */
    }

    render() {
        return (
            <div className='vocalsearch'>
                <div className='waveform' ref={this.waveform}/>
            </div>
        )
    }
}
