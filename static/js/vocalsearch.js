import React from 'react';
import WaveSurfer from 'wavesurfer.js';
/* TODO: turn into a React component */

export class VocalSearch extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            audioStream: this.getAudioStream(),
        };
        console.log(this.state.audioStream);
    }

    /**
     * Requests user permission for audio recording.
     *
     * @returns {MediaStream} The user's audio stream.
     */
    getAudioStream() {
        const constraints = {audio: true, video: false};
        return (navigator.mediaDevices.getUserMedia(constraints)
            .catch((error) => this.handleStreamAccessError(error)));
    }

    handleStreamAccessError(error) {
        /* TODO */
    }

    render() {
        return (
            <div id='vocalsearch'>
                {/*<div id='waveform'>
                    <Waveform
                        query={this.state.query}
                    />
                </div>
                <button id='record_button'>
                    <Recorder
                        recording={this.state.recording}
                        onClick={this.toggleRecord}
                    />
                </button>*/}
            </div>
        )
    }
}
