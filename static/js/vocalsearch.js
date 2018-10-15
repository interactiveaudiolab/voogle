// TODO: get linter working
import React from 'react';
import WaveSurfer from 'wavesurfer.js';

export default class VocalSearch extends React.Component {
    constructor(props) {
        super(props);
        // TODO: fix reference issue. Need an object that accepts
        // this.container.appendChild
        this.state = {
            audioStream: this.getAudioStream(),
            wavesurfer: WaveSurfer.create({container: React.createRef()})
        };
        console.log(this.state.audioStream);
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
        // return (
        //     <div id='vocalsearch'>
        //         <h1>Hello React!</h1>
        //         <div ref={this.wavesurfer.container}></div>
        //     </div>
        // )
        console.log('rendering')
        return (
            <div className='vocalsearch'>
                Hello React!
            </div>
        )
    }
}
