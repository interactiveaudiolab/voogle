import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import MicrophonePlugin from 'wavesurfer.js/dist/plugin/wavesurfer.microphone.js';
import testAudio from '../audio/harp.wav'

export default class VocalSearch extends React.Component {
    constructor(props) {
        super(props);
        this.waveform = React.createRef();
    }

    componentDidMount() {
        this.wavesurfer = WaveSurfer.create({
            container: this.waveform.current,
            waveColor: 'violet',
            progressColor: 'purple',
            hideScrollbar: true,
            scrollParent: true,
            cursorWidth: 0,
            plugins: [MicrophonePlugin.create({})]
        });
        this.wavesurfer.microphone.start();
    }

    render() {
        return (
            <div className='vocalsearch'>
                <div className='waveform' ref={this.waveform}/>
            </div>
        )
    }
}
