import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import Microphone from 'wavesurfer.js/dist/plugin/wavesurfer.microphone.js';
import testAudio from '../audio/harp.wav';

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
            cursorWidth: 0
        });
        this.wavesurfer.load(testAudio);
        // TODO: load an audio file to test waveform
        this.microphone = new Microphone({ wavesurfer: this.wavesurfer });
        this.microphone.start();
    }

    render() {
        return (
            <div className='vocalsearch'>
                <div className='waveform' ref={this.waveform}/>
            </div>
        )
    }
}
