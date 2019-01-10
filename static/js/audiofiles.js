import React, { Component } from 'react';
import '../css/voogle.css';

class AudioFiles extends Component {
    playPauseIcon = (index) => {
        console.log(index, this.props.playing)
        const icon = index === this.props.playing ? 'fa-pause' : 'fa-play';
        const onClick = index === this.props.playing ? this.props.pause : this.props.play;
        return <i className={'fa ' + icon + ' pr-3 pointer'} onClick={onClick}/>
    }

    render() {
        const fileList = this.props.files.map((file, index) => {
            const color = index % 2 ? 'search-dark' : 'search-light';
            return (
                <div
                  className={'row justify-content-between align-items-center d-flex m-0 ' + color}
                  key={index}
                  data-key={index}
                  style={{height: this.props.height}}
                >
                  <div className='col text24 light-purple-text pl-3 lato-400'>
                    { file.filename }
                  </div>
                  <div className='col text24 light-purple-text pr-3 text-right'>
                    {this.playPauseIcon(index)}
                    <i className='fa fa-download pr-1 pointer' onClick={this.props.download}/>
                  </div>
                </div>
            )
        });
        return (
            <div className='audiofiles'>
              { fileList }
            </div>
        )
    }
}

export default AudioFiles;
