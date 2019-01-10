import React, { Component } from 'react';
import '../css/voogle.css';
import download from '../images/download.png'
import downloadHover from '../images/download_hover.png'

class AudioFiles extends Component {

    playPauseIcon = (index) => {
        console.log(index, this.props.playing)
        const icon = index === this.props.playing ? 'fa-pause' : 'fa-play';
        const onClick = index === this.props.playing ? this.props.pause : this.props.play;
        return (
            <i className={'fa ' + icon + ' pr-3 pointer hover-dark-text'} onClick={onClick}/>
        );
    }

    darkenImage = (e) => {
        e.target.setAttribute('src', downloadHover);
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
                  <div className='col text24 pl-3 light-purple-text lato-400'>
                    { file.filename }
                  </div>
                  <div className='col text24 pr-3 text-right'>
                    {this.playPauseIcon(index)}
                    <img
                      className='pr-1 pb-1 pointer'
                      src={download}
                      style={{width: '30px'}}
                      onClick={this.props.download}
                      onMouseOver={(e) => e.currentTarget.src = downloadHover}
                      onMouseOut={(e) => e.currentTarget.src = download}
                    />
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
