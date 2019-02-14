import React, { Component } from 'react';
import '../css/voogle.css';
import download from '../images/download.png'
import downloadHover from '../images/download_hover.png'

class AudioFiles extends Component {
    constructor(props) {
        super(props)

        this.state = {

        }
    }

    playPauseIcon = (index) => {
        const icon = index === this.props.playing ? 'fa-pause' : 'fa-play';
        const onClick = index === this.props.playing ? this.props.pause : this.props.play;
        return (
            <i className={'fa ' + icon + ' pr-3 pointer hover-dark-text'} onClick={onClick}/>
        );
    }

    darkenImage = (e) => {
        e.target.setAttribute('src', downloadHover);
    }

    isMatch = (filename) => {
      if (this.props.text === "") {
          return false;
      }

      const textLower = this.props.text.toLowerCase();
      const length = textLower.length;
      const matchIndex = filename.toLowerCase().indexOf(textLower);
      return !(matchIndex === -1 || length < 3);
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
                  <div filename={file.filename} className='col text24 pl-3 light-purple-text lato400'>
                    <div>{file.displayName}</div>
                  </div>
                  <div className='col text22 pr-3 text-right'>
                    {this.isMatch(file.filename) ? <i className="pr-3 light-purple-text fas fa-star"></i> : null}
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

    renderFilename = (filename) => {
        if (this.props.text === "") {
            return <div>{filename}</div>;
        }

        const textLower = this.props.text.toLowerCase();
        const length = textLower.length;
        const matchIndex = filename.toLowerCase().indexOf(textLower);
        if (matchIndex === -1 || length < 3) {
            return <div>{filename}</div>;
        } else {
            return (
                <div>
                  {filename.substring(0, matchIndex)}
                  <div className='lato700' style={{display: 'inline'}}>
                    {filename.substring(matchIndex, matchIndex + length)}
                  </div>
                  {filename.substring(matchIndex + length)}
                </div>
            );
        }
    }
}

export default AudioFiles;
