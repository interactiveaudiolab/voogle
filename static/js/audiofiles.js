import React, { Component } from 'react';
import '../css/audiofiles.css';

class AudioFiles extends Component {
    render() {
        const { files, loader } = this.props;
        const fileList = files.map((file) => {
            return (
                <div className='row round-box dark-gray mb-1' onClick={() => loader(file.filename)}>
                  <div className='col-9'>
                    { file.filename.slice(file.filename.lastIndexOf('/') + 1) }
                  </div>
                  { this.renderTextMatch(file.textMatch) }
                </div>
            )
        });
        return (
            <div className='audiofiles'>
              { fileList }
            </div>
        )
    }

    renderTextMatch = (isMatch) => {
        if (isMatch) {
            return (<div className='col-3 green rounded text-center'> Text Match </div>)
        } else {
            return (<div className='col-3'/>)
        }
    }
}

export default AudioFiles;
