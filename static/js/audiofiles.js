import React, { Component } from 'react';
import '../css/audiofiles.css';

class AudioFiles extends Component {
    render() {
        const { files, loader } = this.props;
        const fileList = files.map((file) => {
            return (
                <div className='row round-box dark-gray mb-1 mx-2' onClick={() => loader(file.filename)}>
                  <div className='col-4'>
                    { file.filename.slice(file.filename.lastIndexOf('/') + 1) }
                  </div>
                  <div className='col-8 p-0'>
                    <div className='score-box' style={this.renderScore(file.similarityScore)}>
                      { this.renderTextMatch(file.textMatch) }
                    </div>
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

    renderScore = (similarityScore) => {
        const percentage = Math.round((similarityScore * 0.5 + 0.34) * 100);

        return { width: percentage.toString() + '%' };
    }

    renderTextMatch = (isMatch) => {
        if (isMatch) {
            return (<div className='text-match pr-3'>Text Match</div>)
        } else {
            return (<div>&nbsp;</div>)
        }
    }
}

export default AudioFiles;
