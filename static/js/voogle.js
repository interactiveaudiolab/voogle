import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import AudioFiles from './audiofiles.js';
import CircularProgressbar from 'react-circular-progressbar';
import React from 'react';
import Recorder from './recorder.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js'
import WavEncoder from 'wav-encoder';
import WaveSurfer from 'wavesurfer.js';
import logo from '../images/logo.png'
import 'react-circular-progressbar/dist/styles.css'
import '../css/voogle.css';


class Voogle extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            hasRecorded: false,
            instructions: false,
            matches: [],
            matchesHeight: 0,
            playing: null,
            recording: false,
            recordingProgress: 0.0,
            searchHeight: 0,
            searching: false,
            searchTime: 0,
            textInput: ''
        }

        // A handle for stopping recording when the maximum recording length
        // has been reached
        this.timerAnimationId = null;
        this.recordingTimerId = null;

        // The time at which the recording timer was last initiated
        this.recordingStartTime = null;

        // A handle for search animation updates
        this.searchTimerId = null;

        // The time at which search began
        this.searchStartTime = null;

        // The start and end sample indices of the query within the recording
        this.start = null;
        this.end = null;

        // React div references
        this.audioRef = React.createRef();
        this.footerRef = React.createRef();
        this.headerRef = React.createRef();
        this.searchRef = React.createRef();
    }

    componentDidMount() {
        this.updateSearchHeight();
        window.addEventListener('resize', this.updateSearchHeight);
    }

    componentDidUpdate(prevProps, prevState) {
        if (this.state.recording != prevState.recording) {
            if (this.state.recording) {
                this.record();
            } else {
                this.stop();
            }
        }
    }

    componentWillMount() {
        document.body.style.backgroundColor = '#1C142D';
    }

    componentWillUnmount() {
        document.body.style.backgroundColor = null;
        window.removeEventListener('resize', this.updateSearchHeight);
    }

    clearRecording = () => {
        // Erase the recorded audio
        if (this.recorder) {
            this.recorder.clear();
        }
        this.queryBuffer = null;
        this.setState({hasRecorded: false});
    }

    download = (event) => {
        const key = event.target.parentNode.parentNode.firstChild.getAttribute('filename');

        let formData = new FormData;
        formData.append('filename', key);

        fetch('/retrieve', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                response.blob().then(blob => {
                    // Download hack: create a ghost element with a download
                    // link and click it
                    let link = document.createElement('a');
                    link.href = window.URL.createObjectURL(blob);
                    link.download = key;
                    link.click();
                });
            } else {
                console.log('Audio file ${key} could not be found');
            }
        });
    }

    handleTextInput = (event) => {
        this.setState({textInput: event.target.value});
    }

    instructions = () => {
        if (this.state.instructions) {
            return (
                <div className='instruction-overlay'>
                  <div className='instruction-box px-5 pb-5 pt-4 d-flex align-items-center'>
                    <div className='close-inst light-text pl-3 text26'>
                      <i className='fa fa-times' onClick={this.toggleInstructions}/>
                    </div>
                    <div>
                      <p className='m-0 mb-2 lato400 text40 text-center light-text'>
                        Welcome to Voogle!
                      </p>
                      <p className='m-0 mb-2 lato-300 text24 light-text'>
                        Voogle is a search engine that uses audio to search for audio! To use Voogle, click the big purple button and make a sound. Voogle will present you with similar sounds, which you can preview and download.
                      </p>
                      <p className='m-0 lato-300 text24 light-text'>
                        You can narrow your search results by providing a text description of the sound you're looking for. To do this, type your description in the bottom box before Voogling.
                      </p>
                    </div>
                  </div>
                </div>
            );
        } else {
            return null;
        }
    }

    levelDetect = (buffer) => {
        // Take absolute value of each sample for level detection
        let max = 0.0;
        for (let i = 0; i < buffer.length; i++) {
            buffer[i] = Math.abs(buffer[i]);

            // Store maximum sample value
            if (buffer[i] > max) {
                max = buffer[i];
            }
        }

        // Normalize the buffer
        for (let i = 0; i < buffer.length; i++) {
            buffer[i] /= max;
        }

        // Find the first location at which the audio exceeds the threshold
        // level
        let start = buffer.findIndex((x) => {
            return x > this.props.regionStartThreshold;
        });

        // Find the last location at which the audio exceeds the threshold level
        let end = buffer.length - buffer.slice().reverse().findIndex((x) => {
            return x > this.props.regionEndThreshold;
        });

        // If audio never exceeded either threshold, set the entire buffer as
        // the region
        if (start == -1 || end == -1) {
            start = 0;
            end = buffer.length;
        }

        // Convert to seconds and grab the surrounding audio
        start = start / this.samplingRate - this.props.regionStartTolerance;
        end = end / this.samplingRate + this.props.regionEndTolerance;

        // Clip the audio to the bounds of the buffer
        start = Math.max(0, start) * this.samplingRate;
        end = Math.min(buffer.length, end) * this.samplingRate;

        return buffer.slice(start, end);
    }

    makeRecorder = () => {
        window.AudioContext = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioContext;
        this.samplingRate = this.audioContext.sampleRate;

        navigator.getUserMedia = (
            navigator.getUserMedia || navigator.webkitGetUserMedia);
        navigator.getUserMedia(
            {audio: true, video: false},
            this.startUserMedia,
            (error) => console.log(error));
    }

    pause = () => {
        this.audioRef.current.pause();
        this.audioRef.current.currentTime = 0;
        this.setState({playing: null});
    }

    play = (event) => {
        const row = event.target.parentNode.parentNode;
        const playing = parseInt(row.getAttribute('data-key'));
        const filename = row.firstChild.getAttribute('filename');

        let formData = new FormData;
        formData.append('filename', filename);

        fetch('/retrieve', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                response.blob().then(blob => {
                    this.audioRef.current.src = URL.createObjectURL(blob);
                    this.audioRef.current.play();
                    this.setState({playing: playing});
                });
            } else {
                console.log('Audio file ${filename} could not be found');
            }
        });
    }

    record() {
        // Stop current playback
        if (this.state.playing !== null) {
            this.setState({playing: null});
        }

        // Reset previous recording and results
        this.clearRecording();
        this.setState({matches: []});

        if (!this.recorder) {
            this.makeRecorder();
        } else {
            this.recorder.record();
        }

        // Update the timer animation every 100 ms
        this.recordingStartTime = (new Date()).getTime();
        this.timerAnimationId = setInterval(
            () => {
                let currentTime = (new Date()).getTime();
                let elapsed = (currentTime - this.recordingStartTime) / 10;
                let recordingProgress = elapsed / this.props.maxRecordingLength;
                this.setState({recordingProgress: recordingProgress});
            },
            100
        );

        // Stop recording after the maximum allowed recording length
        // has been reached
        this.recordingTimerId = setTimeout(
            () => {
                clearInterval(this.timerAnimationId);
                this.setState({recording: false})
            },
            this.props.maxRecordingLength * 1000
        );
    }

    render() {
        const searchWidth = this.props.foley ? 'col-4' : 'col-8';
        const displayInstructions = this.state.instructions ? 'none' : 'initial';
        return (
          <div>
          {this.instructions()}
          <div className='container'>
            <div ref={this.headerRef} className='row header d-flex align-items-center'>
              <p className='text48 open-sans400 light-purple-text m-0 ml-4 my-2'>
                Voogle
              </p>
              <button
                className='btn no-border hover-light-purple dark-text lato400 float-right ml-auto h-50 mr-4 pointer'
                onClick={this.toggleInstructions}
              >
                Show Instructions
              </button>
            </div>
            <div className='row'>
              <div className={'col p-0 ' + searchWidth}>
                <div className='d-flex justify-content-center align-items-center' style={{height: this.state.searchHeight}}>
                  <div>
                    <p className='open-sans400 text40 light-purple-text mb-2'>
                      {this.searchText()}
                    </p>
                    <div>
                      <div className='voogle-button' onClick={this.toggleRecording}>
                        <img className=' center' src={logo}/>
                        <CircularProgressbar
                          background
                          percentage={this.state.recordingProgress}
                          textForPercentage={null}
                          strokeWidth={5}
                          styles={{
                            background: {fill: '#4E2A83'},
                            path: {
                                animation: 'stroke-dashoffset 0.5s ease 0s',
                                stroke: '#B4A5CB',
                                transition: 'none'
                            },
                            trail: {stroke: '#4E2A83'}
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
                <div ref={this.footerRef} className='light rounded-top py-2'>
                  <input
                    className='transparent no-border w-100 text-box-text-color lato400 text24 ml-4'
                    onBlur={(e) => e.target.placeholder = '(Optional) Enter a text description'}
                    onChange={this.handleTextInput}
                    onFocus={(e) => e.target.placeholder = ''}
                    onKeyDown={this.textSubmit}
                    placeholder='(Optional) Enter a text description'
                    type='text'
                    value={this.state.textInput}
                  />
                </div>
              </div>
              <div className='col-4 p-0'>
                {this.renderMatches()}
              </div>
              {this.props.foley ? this.renderFoley() : null}
            </div>
          </div>
          </div>
        );
    }

    renderMatches = () => {
        const height = this.state.matchesHeight / 12;
        if (this.state.matches.length &&
           !this.state.recording &&
           !this.state.searching) {
            return (
                <div style={{
                    height: this.state.matchesHeight,
                    overflowY: 'auto',
                    overflowX: 'hidden'
                }}>
                  <AudioFiles
                    download={this.download}
                    files={this.state.matches}
                    height={height}
                    play={this.play}
                    playing={this.state.playing}
                    text={this.state.textInput}
                  />
                  <audio ref={this.audioRef} onEnded={() => this.pause()}/>
                </div>
            );
        } else {
            const grayedBoxes = [...Array(12).keys()].map(value => {
                const color = value % 2 ? 'search-light' : 'search-dark';
                return <div className={color} style={{height: height}} key={value}></div>
            });
            return (
                <div>
                  <div className='dim'></div>
                  {grayedBoxes}
                </div>
            );
        }
    }

    searchText = () => {
        if (this.state.searching) {
            return 'Voogling...';
        } else if (this.state.recording) {
            return 'Listening...';
        } else {
            return 'Click to Voogle';
        }
    }

    send = (buffer) => {
        let query = new Blob([this.levelDetect(buffer[0])]);

        let formData = new FormData;
        formData.append('query', query);
        formData.append('sampling_rate', this.samplingRate);
        formData.append('text_input', this.state.textInput);

        this.searchStartTime = (new Date()).getTime();
        fetch('/search', {
            method: 'POST',
            body: formData
        }).then(response => {
            response.json().then(results => {
                let newMatches = [];
                for (let i = 0; i < results.matches.length; i++) {
                    newMatches.push({
                        rank: i,
                        filename: results.matches[i],
                        textMatch: results.text_matches[i],
                        similarityScore: results.similarity_scores[i]
                    })
                }
                this.setState({ matches: newMatches, searching: false });
            });
        });
    }

    sendQuery = () => {
        console.log('sending')
        this.recorder.getBuffer(this.send);
    }

    startUserMedia = (stream) => {
        // Plug the user's mic into the graph
        let audioStream = this.audioContext.createMediaStreamSource(stream);

        // Plug mic into recorder and recorder into waveform
        this.recorder = new Recorder(audioStream, { numChannels: 1});

        this.recorder.record();
    }

    stop = () => {
        // Stop recording
        this.recorder.stop();

        // Indicate that a query is available
        this.setState({
            hasRecorded: true,
            recordingProgress: 0,
            searching: true
        });

        // Stop periodically drawing the waveform while recording
        clearInterval(this.drawIntervalId);

        // Stop updating the timer animation
        clearInterval(this.timerAnimationId);

        // Stop the recording timer
        clearTimeout(this.recordingTimerId);

        // Find the user's audio via level detection
        this.sendQuery();
    }

    textSubmit = (event) => {
        if (event.key === 'Enter') {
            console.log('')
            event.preventDefault();
            event.stopPropagation();
            if (this.state.textInput && this.state.hasRecorded) {
                this.sendQuery();
            }
        }
    }

    toggleInstructions = () => {
        this.setState(state => ({instructions: !state.instructions}));
    }

    toggleRecording = () => {
        // Event handler for the recording button
        if (!this.state.searching) {
            this.setState(state => ({recording: !state.recording}));
        }
    }

    updateSearchHeight = () => {
        const headerRect = this.headerRef.current.getBoundingClientRect();
        const footerRect = this.footerRef.current.getBoundingClientRect();

        const headerHeight = headerRect.bottom - headerRect.top;
        const footerHeight = footerRect.bottom - footerRect.top;
        const matchesHeight = window.innerHeight - headerHeight - 16;
        const searchHeight = matchesHeight - footerHeight;

        this.setState({
            matchesHeight: matchesHeight,
            searchHeight: searchHeight
        });
    }
}

Voogle.defaultProps = {
    // The maximum duration (in seconds) of a user's recording
    maxRecordingLength: 8,

    // The minimum audio buffer value above which automatic region placement
    // will begin
    regionStartThreshold: 0.10,

    // The level below which the automatically placed region will end
    regionEndThreshold: 0.03,

    // The amount of time (in seconds) to add to the beginning of the query
    regionStartTolerance: 0.03,

    // The amount of time (in seconds) to add to the end of the query
    regionEndTolerance: 0.20
};

export default Voogle;
