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
            loadedMatch: null,
            matches: [],
            matchesHeight: 0,
            playingMatch: false,
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

        // Position to start playback in seconds
        this.recordingPlaybackStart = 0;

        // Refs for height calculations
        this.headerRef = React.createRef();
        this.footerRef = React.createRef();
        this.searchRef = React.createRef();
    }

    componentDidMount() {
        this.updateSearchHeight();
        window.addEventListener('resize', this.updateSearchHeight);
    }

    componentDidUpdate(prevProps, prevState) {}

    componentWillMount() {
        document.body.style.backgroundColor = '#1C142D';
    }

    componentWillUnmount() {
        document.body.style.backgroundColor = null;
        window.removeEventListener('resize', this.updateSearchHeight);
    }

    clearRecording = () => {
        // Erase the recorded audio
        this.queryBuffer = null;
        this.recorder.clear();
        this.setState({
            hasRecorded: false,
            playingRecording: false,
        });
    }

    clearMatch = () => {
        this.setState({
            loadedMatch: null,
            playingMatch: false,
        });
    }

    download = () => {
        // Don't download if we have no audio loaded
        if (!this.state.loadedMatch) {
            return;
        }

        // Encode the audio as a WAV file
        WavEncoder.encode({
            sampleRate: this.matchWavesurfer.backend.ac.sampleRate,
            channelData: [this.matchWavesurfer.backend.buffer.getChannelData(0)]
        }).then((buffer) => {
            let blob = new Blob([buffer], {type: 'audio/wav'});
            let filename = this.state.loadedMatch.slice(
                this.state.loadedMatch.lastIndexOf('/') + 1);

            // Download hack: create a ghost element with a download link and
            // click it
            let link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = filename;
            link.click();
        });
    }

    drawRegion = () => {
        // Grab a copy of the audio buffer
        let buffer = this.wavesurfer.backend.buffer.getChannelData(0).slice();

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
        let end = buffer.length - buffer.reverse().findIndex((x) => {
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
        start = Math.max(0, start);
        end = Math.min(this.wavesurfer.getDuration(), end);

        // Save buffer indices for sending query
        this.start = start * this.samplingRate;
        this.end = end * this.samplingRate;

        // Add the region
        this.wavesurfer.addRegion({
            id: 'queryRegion',
            start: start,
            end: end,
            color: 'rgb(36,42,54,0.4)'
        });

        // Start playback at region start
        this.recordingPlaybackStart = start;

        let region = this.wavesurfer.regions.list.queryRegion;

        // Stop playback when region bound is passed
        region.on('out', () => {
            if (this.wavesurfer.getCurrentTime() > region.end - 0.001) {
                this.wavesurfer.stop();
                this.recordingPlaybackStart = start;
                this.setState({
                    playingRecording: false,
                    playRecordingText: 'Play'
                });
            }
        });

        // Change the bounds of the query when the region is resized
        region.on('update-end', (event) => {
            let newRegion = this.wavesurfer.regions.list.queryRegion;
            this.recordingPlaybackStart = newRegion.start + 0.001;
            this.start = Math.ceil(newRegion.start * this.samplingRate);
            this.end = Math.floor(newRegion.end * this.samplingRate);
        });
    }

    handleRecording = () => {
        if (this.state.recording) {
            // Stop playback
            if (this.state.playingRecording) {
                this.setState({
                    playingRecording: false,
                    playRecordingText: 'Play'
                })
            }
            if (this.state.playingMatch) {
                this.setState({
                    playingMatch: false,
                    playMatchText: 'Play'
                });
            }

            // Reset the waveforms
            this.clearRecording();
            this.clearMatch();

            // Clear the existing matches
            this.setState({ matches: [] });

            // Start recording
            this.recorder.record();

            // Periodically draw the waveform while recording
            this.drawIntervalId = setInterval(
                this.draw, this.props.drawingRate);

            // Update the timer animation every 100 ms
            this.recordingStartTime = (new Date()).getTime();
            this.timerAnimationId = setInterval(
                () => {
                    let currentTime = (new Date()).getTime();
                    let elapsed = (currentTime - this.recordingStartTime) /
                        10;
                    let recordingProgress = elapsed /
                        this.props.maxRecordingLength;
                    this.setState({ recordingProgress: recordingProgress });
                },
                100
            );

            // Stop recording after the maximum allowed recording length
            // has been reached
            this.recordingTimerId = setTimeout(
                () => {
                    clearInterval(this.timerAnimationId);
                    this.setState({
                        recording: false,
                        recordButtonText: 'Record'
                    })
                },
                this.props.maxRecordingLength * 1000
            );

        } else {
            // Stop recording
            this.recorder.stop();

            // Indicate that a query is available
            this.setState({ hasRecorded: true, recordingProgress: 0 });

            // Stop periodically drawing the waveform while recording
            clearInterval(this.drawIntervalId);

            // Stop updating the timer animation
            clearInterval(this.timerAnimationId);

            // Stop the recording timer
            clearTimeout(this.recordingTimerId);

            // Find the user's audio via level detection
            this.drawRegion();
        }
    }

    handleTextInput = (event) => {
        this.setState({textInput: event.target.value});
    }

    loadAudio = (key) => {
        // Don't retrieve the audio if we already have it
        if (key === this.state.loadedMatch) {
            this.matchWavesurfer.seekTo(0);
            this.setState({playingMatch: true, playMatchText: 'Pause'});
            return;
        }

        let formData = new FormData;
        formData.append('filename', key);

        fetch('/retrieve', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                response.blob().then(blob => {
                    this.matchWavesurfer.loadBlob(blob);
                    this.setState({ loadedMatch: key });
                });
            } else {
                console.log('Audio file ${key} could not be found');
            }
        })

    }

    render() {
        const searchWidth = this.props.foley ? 'col-4' : 'col-8';
        console.log(this.state.searchHeight);
        return (
          <div className='container'>
            <div ref={this.headerRef} className='row header d-flex align-items-center'>
              <p className='text48 open-sans400 light-purple-text m-0 ml-4 my-2'>Voogle</p>
              <button className='btn no-border light-purple dark-text lato400 float-right ml-auto h-50 mr-4'>Show Instructions</button>
            </div>
            <div className='row'>
              <div className={'col p-0 ' + searchWidth}>
                <div className='d-flex justify-content-center align-items-center' style={{height: this.state.searchHeight}}>
                  <div>
                  <p className='open-sans400 text32 light-purple-text mb-2'>Click to search</p>
                  <img className='img-fluid mx-auto d-block pb-5' src={logo}/>
                  </div>
                </div>
                <div ref={this.footerRef} className='light rounded-top'>
                  <p className='text-box-text-color lato400 text32 ml-3'>Describe your sound</p>
                </div>
              </div>
              <div className='col-4 p-0'>
                {this.renderMatches()}
              </div>
              {this.props.foley ? this.renderFoley() : null}
            </div>
          </div>
        );
    }

    renderMatches = () => {
        const height = this.state.matchesHeight / 10;
        if (this.matches && !this.recording) {
        } else {
            const grayedBoxes = [...Array(10).keys()].map(value => {
                const color = value % 2 ? 'search-light' : 'search-dark';
                return <div className={color} style={{height: height}} key={value}></div>
            });
            console.log(grayedBoxes)
            return <div>{grayedBoxes}</div>
        }
    }

    resizeMatches = () => {
        const top = this.resizeTopDiv.current.getBoundingClientRect().top;
        const btm = this.resizeBottomDiv.current.getBoundingClientRect().bottom;
        const lft = this.resizeTopDiv.current.getBoundingClientRect().left;
        const rgt = this.resizeBottomDiv.current.getBoundingClientRect().right;
        this.setState({ matchDivHeight: btm - top, matchDivWidth: rgt - lft });
    }

    search = () => {
        // Event handler for the search button
        if (this.state.hasRecorded) {
            // Get the full recording
            let buffer = this.wavesurfer.backend.buffer.getChannelData(0);

            // Grab the segment containing the query
            let query = buffer.slice(this.start, this.end)

            // Send the underlying data as a bytestream
            this.sendQuery(new Blob([query.buffer]));
        }
    }

    submit = (event) => {
        if (event.key == 'Enter') {
            this.search();
        }
    }

    sendQuery = (query) => {
        // Don't send search request if no recording exists
        if (!this.state.hasRecorded) {
            return;
        }

        let formData = new FormData;
        formData.append('query', query);
        formData.append('sampling_rate', this.samplingRate);
        formData.append('text_input', this.state.textInput);

        this.setState({ searching: true });
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

    togglePlayMatch = () => {
        // Event handler for the play/pause button
        this.setState(state => {
            if (!state.playingMatch && state.loadedMatch) {
                return {
                    playingMatch: true,
                    playMatchText: 'Pause'
                };
            } else {
                return {
                    playingMatch: false,
                    playMatchText: 'Play'
                }
            }
        });
    }

    toggleRecording = () => {
        // Event handler for the recording button
        this.setState(state => {
            if (state.recording) {
                return {
                    recording: false,
                    recordButtonText: 'Record'
                };
            } else {
                return {
                    recording: true,
                    recordButtonText: 'Stop Recording'
                }
            }
        });
    }

    updateSearchHeight = () => {
        const headerRect = this.headerRef.current.getBoundingClientRect();
        const footerRect = this.footerRef.current.getBoundingClientRect();

        const headerHeight = headerRect.bottom - headerRect.top;
        const footerHeight = footerRect.bottom - footerRect.top;
        const matchesHeight = window.innerHeight - headerHeight - 16;
        const searchHeight = matchesHeight - footerHeight;

        this.setState({ matchesHeight: matchesHeight, searchHeight: searchHeight });
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
