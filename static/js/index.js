import OldInterface from './oldinterface.js'
import React from 'react'
import ReactDOM from 'react-dom'
import Voogle from './voogle.js'

const root = document.getElementById('root');

if (process.env.interface === 'old') {
    ReactDOM.render(<OldInterface/>, root);
} else if (process.env.interface === 'foley') {
    ReactDOM.render(<Voogle foley={true}/>, root);
} else {
    ReactDOM.render(<Voogle/>, root);
}

