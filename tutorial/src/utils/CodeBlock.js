import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';

class CodeBlock extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            codeString: '',
        };
        this._currentFile = null;
    }

    static getDerivedStateFromProps(props, state) {
        if (props.id !== state.prevFile) {
            return {
                codeString: '',
                prevFile: props.file,
            };
        }
        return null;
    }

    componentDidMount() {
        this._loadAsyncData(this.props.file);
    }

    componentDidUpdate(prevProps, prevState) {
        if (!this.state.codeString) {
            this._loadAsyncData(this.props.file);
        }
    }

    componentWillUnmount() {
        this._currentFile = null;
    }

    render() {
        return (
            <SyntaxHighlighter language="python" style={atomOneDark} customStyle={customStyle} codeTagProps={{ style: { color: '#e0e0e0' } }}>
                {this.state.codeString}
            </SyntaxHighlighter>
        );
    }

    _loadAsyncData(file) {
        // if (file === this._currentFile) {
        //     // Data for this id is already loading
        // }

        this._currentFile = file;
        fetch(`/pytorch-for-information-extraction/code-snippets/${file}.py`)
            .then(res => res.text())
            .then(text => {
                if (file === this._currentFile) {
                    this.setState({ codeString: text });
                }
            })
            .catch(e => {
                console.log(e)
            })
    }
}

export default CodeBlock

const customStyle = {
    borderRadius: 0,
    overflow: 'auto',
    maxHeight: '75vh',
    fontSize: "0.67em",
    borderRadius: 8
};