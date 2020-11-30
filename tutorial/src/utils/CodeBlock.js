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

    _highlightLine(lineNumber) {
        let style = { display: 'block' };
        if (this.props.lines && this.props.lines.includes(lineNumber)) {
            style.backgroundColor = 'rgb(144, 202, 249, 0.15)';
            // style.color = 'blue';
        }
        return { style };
    }

    render() {
        return (
            <div   class="code-block">
                <SyntaxHighlighter language="python" lineProps={this._highlightLine.bind(this)}
                wrapLines={true} lineNumberStyle={{ color: "#80d6ff" }} style={atomOneDark}
                showLineNumbers={true} customStyle={customStyle} codeTagProps={{ style: { color: '#e0e0e0' } }}>
                    {this.state.codeString}
                </SyntaxHighlighter>
            </div>
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