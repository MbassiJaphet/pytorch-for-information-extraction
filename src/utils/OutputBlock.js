import React from 'react';

class OutputBlock extends React.Component {
    constructor(props) {
        super(props);
        this._currentFile = null;
        this.state = {
            outputString: '',
        };
    }

    static getDerivedStateFromProps(props, state) {
        if (props.id !== state.prevFile) {
            return {
                outputString: '',
                prevFile: props.file,
            };
        }
        return null;
    }

    componentDidMount() {
        this._loadAsyncData(this.props.file);
    }

    componentDidUpdate(prevProps, prevState) {
        if (!this.state.outputString) {
            this._loadAsyncData(this.props.file);
        }
    }

    componentWillUnmount() {
        this._currentFile = null;
    }


    render() {
        return (
            <pre className="output-block">
                <code>{this.state.outputString}</code>
            </pre>
        );
    }

    _loadAsyncData(file) {
        // if (file === this._currentFile) {
        //     // Data for this id is already loading
        // }

        this._currentFile = file;
        fetch(`/pytorch-for-information-extraction/code-snippets/${file}.txt`)
            .then(res => res.text())
            .then(text => {
                if (file === this._currentFile) {
                    this.setState({ outputString: text });
                }
            })
            .catch(e => {
                console.log(e)
            })
    }
}

export default OutputBlock