import React from 'react';

const CodeBlock = ({ code }) => {
    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
        alert("Code copied to clipboard!");
    };

    return (
        <div className="code-block">
            <button className="copy-button" onClick={copyToClipboard}>Copy</button>
            <pre>
                <code>{code}</code>
            </pre>
        </div>
    );
};

export default CodeBlock;
