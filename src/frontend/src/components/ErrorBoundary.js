import React from 'react';

class ErrorBoundary extends React.Component {
    state = { error: null };

    static getDerivedStateFromError(error) {
        return { error };
    }

    componentDidCatch(error, info) {
        console.error('Render error:', error, info.componentStack);
    }

    render() {
        if (this.state.error) {
            return (
                <div className="container mt-5 text-center">
                    <h4>λύποῦμαι. Encountered an error in displaying this page.</h4>
                    <p className="text-muted">{String(this.state.error)}</p>
                    <button
                        className="btn btn-primary"
                        onClick={() => this.setState({ error: null })}
                    >
                        Try again
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

export default ErrorBoundary;