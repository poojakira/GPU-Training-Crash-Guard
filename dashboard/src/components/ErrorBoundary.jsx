import React from 'react';

// NVIDIA-style modern Error Boundary
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Dashboard Component Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full w-full bg-[#111] border border-red-500/30 rounded-lg p-6">
          <div className="text-center">
            <h3 className="text-red-500 font-bold mb-2">Component Crashed</h3>
            <p className="text-gray-400 text-sm mb-4">
              {this.state.error?.message || "An unexpected error occurred rendering this panel."}
            </p>
            <button
              className="px-4 py-2 bg-[#1a1a1a] hover:bg-[#222] border border-[#333] rounded text-sm text-[#76b900] transition-colors"
              onClick={() => this.setState({ hasError: false })}
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
