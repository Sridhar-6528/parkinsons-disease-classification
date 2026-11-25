"""
Parkinson's Multimodal AI Web Application Launcher
"""

from web_app import app

if __name__ == '__main__':
    print("Starting Parkinson's AI Web Diagnostic Application on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
