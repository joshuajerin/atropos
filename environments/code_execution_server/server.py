"""
Instructions:

# Build the image
docker build -t cpp-flask-executor .

# Run the container
docker run -p 5002:5002 cpp-flask-executor

curl -X POST http://localhost:5002/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "#include<iostream>\nint main(){int x; std::cin>>x; std::cout<<x*x; return 0;}", "input": "4"}'



"""

import os
import subprocess
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/execute", methods=["POST"])
def execute_code():
    try:
        # Receive C++ code from API request
        data = request.json
        code = data.get("code", "")
        test_cases = data.get("input", "")

        if not code:
            return jsonify({"error": "No code provided"}), 400

        file_id = str(uuid.uuid4())
        py_filename = f"/tmp/{file_id}.py"

        # Save the Python code to a file
        with open(py_filename, "w") as f:
            f.write(code)

        # Execute Python code using subprocess
        exec_result = subprocess.run(
            ["python", py_filename],
            input=test_cases,
            capture_output=True,
            text=True,
            timeout=5,  # Prevent infinite loops
        )

        # Cleanup temporary files
        os.remove(py_filename)

        return jsonify({"output": exec_result.stdout, "error": exec_result.stderr})
        """ C++ stuff

        file_id = str(uuid.uuid4())
        cpp_filename = f"/tmp/{file_id}.cpp"
        exe_filename = f"/tmp/{file_id}.out"

        # Save code to a file
        with open(cpp_filename, "w") as f:
            f.write(code)

        # Compile C++ code
        compile_result = subprocess.run(["g++", cpp_filename, "-o", exe_filename],
                                        capture_output=True, text=True)

        for i in range(5):
            print(i)
            time.sleep(1)

        if compile_result.returncode != 0:
            return jsonify({"error": "Compilation failed", "details": compile_result.stderr})

        print(test_cases)
        # Execute compiled binary
        exec_result = subprocess.run([exe_filename], input=test_cases, capture_output=True, text=True, timeout=5)

        # Cleanup temporary files
        os.remove(cpp_filename)
        os.remove(exe_filename)

        return jsonify({"output": exec_result.stdout, "error": exec_result.stderr})"""

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Code execution server")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Server host (default: 127.0.0.1, use 0.0.0.0 to allow external connections)")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--with-ssl", action="store_true", help="Enable SSL (requires cert.pem and key.pem files)")
    
    args = parser.parse_args()
    
    # Add basic API key authentication when not in debug mode
    if not args.debug and args.host == "0.0.0.0":
        from functools import wraps
        from flask import request, abort
        
        API_KEY = os.environ.get("CODE_EXEC_API_KEY", os.urandom(24).hex())
        print(f"API Key: {API_KEY}")
        print("Use this key in the X-API-Key header for authentication")
        
        def require_api_key(view_function):
            @wraps(view_function)
            def decorated_function(*args, **kwargs):
                if request.headers.get("X-API-Key") != API_KEY:
                    abort(401)  # Unauthorized
                return view_function(*args, **kwargs)
            return decorated_function
        
        # Apply the decorator to the route
        app.view_functions["execute_code"] = require_api_key(app.view_functions["execute_code"])
        
        print("WARNING: Server is binding to all interfaces (0.0.0.0)")
        print("API key authentication is enabled, but consider these additional security measures:")
        print("1. Use a firewall to restrict access")
        print("2. Set up SSL/TLS for encrypted communication")
        print("3. Run in a restricted Docker container")
    
    # Run the app with SSL if requested
    if args.with_ssl:
        if os.path.exists("cert.pem") and os.path.exists("key.pem"):
            app.run(host=args.host, port=args.port, ssl_context=("cert.pem", "key.pem"), debug=args.debug)
        else:
            print("ERROR: SSL certificates not found. Create cert.pem and key.pem or disable --with-ssl")
            print("You can generate self-signed certificates with:")
            print("openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
