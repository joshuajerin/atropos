"""
Run the Trajectory API server.
"""

import argparse
import os
import secrets
import ssl

import uvicorn

from atroposlib.api.security import get_default_api_key


def main():
    """
    Run the API server.
    Args:
        host: The host to run the API server on.
        port: The port to run the API server on.
        reload: Whether to reload the API server on code changes.
        ssl_cert: Path to SSL certificate file
        ssl_key: Path to SSL key file
        local_only: Whether to bind only to localhost (127.0.0.1)
        use_ssl: Whether to use SSL/TLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Server host (default: 127.0.0.1, use 0.0.0.0 to allow external connections)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--ssl-cert", type=str, help="Path to SSL certificate file")
    parser.add_argument("--ssl-key", type=str, help="Path to SSL key file")
    parser.add_argument("--use-ssl", action="store_true", help="Enable SSL/TLS")
    parser.add_argument("--show-api-key", action="store_true", help="Display the API key and exit")
    
    args = parser.parse_args()
    
    # Display the API key if requested
    if args.show_api_key:
        print(f"API Key: {get_default_api_key()}")
        print("Use this key for API authentication")
        print("For security, set your own key with the ATROPOS_API_KEY environment variable")
        return
    
    # Print warnings for security considerations
    if args.host == "0.0.0.0" and not args.use_ssl:
        print("WARNING: Server is binding to all interfaces (0.0.0.0) without SSL/TLS")
        print("This configuration may expose your server to security risks")
        print("Consider using --use-ssl with appropriate certificates")
        print("Or restrict to localhost with --host=127.0.0.1")
    
    # Configure SSL if requested
    ssl_context = None
    if args.use_ssl:
        if args.ssl_cert and args.ssl_key:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
        else:
            print("ERROR: SSL/TLS requested but no certificate/key provided")
            print("Please provide --ssl-cert and --ssl-key parameters")
            return
    
    # Display connection information
    protocol = "https" if args.use_ssl else "http"
    host_display = args.host if args.host != "0.0.0.0" else "your_ip_address"
    print(f"Starting server at {protocol}://{host_display}:{args.port}")
    print(f"API Authentication is ENABLED - secure your API key")
    
    # Run the server
    uvicorn.run(
        "atroposlib.api:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        ssl_keyfile=args.ssl_key if args.use_ssl else None,
        ssl_certfile=args.ssl_cert if args.use_ssl else None
    )


if __name__ == "__main__":
    main()
