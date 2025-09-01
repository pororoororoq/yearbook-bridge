"""
Health check endpoint for the bridge server
Place this in api/health.py
"""

def handler(request):
    """Vercel serverless function handler"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": {
            "status": "healthy",
            "service": "yearbook-bridge"
        }
    }
