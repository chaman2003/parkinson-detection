/**
 * Configuration for Backend API URL
 * Supports both local development and production deployment
 */

const AppConfig = {
    /**
     * Get the backend API base URL based on environment
     * 
     * For Vercel deployment:
     *   - Reads from BACKEND_URL environment variable
     *   - Should be set to: https://elease-unmeaning-mireille.ngrok-free.dev
     * 
     * For local development (run-locally.ps1):
     *   - Uses relative path '/api' which is proxied by server.py to localhost:5000
     */
    getBackendUrl() {
        // Check if we're in browser environment
        if (typeof window === 'undefined') {
            return '/api';
        }

        // Check if we're running locally (localhost or 127.0.0.1)
        const hostname = window.location.hostname;
        const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';

        if (isLocal) {
            // Local development: use proxy path
            // The server.py will proxy /api/* to localhost:5000
            console.log('🏠 Local development mode detected');
            return '/api';
        } else {
            // Production (Vercel or ngrok): use environment variable + /api path
            // Vercel will inject BACKEND_URL as a global variable
            const backendUrl = window.BACKEND_URL || 
                             (typeof process !== 'undefined' && process.env?.BACKEND_URL);
            
            if (backendUrl) {
                console.log('🚀 Production mode: Using configured backend URL:', backendUrl);
                // Add /api to the backend URL for production
                return backendUrl + '/api';
            } else {
                console.error('⚠️ BACKEND_URL not configured!');
                console.error('Please set BACKEND_URL to: https://elease-unmeaning-mireille.ngrok-free.dev');
                // Fallback to relative path (will likely fail in production)
                return '/api';
            }
        }
    },

    /**
     * Get the full API endpoint URL
     */
    getApiUrl(endpoint) {
        const baseUrl = this.getBackendUrl();
        // Remove leading slash from endpoint if present
        const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
        return `${baseUrl}/${cleanEndpoint}`;
    }
};

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.AppConfig = AppConfig;
}
