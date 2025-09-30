// Service Worker for Parkinson's Detection PWA
const CACHE_NAME = 'parkinson-detection-v1.0.0';
const API_CACHE_NAME = 'parkinson-api-v1.0.0';

// Files to cache for offline functionality
const STATIC_CACHE_FILES = [
    '/frontend/',
    '/frontend/index.html',
    '/frontend/styles.css',
    '/frontend/app.js',
    '/frontend/manifest.json',
    // SVG icon files
    '/frontend/assets/icon-144.svg',
    '/frontend/assets/icon-192.svg',
    '/frontend/assets/icon-512.svg'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    /\/api\/health/,
    /\/api\/models\/info/
];

// Install event - cache static files
self.addEventListener('install', (event) => {
    console.log('Service Worker: Install event');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Service Worker: Caching static files');
                return cache.addAll(STATIC_CACHE_FILES.map(url => new Request(url, {
                    cache: 'reload'
                })));
            })
            .catch((error) => {
                console.error('Service Worker: Error caching static files:', error);
            })
    );
    
    // Force activation
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker: Activate event');
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== API_CACHE_NAME) {
                        console.log('Service Worker: Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    
    // Take control of all clients
    self.clients.claim();
});

// Fetch event - serve cached content when offline
self.addEventListener('fetch', (event) => {
    const request = event.request;
    const url = new URL(request.url);
    
    // Handle different types of requests
    if (request.method === 'GET') {
        // Handle API requests
        if (url.pathname.startsWith('/api/')) {
            event.respondWith(handleApiRequest(request));
        }
        // Handle static file requests
        else {
            event.respondWith(handleStaticRequest(request));
        }
    }
    // Handle POST requests (for analysis)
    else if (request.method === 'POST' && url.pathname === '/api/analyze') {
        event.respondWith(handleAnalysisRequest(request));
    }
});

// Handle static file requests with cache-first strategy
async function handleStaticRequest(request) {
    try {
        // Try cache first
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Fallback to network
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse.status === 200) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
        
    } catch (error) {
        console.error('Service Worker: Error handling static request:', error);
        
        // Return offline page or default response
        if (request.destination === 'document') {
            return caches.match('/index.html');
        }
        
        // For other assets, return a generic error response
        return new Response('Offline - Resource not available', {
            status: 503,
            statusText: 'Service Unavailable'
        });
    }
}

// Handle API requests with network-first strategy
async function handleApiRequest(request) {
    const url = new URL(request.url);
    
    try {
        // Try network first for API requests
        const networkResponse = await fetch(request);
        
        // Cache successful responses for certain endpoints
        if (networkResponse.status === 200 && shouldCacheApiResponse(url.pathname)) {
            const cache = await caches.open(API_CACHE_NAME);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
        
    } catch (error) {
        console.log('Service Worker: Network failed for API request, trying cache');
        
        // Fallback to cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline response for specific endpoints
        return getOfflineApiResponse(url.pathname);
    }
}

// Handle analysis requests (special case for offline demo)
async function handleAnalysisRequest(request) {
    try {
        // Try network first
        const networkResponse = await fetch(request);
        return networkResponse;
        
    } catch (error) {
        console.log('Service Worker: Analysis request failed, providing offline demo response');
        
        // Return demo response when offline
        const demoResponse = {
            prediction: 'Not Affected',
            confidence: 0.75,
            voice_confidence: 0.7,
            tremor_confidence: 0.8,
            features: {
                'Voice Stability': 0.8,
                'Tremor Frequency': 0.6,
                'Speech Rhythm': 0.7,
                'Motion Variability': 0.5,
                'Vocal Tremor': 0.6,
                'Postural Stability': 0.7
            },
            metadata: {
                processing_time: 2.0,
                audio_duration: 10.0,
                motion_samples: 1000,
                model_version: '1.0.0 (offline)',
                offline_mode: true
            }
        };
        
        return new Response(JSON.stringify(demoResponse), {
            status: 200,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
}

// Check if API response should be cached
function shouldCacheApiResponse(pathname) {
    return API_CACHE_PATTERNS.some(pattern => pattern.test(pathname));
}

// Get offline response for API endpoints
function getOfflineApiResponse(pathname) {
    const offlineResponses = {
        '/api/health': {
            status: 'offline',
            timestamp: new Date().toISOString(),
            version: '1.0.0'
        },
        '/api/models/info': {
            models: {
                voice_analysis: {
                    type: 'ensemble',
                    algorithms: ['SVM', 'Random Forest', 'XGBoost'],
                    features: ['MFCC', 'Spectral', 'Prosodic', 'Voice Quality']
                },
                tremor_analysis: {
                    type: 'ensemble',
                    algorithms: ['SVM', 'Random Forest', 'XGBoost'],
                    features: ['Frequency Domain', 'Time Domain', 'Statistical']
                }
            },
            version: '1.0.0 (offline)',
            offline_mode: true
        }
    };
    
    const response = offlineResponses[pathname];
    
    if (response) {
        return new Response(JSON.stringify(response), {
            status: 200,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
    
    return new Response(JSON.stringify({
        error: 'Offline - Service not available',
        offline_mode: true
    }), {
        status: 503,
        headers: {
            'Content-Type': 'application/json'
        }
    });
}

// Handle background sync for when connection is restored
self.addEventListener('sync', (event) => {
    console.log('Service Worker: Background sync triggered');
    
    if (event.tag === 'background-sync') {
        event.waitUntil(
            // Handle any queued operations when connection is restored
            handleBackgroundSync()
        );
    }
});

async function handleBackgroundSync() {
    // This could be used to sync any offline data when connection is restored
    console.log('Service Worker: Performing background sync');
    
    try {
        // Check if we can reach the server
        const response = await fetch('/api/health');
        if (response.ok) {
            console.log('Service Worker: Connection restored');
            
            // Notify all clients that connection is restored
            const clients = await self.clients.matchAll();
            clients.forEach(client => {
                client.postMessage({
                    type: 'CONNECTION_RESTORED'
                });
            });
        }
    } catch (error) {
        console.log('Service Worker: Still offline');
    }
}

// Handle push notifications (if needed in the future)
self.addEventListener('push', (event) => {
    console.log('Service Worker: Push notification received');
    
    const options = {
        body: event.data ? event.data.text() : 'New update available',
        icon: '/assets/icon-192.svg',
        badge: '/assets/icon-192.svg',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: '1'
        },
        actions: [
            {
                action: 'explore',
                title: 'Open App',
                icon: '/assets/icon-192.svg'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/assets/icon-192.svg'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('Parkinson\'s Detection PWA', options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    console.log('Service Worker: Notification clicked');
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Message handling for communication with main app
self.addEventListener('message', (event) => {
    console.log('Service Worker: Message received:', event.data);
    
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

console.log('Service Worker: Script loaded');