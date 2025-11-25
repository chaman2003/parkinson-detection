"""
Test Script: Multiple Consecutive Requests
Tests if the server can handle multiple requests without restart

Run this after starting the backend server with: python app.py
"""

import requests
import json
import time
import os

# Configuration
BASE_URL = "elease-unmeaning-mireille.ngrok-free.dev/api"
TEST_ITERATIONS = 3  # Number of consecutive tests to run

def create_test_audio():
    """Create a simple test audio file"""
    # We'll just use an existing file from the dataset
    dataset_path = "datasets/voice_dataset"
    if os.path.exists(dataset_path):
        files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
        if files:
            return os.path.join(dataset_path, files[0])
    return None

def create_test_motion_data():
    """Create test motion data"""
    return json.dumps({
        "gyroscope": [{"x": 0.1, "y": 0.2, "z": 0.3, "timestamp": 1000 + i*10} for i in range(100)],
        "accelerometer": [{"x": 0.5, "y": 0.6, "z": 9.8, "timestamp": 1000 + i*10} for i in range(100)]
    })

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_single_analysis(iteration):
    """Test a single analysis request"""
    print(f"\n{'='*60}")
    print(f"TEST ITERATION {iteration}")
    print(f"{'='*60}")
    
    # Get test audio file
    audio_file_path = create_test_audio()
    if not audio_file_path:
        print("✗ No test audio file available")
        return False
    
    # Prepare test data
    motion_data = create_test_motion_data()
    
    try:
        # Send request
        print(f"Sending analysis request...")
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {
                'motion_data': motion_data,
                'test_mode': 'both'
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/analyze", files=files, data=data, timeout=60)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Analysis completed in {elapsed:.2f}s")
                print(f"  Prediction: {result.get('prediction', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 0):.2%}")
                return True
            else:
                print(f"✗ Analysis failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Analysis error: {e}")
        return False

def main():
    """Run the test suite"""
    print("\n" + "="*60)
    print("MULTIPLE REQUEST TEST - Server Restart Fix Verification")
    print("="*60)
    
    # Test health check first
    print("\n1. Testing server health...")
    if not test_health_check():
        print("\n✗ Server is not responding. Make sure backend is running.")
        return
    
    # Run multiple consecutive tests
    print(f"\n2. Running {TEST_ITERATIONS} consecutive analysis tests...")
    print("   (This tests if server can handle multiple requests without restart)\n")
    
    results = []
    for i in range(1, TEST_ITERATIONS + 1):
        success = test_single_analysis(i)
        results.append(success)
        
        # Small delay between requests
        if i < TEST_ITERATIONS:
            print(f"\nWaiting 2 seconds before next test...")
            time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✅ SUCCESS! All tests passed without server restart!")
        print("The server restart fix is working correctly.")
    else:
        print(f"\n⚠️ WARNING! {len(results) - sum(results)} test(s) failed.")
        print("The server may still have issues with consecutive requests.")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
