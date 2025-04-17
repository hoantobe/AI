from app.utils.rate_limiter import RateLimiter
import time

def test_rate_limiter():
    limiter = RateLimiter(requests_per_min=2)
    
    # First request - should pass immediately
    start = time.time()
    limiter.wait_if_needed()
    assert time.time() - start < 0.1
    
    # Second request - should also pass
    limiter.wait_if_needed()
    assert time.time() - start < 0.1
    
    # Third request - should wait
    limiter.wait_if_needed()
    assert time.time() - start >= 60