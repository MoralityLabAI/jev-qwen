"""Put src/ on sys.path so scripts run without installing the package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Verify TLS against the operating system's trust store instead of certifi's bundle. On this
# machine an antivirus re-signs HTTPS traffic with a root that only the Windows store knows,
# so Hugging Face requests fail certificate verification otherwise. Verification stays on.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
