"""Run this yourself: it prompts for your WRDS password directly (never
passed through the assistant) and pulls a few days of AAPL from CRSP to
confirm WRDSDataLoader actually works against your account.

    venv/Scripts/python.exe scripts/verify_wrds.py YOUR_WRDS_USERNAME
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.data_layer import WRDSDataLoader


def main():
    if len(sys.argv) != 2:
        print("Usage: verify_wrds.py YOUR_WRDS_USERNAME")
        sys.exit(1)

    username = sys.argv[1]
    loader = WRDSDataLoader(["AAPL"], "2024-01-02", "2024-01-31", wrds_username=username)

    print(f"Connecting to WRDS as '{username}' -- you'll be prompted for your password now.")
    prices = loader.load_adj_close()

    print("\nSuccess. CRSP adjusted close for AAPL, Jan 2024:")
    print(prices)
    print(f"\n{len(prices)} trading days pulled. WRDSDataLoader is verified working.")


if __name__ == "__main__":
    main()
