import sys

from common import plot_result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_result.py <results.json>")
        sys.exit(1)
    plot_result(sys.argv[1])
