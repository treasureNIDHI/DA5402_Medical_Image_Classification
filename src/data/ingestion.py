import os

RAW_DIR = "data/raw"

def main():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)

    print("Data ingestion pipeline initialized")
    print("Existing data preserved in data/raw")

if __name__ == "__main__":
    main()