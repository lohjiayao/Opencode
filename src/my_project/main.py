import pandas as pd

def main():
    print("Welcome to my learning project!")
    
    # Example with pandas
    data = {"name": ["Alice", "Bob", "Charlie"], "score": [85, 92, 78]}
    df = pd.DataFrame(data)
    print("\nSample DataFrame:")
    print(df)

if __name__ == "__main__":
    main()
