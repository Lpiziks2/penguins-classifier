import pandas as pd
import seaborn as sns
import sqlite3
import os

def create_database():
    """Download the penguins dataset and store it in a SQLite database."""
    print("Loading penguins dataset...")
    # Load the penguins dataset
    penguins = sns.load_dataset("penguins").dropna()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create SQLite database
    print("Creating SQLite database...")
    conn = sqlite3.connect('data/penguins.db')
    
    # Create ISLANDS table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS ISLANDS (
        island_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    ''')
    
    # Insert unique islands
    islands = penguins['island'].unique()
    for i, island in enumerate(islands, 1):
        conn.execute('INSERT OR IGNORE INTO ISLANDS (island_id, name) VALUES (?, ?)', 
                    (i, island))
    
    # Create island_id mapping
    island_mapping = {island: i for i, island in enumerate(islands, 1)}
    
    # Add island_id to penguins dataframe
    penguins['island_id'] = penguins['island'].map(island_mapping)
    
    # Add animal_id column as unique identifier
    penguins['animal_id'] = range(1, len(penguins) + 1)
    
    # Create PENGUINS table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS PENGUINS (
        animal_id INTEGER PRIMARY KEY,
        species TEXT NOT NULL,
        bill_length_mm REAL,
        bill_depth_mm REAL,
        flipper_length_mm REAL,
        body_mass_g REAL,
        sex TEXT,
        island_id INTEGER,
        FOREIGN KEY (island_id) REFERENCES ISLANDS(island_id)
    )
    ''')
    
    # Insert penguin data
    penguins[['animal_id', 'species', 'bill_length_mm', 'bill_depth_mm', 
              'flipper_length_mm', 'body_mass_g', 'sex', 'island_id']].to_sql(
        'PENGUINS', conn, if_exists='replace', index=False)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database created successfully with {len(penguins)} penguin records.")
    print(f"Islands in database: {', '.join(islands)}")
    print(f"Penguin species in database: {', '.join(penguins['species'].unique())}")

if __name__ == "__main__":
    create_database()
