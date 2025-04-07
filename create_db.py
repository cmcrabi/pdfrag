import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection parameters from environment variables
params = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT")
}

def create_database():
    try:
        # Connect to the default 'postgres' database first
        conn = psycopg2.connect(
            database="postgres",
            **params
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Create a cursor
        cur = conn.cursor()
        
        # Create the new database
        db_name = os.getenv("DB_NAME", "pdf_rag_db")
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cur.execute(f"CREATE DATABASE {db_name}")
        
        print(f"Database '{db_name}' created successfully!")
        
        # Close the connection to postgres database
        cur.close()
        conn.close()
        
        # Now connect to our new database and create the vector extension
        conn = psycopg2.connect(
            database=db_name,
            **params
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Create the vector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("Vector extension created successfully!")
        
        # Close all connections
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

if __name__ == "__main__":
    create_database() 