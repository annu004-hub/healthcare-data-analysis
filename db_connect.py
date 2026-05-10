import mysql.connector
import pandas as pd

def get_connection():
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root123",    # your new working password
        database="healthcare"
    )
    return conn

def run_query(query):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    try:
        conn = get_connection()
        if conn.is_connected():
            print("MySQL Connected Successfully!")
            conn.close()
    except mysql.connector.Error as e:
        print("Error code   :", e.errno)
        print("Error message:", e.msg)