import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="pose_db",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )