from .db import get_connection

def get_latest_landmarks(limit=33):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT landmark_id, x, y, z, created_at
        FROM pose_landmarks
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results