import numpy as np
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

PG_CONFIG = {
    "host": "",
    "port": 5433,
    "database": "kapsulkeasliandb",
    "user": "",
    "password": ""
}

QDRANT_CONFIG = {
    "host": "",
    "port": 6334,
    "api_key": "",
    "collection_name": "user"
}

pg_conn = psycopg2.connect(**PG_CONFIG)
pg_cursor = pg_conn.cursor()

qdrant = QdrantClient(
    host=QDRANT_CONFIG["host"],
    port=QDRANT_CONFIG["port"],
    prefer_grpc=True,
    https=False,
    api_key=QDRANT_CONFIG["api_key"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

if QDRANT_CONFIG["collection_name"] not in qdrant.get_collections().collections:
    qdrant.recreate_collection(
        collection_name=QDRANT_CONFIG["collection_name"],
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

pg_cursor.execute("SELECT id FROM users;")
rows = pg_cursor.fetchall()

points = []
for row in rows:
    id_pg, = row
    initial_vector = np.zeros(384, dtype=np.float32).tolist()
    points.append(PointStruct(
        id=id_pg,
        vector=initial_vector,
        payload={"book_count": 0, "short_term_ids": []}
    ))

try:
    qdrant.upsert(
        collection_name=QDRANT_CONFIG["collection_name"],
        points=points
    )
    print(f"Sukses menyimpan {len(points)} item ke Qdrant!")
except Exception as e:
    print("Gagal insert ke Qdrant:", e)

pg_cursor.close()
pg_conn.close()

print(f"Sukses menyimpan {len(points)} item ke Qdrant!")
