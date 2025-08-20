import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

PG_CONFIG = {
    "host": "",
    "port": 5432,
    "database": "fondasikehidupandb",
    "user": "",
    "password": ""
}

QDRANT_CONFIG = {
    "host": "",
    "port": 6334,
    "api_key": "",
    "collection_name": "book"
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

pg_cursor.execute("SELECT id, title, synopsis, book_picture FROM book;")
rows = pg_cursor.fetchall()

points = []
for row in rows:
    id_pg, title, synopsis, book_picture = row
    passage = f"{title}. {synopsis}"
    embedding = model.encode(passage, normalize_embeddings=True).tolist()
    points.append(PointStruct(
        id=id_pg,
        vector=embedding,
        payload={
            "id": id_pg,
            "title": title,
            "bookPicture": book_picture
        }
    ))

qdrant.upsert(
    collection_name=QDRANT_CONFIG["collection_name"],
    points=points
)

pg_cursor.close()
pg_conn.close()

print(f"Sukses menyimpan {len(points)} item ke Qdrant!")
