import os
from dotenv import load_dotenv # השורה החדשה שהוספנו
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Settings
from llama_index.readers.web import SimpleWebPageReader
import pip_system_certs.wrapt_requests
# 1. טעינת המפתחות הסודיים מקובץ ה-.env
load_dotenv()

def main():
    print("🚀 מתחיל תהליך אינדוקס נתוני אמת...")

    # 2. אתחול Pinecone (הוא כבר ימשוך את המפתח אוטומטית מ-os.environ)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "bituach-yashir-knowledge"

    if index_name not in pc.list_indexes().names():
        print(f"יוצר אינדקס חדש ב-Pinecone בשם: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,  # המימד התואם למודל של Cohere
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    pinecone_index = pc.Index(index_name)

    # 3. הגדרת Cohere כמנוע ה-Embeddings
    embed_model = CohereEmbedding(
        cohere_api_key=os.environ["COHERE_API_KEY"],
        model_name="embed-multilingual-v3.0"
    )
    Settings.embed_model = embed_model

    # 4. שאיבת הנתונים האמיתיים מהרשת
    urls = [
        "https://www.555.co.il/financial_reports/report_30_09_24.html",
        "https://www.555.co.il/car-insurance/comprehensive.html"
    ]

    import requests
    from bs4 import BeautifulSoup
    from llama_index.core import Document

    print("🌐 שואב נתונים ישירות מהקישורים (בשיטה היציבה)...")

    # הוסיפי את ה-headers האלו לפני הלופ
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    documents = []
    for url in urls:
        try:
            # הפעם נוסיף את ה-headers לקריאה
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                doc = Document(text=text, metadata={"url": url})
                documents.append(doc)
                print(f"✅ נשאב בהצלחה: {url}")
            else:
                print(f"⚠️ אתר החזיר שגיאה {response.status_code}: {url}")

        except Exception as e:
            print(f"❌ שגיאה בשאיבת {url}: {e}")

    print(f"✅ נשאבו בהצלחה {len(documents)} עמודי אינטרנט. מעבד ושומר בענן...")

    # 5. עיבוד ושמירה במסד הנתונים
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    print("🏆 האינדוקס הושלם בהצלחה! מסד הנתונים מעודכן עם מידע אמיתי.")


if __name__ == "__main__":
    main()