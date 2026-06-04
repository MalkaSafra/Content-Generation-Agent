import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding

# ==========================================
# 1. טעינת משתני סביבה ואבטחה
# ==========================================
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise ValueError("❌ חסרים מפתחות API! ודאי שקובץ ה-.env מעודכן.")

# ==========================================
# 2. הגדרת Cohere כ"מוח" המרכזי
# ==========================================
print("🧠 מגדיר את Cohere כמנוע הראשי...")

Settings.llm = Cohere(
    api_key=COHERE_API_KEY,
    model="command-r-plus-08-2024",
    temperature=0.2
)

Settings.embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-multilingual-v3.0"
)

# ==========================================
# 3. חיבור ל-Pinecone (הזיכרון)
# ==========================================
print("🔌 מתחבר למסד הנתונים הוקטורי...")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# ==========================================
# 4. תבנית הפרומפט המנצחת (AEO, CRO & Placement)
# ==========================================
PROMPT_TEMPLATE = """
אתה קופירייטר המרות (CRO) ומומחה אופטימיזציית תוכן לבינה מלאכותית (AEO) של 'ביטוח ישיר', האחראי על תחום: {category}.
הטון שלך: "חבר חכם בגובה העיניים" – אך במקביל, אתה כותב טקסט "צפוף נתונים" שמשדר סמכותיות מוחלטת למנועי חיפוש ולמשתמשים.

מערכת המחקר זיהתה פער תוכני שעלינו לפתור (Vulnerability): 
"{vulnerability}"

הנחיית האסטרטג לפתרון (Action Plan):
"{marketing_plan}"

=== משימת הכתיבה והאסטרטגיה ===
כתוב את התוכן השיווקי הסופי המיועד לאתר, וספק המלצה אסטרטגית היכן למקם אותו.
עליך לכלול את 2 החלקים הבאים:

חלק א': המלצת הטמעה אסטרטגית (Placement Strategy)
ציין ב-1-2 משפטים היכן המיקום הנכון ביותר באתר להטמיע את התוכן הזה לאורך "מסע הלקוח". נמק את בחירתך מנקודת מבט של חווית משתמש (UX) ופסיכולוגיית צרכנים.

חלק ב': התוכן השיווקי (The Content)
עליך לעמוד בדרישות המבנה הבאות (השתמש בשיקול דעת לבחירת הכלים הנכונים):
* כותרת מגנטית וישירה (H2/H3).
* BLUF (השורה התחתונה בהתחלה): התשובה המרכזית בפסקה הראשונה.
* הוכחות מספריות (Data Anchoring).
* מבנה סריק: טבלת Markdown (אם רלוונטי להשוואה), או רשימת תבליטים (**Bold**), או שאלות ותשובות (FAQ).
* הנעה לפעולה ממוקדת עובדות (CTA).

=== חוקי אפס סובלנות להזיות ונראות (Zero Hallucination & Styling Policy) ===
- בוסס את טענותיך *אך ורק* על טקסט המקור המצורף מטה. אין להמציא נתונים או שירותים.
- אם המידע חסר בטקסט המקור, התמקד בנתונים החזקים שכן קיימים בו ללא שימוש בתארים ריקים מתוכן.
- איסור מוחלט על אימוג'ים (No Emojis): אל תשלב שום אימוג'י, אייקון או סמיילי בתוכן. הנראות חייבת להיות נקייה, תאגידית ורשמית. רשימות תבליטים יסומנו בנקודות סטנדרטיות בלבד (- או *).

[טקסט המקור לביסוס התוכן - RAG Context]:
---------------------
{context_str}
---------------------
"""

# ==========================================
# 5. פונקציית האייג'נט
# ==========================================
def run_marketing_agent(json_input_str):
    data = json.loads(json_input_str)

    category = data.get("category", "ביטוח")
    vulnerability = data.get("vulnerability", "")
    marketing_plan = data.get("action_plan", {}).get("marketing", "")

    print(f"\n🎯 האייג'נט מנתח פער בקטגוריית '{category}'...")

    search_query = f"{category}: {vulnerability} {marketing_plan}"

    retriever = index.as_retriever(similarity_top_k=4)
    nodes = retriever.retrieve(search_query)

    retrieved_context = "\n\n".join([n.get_content() for n in nodes])

    if not retrieved_context.strip():
        print("⚠️ אזהרה: לא נמצא מידע רלוונטי באינדקס לשאילתה זו.")

    final_prompt = PROMPT_TEMPLATE.format(
        category=category,
        vulnerability=vulnerability,
        marketing_plan=marketing_plan,
        context_str=retrieved_context
    )

    print("✍️ קופירייטר ה-AI (Cohere) כותב את התוכן ומגבש המלצת הטמעה...")
    messages = [ChatMessage(role="user", content=final_prompt)]
    response = Settings.llm.chat(messages)

    return response.message.content


# ==========================================
# טסט הרצה
# ==========================================
if __name__ == "__main__":
    # JSON מחמיר לבדיקת נתוני שירות (מוסכים, תביעות ומהירות דיגיטלית)
    sample_json = """
    {
      "category": "ביטוח רכב מקיף - תביעות ושירות",
      "score_after": 7.5,
      "vulnerability": "משתמשים ומנועי חיפוש לא מוצאים מידע מרגיע על מהירות הטיפול בתביעות. ישנו חשש מרכזי של הלקוח מתהליך בירוקרטי ארוך, חוסר זמינות, והמתנה ארוכה לרכב חלופי לאחר תאונה.",
      "action_plan": {
        "technical": "יצירת סכמת FAQ מובנית (Schema Markup) לטובת הופעה ב-Google AI Overviews.",
        "marketing": "להדגיש את המהירות והדיגיטליות של התהליך. להשתמש בנתוני זמנים ברורים מתוך הפוליסות, ולהפוך את 'התהליך הדיגיטלי' מתכונה יבשה לתועלת של 'אפס ניירת ומינימום כאב ראש'."
      },
      "raw_chat_logs": []
    }
    """

    try:
        result = run_marketing_agent(sample_json)
        print("\n" + "✨ " * 20)
        print("התוצר השיווקי הסופי:")
        print("✨ " * 20 + "\n")
        print(result)
    except Exception as e:
        print(f"\n❌ שגיאה במהלך ריצת האייג'נט: {e}")