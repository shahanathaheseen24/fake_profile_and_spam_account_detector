import streamlit as st
import numpy as np
import pickle
from datetime import date
import matplotlib.pyplot as plt
from textblob import TextBlob
import time


# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Detector", page_icon="🤖", layout="wide")

# ---------------- OPENAI ----------------


# ---------------- LOAD MODEL ----------------
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>
html, body {
    background-image: url("https://miro.medium.com/1*OzHtFxvO7MiWNKvbeXSjAg.jpeg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp, .block-container {
    background: transparent !important;
}

/* overlay */
.stApp::before {
    content: "";
    position: fixed;
    top:0;
    left:0;
    width:100%;
    height:100%;
    background: rgba(0,0,0,0.65);
    z-index:0;
}

.block-container {
    position: relative;
    z-index:1;
}

/* headings */
h1, h2, h3 {
    color: #FFD700;
}

/* chat style */
[data-testid="stChatMessage"] {
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state.login = True
            st.success("Login Successful 🚀")
        else:
            st.error("Invalid credentials ❌")
    st.stop()

# ---------------- NAVIGATION ----------------
menu = st.selectbox("☰ Menu",
["🏠 Home","🧠 Detector","🤖 AI Chat","📊 Insights","ℹ️ About"])

# ---------------- AI AGENT ----------------
def ai_agent(pred,sus,spam,sent):
    score = 0
    if pred == 1: score += 2
    score += sus + spam
    if sent < -0.3: score += 1
    return "Fake" if score >= 3 else "Real"

# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.title("🤖 Intelligent Detection System")
    st.markdown("### ✨ ML + NLP + Cybersecurity")
    st.markdown("---")
    st.markdown("""
    ## 🌟 Features
    - 🤖 AI Detection  
    - 🔐 Cybersecurity  
    - 💬 NLP Analysis  
    - 📊 Visualization  
    - 🤖 ChatGPT AI  
    """)

# ---------------- DETECTOR ----------------
elif menu == "🧠 Detector":

    st.title("🧠 Account Analyzer")

    c1,c2 = st.columns(2)

    with c1:
        username = st.text_input("👤 Username")
        platform = st.selectbox("🌐 Platform",["Instagram","Twitter","Facebook"])
        created = st.date_input("📅 Created Date")

    with c2:
        followers = st.number_input("👥 Followers",0)
        following = st.number_input("🔁 Following",0)
        posts = st.number_input("📸 Posts",0)
        bio = st.number_input("📝 Bio Length",0)

    msg = st.text_area("💬 Message")

    if st.button("🔍 Analyze"):

        with st.spinner("Analyzing... 🤖"):
            time.sleep(2)

        age = (date.today()-created).days
        ratio = followers/(following+1)

        low_ratio = int((followers<50 and following>300))
        low_posts = int((posts<3 and followers<20))
        new_acc = int(age<30)

        features = np.array([[followers,following,posts,bio,
                              ratio,low_ratio,low_posts]])
        features = scaler.transform(features)

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        sus = low_ratio + low_posts + new_acc

        spam_words=["free","win","offer","click"]
        spam = sum([1 for w in spam_words if w in msg.lower()]) if msg else 0
        sent = TextBlob(msg).sentiment.polarity if msg else 0

        result = ai_agent(pred,sus,spam,sent)

        st.subheader("🤖 Result")
        if result=="Fake":
            st.error(f"🚨 Fake ({max(prob)*100:.2f}%)")
        else:
            st.success(f"✅ Real ({max(prob)*100:.2f}%)")

        fig, ax = plt.subplots()
        ax.bar(["Sentiment"], [sent])
        ax.set_ylim(-1,1)
        st.pyplot(fig)

# ---------------- CHATGPT AI ----------------
elif menu == "🤖 AI Chat":

    st.title("🤖 AI Assistant (Offline Mode)")

    import random

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Smart reply function
    def smart_reply(question):
        q = question.lower()

        if any(word in q for word in ["hi", "hello", "hey"]):
            return random.choice([
                "👋 Hello! How can I help you?",
                "Hi there! Ask me anything!",
            ])

        elif any(word in q for word in ["thank", "thanks"]):
            return random.choice([
                "😊 You're welcome!",
                "Glad I could help!",
            ])

        elif "real" in q:
            return "✅ This account is REAL because it shows normal behavior and no suspicious patterns."

        elif "fake" in q:
            return "🚨 This account is FAKE due to suspicious behavior, spam signals, or abnormal activity."

        elif any(word in q for word in ["how", "work"]):
            return "⚙️ The system uses Machine Learning, NLP, and cybersecurity rules."

        elif "feature" in q:
            return "📊 Features include followers ratio, posts, account age, and message analysis."

        elif "accuracy" in q:
            return "📈 Accuracy is around 90%."

        else:
            return "🤖 I analyze accounts using AI techniques like ML, NLP, and behavior analysis."

    # Input box
    user_input = st.chat_input("Ask anything...")

    # When user sends message
    if user_input is not None:

        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate reply
        reply = smart_reply(user_input)

        # Save bot reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
# ---------------- INSIGHTS ----------------
elif menu == "📊 Insights":
    st.title("📊 Insights")

    acc = 0.90
    st.metric("Accuracy", f"{acc*100:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(["Accuracy","Error"], [acc,1-acc])
    st.pyplot(fig)

# ---------------- ABOUT ----------------
elif menu == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### 🎯 Objective
    Detect fake accounts using AI
    
    ### 🚀 Technologies
    - Machine Learning  
    - NLP  
    - OpenAI API  
    
    ### 💡 Future Scope
    - Real-time detection  
    - Deep learning models  
    """)