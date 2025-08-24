# 📚 PDF Reader QA with Web Search & Email (Google Gemini)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)  
![Gemini API](https://img.shields.io/badge/Google%20Gemini-API-4285F4?logo=google&logoColor=white)  
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-orange)  

An interactive **Streamlit app** that lets you upload PDFs, ask questions, and get **AI-powered answers**.  
If the answer isn’t found in your documents, it will automatically **search the web**.  
You can also **download answers as PDFs** or even **send them via email** with attachments.  

---

## ✨ Features
- 📄 Upload and index multiple PDFs  
- 🔍 Ask natural language questions about your PDFs  
- 🌐 Falls back to **web search** if context is missing  
- 🧠 Powered by **Google Gemini API** (via LangChain)  
- 📑 Save Q&A history in **SQLite database**  
- 📤 Download single or combined answers as **PDF reports**  
- 📧 Email results directly with **SMTP integration**  
- 💾 Persistent history: revisit previous Q&As anytime  
- 🎨 Simple and clean **Streamlit UI**  

---

## ⚙️ How it Works
1. **Upload PDFs** → Click **Build/Refresh Index**  
2. **Ask a Question** → Click **Ask**  
3. The app first searches inside your uploaded PDFs  
4. If no relevant answer is found → performs a **Web Search** (via SerpAPI)  
5. Displays the best possible answer on screen  
6. Options:  
   - 📥 Download as a PDF  
   - 📧 Send the result to your email (with/without PDF attachment)  

---

## 🔑 API Keys Required
You need the following API keys to run the app:  

- **Google Gemini API Key** → Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)  
- **SerpAPI Key** → Create a free account at [SerpAPI](https://serpapi.com/)  

👉 Paste your keys into the **sidebar input fields** in the app.  

---

## 📧 Email Setup
This app can send answers via **SMTP email**.  

Provide the following in the sidebar:
- **SMTP host** (default: `smtp.gmail.com`)  
- **SMTP port** (`465` for SSL or `587` for TLS)  
- **SMTP email (username)**  
- **SMTP password / App password**  
- **Recipient email**  

⚠️ **Tip for Gmail Users:**  
- Go to [Google Account Security → App Passwords](https://myaccount.google.com/apppasswords)  
- Generate a 16-character **app password**  
- Use that instead of your normal Gmail password  

---

## 🚀 Run Locally
Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/pdf-qa-gemini.git
cd pdf-qa-gemini
pip install -r requirements.txt
streamlit run app.py

## 🌍 Future Improvements
- 🔊 Add **voice query support** (speak to ask questions)  
- 📱 Build a **mobile-friendly version** (Streamlit Cloud / Flutter frontend)  
- 🧠 Improve retrieval with **advanced embeddings** (FAISS/Weaviate)  
- 🔒 Add **user authentication & profiles**  
- 📊 Create an **analytics dashboard** to view usage stats  
- 📝 Support for more file types (Word, Excel, TXT, CSV)  
- 🤝 Enable **collaborative mode** (share results with other users)  
- ☁️ Add **cloud storage support** (Google Drive, OneDrive)  

---

## 🤝 Contributing
We welcome contributions! 🎉  

1. **Fork** the repository  
2. **Create a new branch** (`feature-xyz`)  
3. **Make your changes**  
4. **Commit and push** your branch  
5. **Open a Pull Request** 🚀  

💡 Feel free to open issues for bug reports, feature requests, or suggestions.  

