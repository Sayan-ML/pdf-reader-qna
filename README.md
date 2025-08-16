# 📚 PDF Reader QA with Web Search & Email (Google Gemini)

An interactive **Streamlit app** that lets you upload PDFs, ask questions, and get **AI-powered answers**.  
If the answer isn’t found in your documents, it will automatically **search the web**.  
You can **download answers as PDFs** or even **send them via email** with attachments.  

---

## ✨ Features
- 📄 Upload and index multiple PDFs  
- 🔍 Ask natural language questions about your PDFs  
- 🌐 Falls back to **web search** if context is missing  
- 🧠 Powered by **Google Gemini API** (via LangChain)  
- 📑 Save Q&A history in SQLite database  
- 📤 Download single/combined answers as **PDF reports**  
- 📧 Email results directly with **SMTP integration**  

---

## ⚙️ How it Works
1. **Upload PDFs** → Click **Build/Refresh Index**  
2. **Ask a Question** → Click **Ask**  
3. The app first searches your PDFs  
4. If no answer → performs a **Web Search** (via SerpAPI)  
5. View the answer on screen, download it as a PDF, or email it  

---

## 🔑 API Keys Required
You need the following API keys:

- **Google Gemini API Key** → Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)  
- **SerpAPI Key** → Create a free account at [SerpAPI](https://serpapi.com/)  

Paste your keys into the sidebar input fields.

---

## 📧 Email Setup
This app can send answers via **SMTP email**.  

Provide the following in the sidebar:
- **SMTP host** (default: `smtp.gmail.com`)  
- **SMTP port** (`465` or `587`)  
- **SMTP email (username)**  
- **SMTP password / App password**  
- **Recipient email**  

⚠️ **Tip for Gmail Users:**  
- Go to [Google Account Security → App Passwords](https://myaccount.google.com/apppasswords)  
- Generate a 16-character app password  
- Use that instead of your normal Gmail password  

---

## 🚀 Run Locally
Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/pdf-qa-gemini.git
cd pdf-qa-gemini
pip install -r requirements.txt
