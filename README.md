# Quantis

**Live → [quantis-ops.streamlit.app](https://quantis-ops.streamlit.app)**

---

I built Quantis because I kept seeing the same problem — operations teams had data, but no fast way to turn it into answers. Why did we run out of stock? Which product is bleeding the most revenue? What happens next week? Answering those questions either took days of Excel work or an expensive consultant.

Quantis does it in 30 seconds. Upload a CSV, get a full revenue loss audit, AI root-cause diagnosis, a demand forecast, and a PDF report sent straight to your inbox.

---

## What it actually does

You upload your sales data and it tells you:

- Which stockout events happened and exactly how much revenue each one cost
- Which specific products are the biggest problem (if you have product data)
- What demand is likely to look like over the next 4 weeks
- Why the stockouts are probably happening — diagnosed by Groq's LLaMA-3.3 70B
- All of the above packaged into a clean PDF and emailed to whoever needs to see it

---

## The AI part

Most AI tools use GPT-4 which gets expensive fast. I used Groq's LLaMA-3.3 70B — it's genuinely good at operations reasoning and runs on Groq's free tier. The diagnosis it produces reads like something a supply chain consultant would write, not a chatbot.

The goal was always to make enterprise-grade diagnostics accessible without the enterprise price tag.

---

## Stack

- **Streamlit** — frontend and deployment
- **Groq + LLaMA-3.3 70B** — AI root cause analysis
- **Plotly + Matplotlib** — charts and visualisations
- **SciPy** — linear regression for the demand forecast
- **ReportLab** — PDF generation
- **Gmail SMTP** — email delivery

---

## Running it locally

```bash
git clone https://github.com/AdityaSingh-ist/Quantis.git
cd Quantis
pip install -r requirements.txt
cp .env.example .env
```

Fill in your `.env`:

```
GROQ_API_KEY=get this free at console.groq.com
SENDER_EMAIL=your gmail address
SENDER_PASSWORD=16 character app password from myaccount.google.com
```

Then:

```bash
streamlit run app.py
```

---

## CSV format

| Column | Required | Example |
|---|---|---|
| date | yes | 2024-01-15 |
| inventory | yes | 0 |
| units_sold | yes | 45 |
| price_per_unit | yes | 250 |
| product | no | Widget A |

No data? There's a sample CSV download button inside the app.

---

## Built by

**Aditya Singh** — [quantis-ops.streamlit.app](https://quantis-ops.streamlit.app)

Operations × AI. Built with the belief that good diagnostics shouldn't cost a fortune.
