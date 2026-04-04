import streamlit as st
import pandas as pd
import os
import requests
import io
import smtplib
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from scipy.stats import linregress
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime

import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT

# ════════════════════════════════════════════════════════
# FIX 1 — Page title & icon (browser tab)
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quantis — Precision Operations Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════
# SECURITY — credentials via environment variables
# Create a .env file locally with these keys.
# Never hardcode secrets in source code.
# ════════════════════════════════════════════════════════
SENDER_EMAIL    = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")


# ════════════════════════════════════════════════════════
# SAMPLE CSV — lets users try the app without their own data
# ════════════════════════════════════════════════════════
SAMPLE_CSV = """date,inventory,units_sold,price_per_unit,product
2024-01-01,150,45,250,Widget A
2024-01-02,0,30,250,Widget A
2024-01-03,200,60,180,Widget B
2024-01-04,0,25,180,Widget B
2024-01-05,300,90,320,Widget C
2024-01-06,120,40,250,Widget A
2024-01-07,0,55,320,Widget C
2024-01-08,80,35,180,Widget B
2024-01-09,0,45,250,Widget A
2024-01-10,250,70,320,Widget C
2024-01-11,0,30,180,Widget B
2024-01-12,190,65,250,Widget A
2024-01-13,0,50,320,Widget C
2024-01-14,110,40,180,Widget B
2024-01-15,0,35,250,Widget A
2024-01-16,280,85,320,Widget C
2024-01-17,60,30,180,Widget B
2024-01-18,0,55,250,Widget A
2024-01-19,320,95,320,Widget C
2024-01-20,0,40,180,Widget B
2024-01-21,170,50,250,Widget A
2024-01-22,0,65,320,Widget C
2024-01-23,90,35,180,Widget B
2024-01-24,0,45,250,Widget A
2024-01-25,340,100,320,Widget C
2024-01-26,130,42,180,Widget B
2024-01-27,0,58,250,Widget A
2024-01-28,0,72,320,Widget C
2024-01-29,75,28,180,Widget B
2024-01-30,0,48,250,Widget A"""



# ════════════════════════════════════════════════════════
# FIX 2 — CORE ANALYSIS with product-level support
# ════════════════════════════════════════════════════════
def analyze_operations_metrics(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    stockouts = df[(df['inventory'] == 0) & (df['units_sold'] > 0)]
    avg_price  = df['price_per_unit'].mean()
    revenue_loss = stockouts['units_sold'].sum() * avg_price

    df['week'] = df['date'].dt.isocalendar().week
    weekly_sales = df.groupby('week')['units_sold'].sum()
    peak_week = int(weekly_sales.idxmax())

    # ── Product-level (only when 'product' column exists) ──────────────────
    has_product   = 'product' in df.columns
    product_table = None
    worst_product = worst_rate = worst_loss = None

    if has_product:
        rows = []
        for prod, grp in df.groupby('product'):
            so_mask = (grp['inventory'] == 0) & (grp['units_sold'] > 0)
            so_count  = so_mask.sum()
            so_rate   = so_count / len(grp) * 100
            rev_lost  = grp.loc[so_mask, 'units_sold'].sum() * grp['price_per_unit'].mean()
            rows.append({
                'Product':         prod,
                'Total Records':   len(grp),
                'Stockout Events': int(so_count),
                'Stockout Rate %': round(so_rate, 1),
                'Revenue Lost':    rev_lost,
            })
        product_table = (pd.DataFrame(rows)
                           .sort_values('Revenue Lost', ascending=False)
                           .reset_index(drop=True))
        worst_row     = product_table.loc[product_table['Stockout Rate %'].idxmax()]
        worst_product = worst_row['Product']
        worst_rate    = worst_row['Stockout Rate %']
        worst_loss    = worst_row['Revenue Lost']

    return {
        'total_rows':       len(df),
        'stockout_count':   int(len(stockouts)),
        'revenue_loss':     f"₹{revenue_loss:,.0f}",
        'revenue_loss_raw': revenue_loss,
        'stockout_rate':    f"{len(stockouts)/len(df):.1%}",
        'stockout_pct':     len(stockouts)/len(df)*100,
        'peak_week':        peak_week,
        'df':               df,
        'has_product':      has_product,
        'product_table':    product_table,
        'worst_product':    worst_product,
        'worst_rate':       f"{worst_rate:.0f}%" if worst_rate is not None else None,
        'worst_loss':       f"₹{worst_loss:,.0f}" if worst_loss is not None else None,
    }


# ════════════════════════════════════════════════════════
# GROQ AI
# ════════════════════════════════════════════════════════
def groq_ai_diagnosis(metrics, df_sample):
    if not GROQ_API_KEY:
        return "ERROR: GROQ_API_KEY not set. Add it to your .env file."
    model   = "llama-3.3-70b-versatile"
    sample  = df_sample.head(5).to_csv(index=False)
    product_note = (f"\n- Worst product: {metrics['worst_product']} "
                    f"({metrics['worst_rate']} stockout rate, {metrics['worst_loss']} lost)"
                    if metrics['worst_product'] else "")
    prompt = f"""You are a senior supply-chain operations consultant.
Dataset summary:
- Total records : {metrics['total_rows']}
- Stockout events: {metrics['stockout_count']} ({metrics['stockout_rate']}){product_note}
- Revenue lost  : {metrics['revenue_loss']}
- Peak week     : Week {metrics['peak_week']}
Sample data:
{sample}
Respond in exactly this format, plain text only, no markdown:

ROOT CAUSES:
- [Cause 1]
- [Cause 2]
- [Cause 3]

IMMEDIATE FIXES (30 days):
- [Fix 1]
- [Fix 2]

STRATEGIC RECOMMENDATION:
[One paragraph, max 2 sentences]"""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 400},
            timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "ERROR: Request timed out."
    except Exception as e:
        return f"ERROR: {str(e)}"


def fmt_ai(raw):
    parts = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.endswith(':') and not line.startswith(('-','•')):
            parts.append(f'<div class="ai-head">{line}</div>')
        elif line.startswith(('-','•')):
            parts.append(f'<div class="ai-bullet"><span class="ai-dot">—</span>'
                         f'<span>{line[1:].strip()}</span></div>')
        else:
            parts.append(f'<div class="ai-para">{line}</div>')
    return '\n'.join(parts)


# ════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════
def weekly_demand_chart(df):
    df = df.copy()
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    w = df.groupby('week')['units_sold'].sum().reset_index()
    pk = int(w.loc[w['units_sold'].idxmax(), 'week'])
    w['color'] = w['week'].apply(lambda x: '#ede9e3' if x == pk else '#2a2a2a')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=w['week'], y=w['units_sold'],
                         marker_color=w['color'], marker_line_width=0,
                         hovertemplate='<b>Week %{x}</b><br>Units: %{y:,}<extra></extra>'))
    pv = int(w.loc[w['week']==pk, 'units_sold'].values[0])
    fig.add_annotation(x=pk, y=pv, text=f"PEAK — W{pk}",
                       showarrow=True, arrowhead=0, arrowcolor='#ede9e3', arrowwidth=1,
                       font=dict(family='monospace', size=11, color='#ede9e3'),
                       bgcolor='#1a1a1a', borderpad=6, yshift=10)
    fig.update_layout(
        paper_bgcolor='#080808', plot_bgcolor='#0d0d0d',
        font=dict(family='monospace', color='#aaa', size=11),
        xaxis=dict(title='WEEK', tickprefix='W', gridcolor='#1e1e1e', zeroline=False,
                   title_font=dict(size=11, color='#aaa'), tickfont=dict(color='#aaa')),
        yaxis=dict(title='UNITS SOLD', gridcolor='#1e1e1e', zeroline=False,
                   title_font=dict(size=11, color='#aaa'), tickfont=dict(color='#aaa')),
        margin=dict(l=50,r=30,t=40,b=50), showlegend=False, bargap=0.35, height=340)
    return fig, pk


# ════════════════════════════════════════════════════════
# EMAIL
# ════════════════════════════════════════════════════════
def build_email_html(metrics, ai_text, email_to):
    rev = metrics['revenue_loss'].replace('₹','Rs. ')
    ts  = datetime.now().strftime('%d %B %Y, %H:%M IST')
    ai_sec = ""
    if ai_text and not ai_text.startswith("ERROR"):
        lines = ""
        for line in ai_text.split('\n'):
            line = line.strip()
            if not line: lines += "<br>"
            elif line.endswith(':') and not line.startswith(('-','•')):
                lines += f'<p style="font-family:monospace;font-size:11px;color:#2e7d52;font-weight:bold;letter-spacing:2px;margin:16px 0 4px;border-bottom:1px solid #ddd;padding-bottom:4px;">{line}</p>'
            elif line.startswith(('-','•')):
                lines += f'<p style="font-family:monospace;font-size:12px;color:#333;margin:4px 0 4px 12px;">&#8212; {line[1:].strip()}</p>'
            else:
                lines += f'<p style="font-family:monospace;font-size:12px;color:#555;margin:8px 0;">{line}</p>'
        ai_sec = f'<tr><td style="padding:24px 32px 0;"><p style="font-family:Arial;font-size:11px;font-weight:bold;letter-spacing:2px;color:#fff;background:#111;padding:8px 12px;margin:0 0 16px;">GROQ AI ROOT CAUSE ANALYSIS — LLaMA-3.3 70B</p>{lines}</td></tr>'

    worst_sec = ""
    if metrics['worst_product']:
        wl = metrics['worst_loss'].replace('₹','Rs. ')
        worst_sec = f'<tr><td style="padding:12px 32px 0;"><p style="margin:0;padding:12px 16px;background:#fff5f5;border-left:3px solid #d64545;font-size:12px;color:#d64545;font-family:monospace;">⚠ WORST OFFENDER: {metrics["worst_product"]} — {metrics["worst_rate"]} stockout rate — {wl} lost</p></td></tr>'

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f0f0f0;font-family:Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f0f0;padding:32px 0;">
<tr><td align="center">
<table width="620" cellpadding="0" cellspacing="0" style="background:#fff;border:1px solid #ddd;">
  <tr><td style="background:#0d0d0d;padding:20px 32px;">
    <p style="font-family:monospace;font-size:11px;color:#888;margin:0;letter-spacing:2px;">QUANTIS — PRECISION OPERATIONS INTELLIGENCE v3.0</p>
    <p style="font-family:Arial;font-size:22px;font-weight:bold;color:#ede9e3;margin:8px 0 0;">🚨 Revenue Loss Alert</p>
  </td></tr>
  <tr><td style="background:#fff5f5;border-left:4px solid #d64545;padding:16px 32px;">
    <p style="margin:0;font-size:13px;color:#d64545;font-weight:bold;">CRITICAL: {metrics['stockout_rate']} stockout rate — {metrics['stockout_count']} events — {rev} estimated loss</p>
    <p style="margin:4px 0 0;font-size:11px;color:#888;">Week {metrics['peak_week']} is highest risk · {ts}</p>
  </td></tr>
  <tr><td style="padding:24px 32px 0;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="25%" style="padding:0 6px 0 0;"><div style="border:1px solid #eee;padding:14px 12px;text-align:center;"><p style="margin:0;font-size:9px;color:#999;font-family:monospace;letter-spacing:2px;">RECORDS</p><p style="margin:4px 0 0;font-size:22px;font-weight:bold;color:#111;">{metrics['total_rows']}</p></div></td>
      <td width="25%" style="padding:0 6px;"><div style="border:1px solid #fcc;background:#fff5f5;padding:14px 12px;text-align:center;"><p style="margin:0;font-size:9px;color:#d64545;font-family:monospace;letter-spacing:2px;">STOCKOUTS</p><p style="margin:4px 0 0;font-size:22px;font-weight:bold;color:#d64545;">{metrics['stockout_count']}</p></div></td>
      <td width="25%" style="padding:0 6px;"><div style="border:1px solid #fcc;background:#fff5f5;padding:14px 12px;text-align:center;"><p style="margin:0;font-size:9px;color:#d64545;font-family:monospace;letter-spacing:2px;">RATE</p><p style="margin:4px 0 0;font-size:22px;font-weight:bold;color:#d64545;">{metrics['stockout_rate']}</p></div></td>
      <td width="25%" style="padding:0 0 0 6px;"><div style="border:1px solid #111;background:#111;padding:14px 12px;text-align:center;"><p style="margin:0;font-size:9px;color:#888;font-family:monospace;letter-spacing:2px;">LOST REVENUE</p><p style="margin:4px 0 0;font-size:18px;font-weight:bold;color:#ede9e3;">{rev}</p></div></td>
    </tr></table>
  </td></tr>
  {worst_sec}
  {ai_sec}
  <tr><td style="padding:24px 32px 0;"><p style="font-family:Arial;font-size:11px;font-weight:bold;letter-spacing:2px;color:#fff;background:#111;padding:8px 12px;margin:0 0 12px;">STANDARD RECOMMENDATIONS</p>
    <p style="font-size:12px;color:#333;margin:0 0 8px;"><b>01 IMMEDIATE —</b> Increase safety stock for Week {metrics['peak_week']}. Pre-position 2 weeks prior.</p>
    <p style="font-size:12px;color:#333;margin:0 0 8px;"><b>02 30 DAYS —</b> ABC/XYZ segmentation on top stockout SKUs.</p>
    <p style="font-size:12px;color:#333;margin:0 0 8px;"><b>03 90 DAYS —</b> Rolling 12-week demand forecast reduces stockout probability 60–70%.</p>
  </td></tr>
  <tr><td style="padding:20px 32px;"><p style="margin:0;padding:12px 16px;background:#f5f5f5;border:1px solid #ddd;font-size:11px;color:#666;font-family:monospace;">📎 Full 4-page diagnostic PDF attached.</p></td></tr>
  <tr><td style="background:#0d0d0d;padding:16px 32px;">
    <p style="margin:0;font-family:monospace;font-size:9px;color:#444;letter-spacing:1px;">QUANTIS — Precision Operations Intelligence v3.0 · {ts}</p>
    <p style="margin:4px 0 0;font-family:monospace;font-size:9px;color:#333;">CONFIDENTIAL — Prepared for {email_to}</p>
  </td></tr>
</table></td></tr></table></body></html>"""


def send_real_email(email_to, metrics, pdf_data, ai_insights, pdf_fname):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return False, "Email credentials not configured. Set SENDER_EMAIL and SENDER_PASSWORD in your .env file."
    try:
        msg = MIMEMultipart('mixed')
        msg['From']    = f"Quantis Operations <{SENDER_EMAIL}>"
        msg['To']      = email_to
        msg['Subject'] = f"🚨 Quantis Alert: {metrics['revenue_loss']} Revenue Loss — Week {metrics['peak_week']} Critical"
        msg.attach(MIMEText(build_email_html(metrics, ai_insights, email_to), 'html', 'utf-8'))
        pdf_part = MIMEApplication(pdf_data, _subtype='pdf')
        pdf_part.add_header('Content-Disposition', 'attachment', filename=pdf_fname)
        msg.attach(pdf_part)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email_to, msg.as_string())
        return True, f"Delivered to {email_to}"
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail auth failed. Check App Password."
    except Exception as e:
        return False, str(e)


# ════════════════════════════════════════════════════════
# PDF REPORT
# ════════════════════════════════════════════════════════
def generate_pdf_report(metrics, df, email, ai_text=""):
    buf = io.BytesIO()
    LM = RM = 2.2*cm
    W  = A4[0] - LM - RM
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=LM, rightMargin=RM,
                            topMargin=2*cm, bottomMargin=2*cm)
    INK   = colors.HexColor('#111111')
    INK2  = colors.HexColor('#444444')
    INK3  = colors.HexColor('#777777')
    RULE  = colors.HexColor('#dddddd')
    HDR   = colors.HexColor('#0d0d0d')
    HTXT  = colors.HexColor('#ede9e3')
    RED   = colors.HexColor('#d64545')
    GRN   = colors.HexColor('#2e7d52')
    AMB   = colors.HexColor('#b45309')
    ALT   = colors.HexColor('#f5f5f5')

    def S(fn='Helvetica',sz=9,col=INK,al=TA_LEFT,lm=1.5,sb=0,sa=2):
        return ParagraphStyle('',fontName=fn,fontSize=sz,leading=sz*lm,textColor=col,alignment=al,spaceBefore=sb,spaceAfter=sa)
    def H(sz=10,col=INK,al=TA_LEFT): return S('Helvetica-Bold',sz,col,al,1.25,0,4)
    def M(sz=8,col=INK2,bold=False,al=TA_LEFT): return S('Courier-Bold' if bold else 'Courier',sz,col,al,1.5)

    def sec_head(lbl):
        t=Table([[Paragraph(lbl,S('Helvetica-Bold',8,HTXT,TA_LEFT,1.2))]],colWidths=[W])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HDR),
                               ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                               ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10)]))
        return t

    def footer():
        t=Table([[Paragraph('QUANTIS — Precision Operations Intelligence | v3.0',M(7,INK3)),
                  Paragraph('CONFIDENTIAL',M(7,INK3,al=TA_RIGHT))]],colWidths=[W*.7,W*.3])
        t.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0),
                               ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
        return t

    rev = metrics['revenue_loss'].replace('₹','Rs. ')
    story=[]

    # PAGE 1
    ch=Table([[Paragraph('QUANTIS',S('Helvetica-Bold',11,HTXT)),
               Paragraph('PRECISION OPERATIONS INTELLIGENCE — v3.0',S('Helvetica',8,HTXT,TA_RIGHT))]],
             colWidths=[W*.5,W*.5])
    ch.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HDR),
                            ('TOPPADDING',(0,0),(-1,-1),14),('BOTTOMPADDING',(0,0),(-1,-1),14),
                            ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14)]))
    story+=[ch,Spacer(1,1.0*cm),Paragraph('REVENUE',H(44,INK)),Paragraph('DIAGNOSTICS',H(44,INK3)),
            Paragraph('Operational Intelligence Report',S('Helvetica',13,INK2)),
            Spacer(1,.5*cm),HRFlowable(width=W,thickness=1.5,color=INK),Spacer(1,.6*cm)]

    for k,v in [('PREPARED FOR',email),('GENERATED',datetime.now().strftime('%d %B %Y, %H:%M IST')),
                ('CLASSIFICATION','CONFIDENTIAL — INTERNAL USE ONLY'),
                ('PLATFORM','Quantis Precision Operations Intelligence v3.0'),
                ('AI ENGINE','Groq LLaMA-3.3 70B Versatile')]:
        mt=Table([[Paragraph(k,M(8,INK3,True)),Paragraph(v,M(8,INK))]],colWidths=[W*.32,W*.68])
        mt.setStyle(TableStyle([('LINEBELOW',(0,0),(-1,-1),.5,RULE),
                                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                                ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
        story.append(mt)
    story.append(Spacer(1,2*cm))

    kd=[[Paragraph('RECORDS',M(7,INK3,True)),Paragraph(str(metrics['total_rows']),H(18,INK)),Paragraph('',M(7))],
        [Paragraph('STOCKOUT EVENTS',M(7,RED,True)),Paragraph(str(metrics['stockout_count']),H(18,RED)),Paragraph('CRITICAL',M(7,RED))],
        [Paragraph('STOCKOUT RATE',M(7,RED,True)),Paragraph(metrics['stockout_rate'],H(18,RED)),Paragraph('ABOVE THRESHOLD',M(7,RED))],
        [Paragraph('REVENUE LOST',M(7,INK3,True)),Paragraph(rev,H(16,INK)),Paragraph('RECOVERABLE',M(7,GRN))]]
    if metrics['worst_product']:
        wl=metrics['worst_loss'].replace('₹','Rs. ')
        kd.append([Paragraph('WORST PRODUCT',M(7,RED,True)),Paragraph(metrics['worst_product'],H(14,INK)),Paragraph(metrics['worst_rate'],M(7,RED))])
        kd.append([Paragraph('PRODUCT LOSS',M(7,RED,True)),Paragraph(wl,H(12,INK)),Paragraph('EMERGENCY',M(7,RED))])

    tt=Table([[row[0],row[1],row[2]] for row in kd],colWidths=[W*.32,W*.40,W*.28])
    ts_style=[('BOX',(0,0),(-1,-1),.5,RULE),('INNERGRID',(0,0),(-1,-1),.3,RULE),
              ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
              ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
              ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
              ('BACKGROUND',(0,1),(2,2),colors.HexColor('#fff5f5')),
              ('BACKGROUND',(0,3),(2,3),colors.HexColor('#f0faf5'))]
    tt.setStyle(TableStyle(ts_style))
    story+=[tt,Spacer(1,1*cm),HRFlowable(width=W,thickness=.5,color=RULE),Spacer(1,.3*cm),footer()]

    # PAGE 2 — CHARTS
    story.append(PageBreak())
    story.append(sec_head('SECTION 2 — DATA VISUALISATIONS'))
    story.append(Spacer(1,.5*cm))

    df2=df.copy(); df2['week']=df2['date'].dt.isocalendar().week.astype(int)
    wk=df2.groupby('week').agg(units_sold=('units_sold','sum'),stockouts=('inventory',lambda x:(x==0).sum())).reset_index()
    pk=int(wk.loc[wk['units_sold'].idxmax(),'week'])

    story+=[Paragraph('Chart 1 — Weekly Units Sold',H(10,INK)),
            Paragraph(f'Highlighted bar = peak demand (Week {pk}).',S('Helvetica',8,INK2)),Spacer(1,.25*cm)]
    def c_bar(plt):
        fig,ax=plt.subplots(figsize=(8,2.8)); fig.patch.set_facecolor('white'); ax.set_facecolor('#f9f9f9')
        bc=['#111111' if w==pk else '#cccccc' for w in wk['week']]
        ax.bar(wk['week'],wk['units_sold'],color=bc,width=.6,zorder=3)
        ax.axhline(wk['units_sold'].mean(),color='#d64545',linestyle='--',linewidth=1,label='Avg',zorder=4)
        pv=int(wk.loc[wk['week']==pk,'units_sold'].values[0])
        ax.annotate(f'PEAK W{pk}\n{pv:,}',xy=(pk,pv),xytext=(pk+1.5,pv*.92),fontsize=7,color='#111',arrowprops=dict(arrowstyle='->',color='#111',lw=.8))
        ax.set_xlabel('Week',fontsize=8,color='#444'); ax.set_ylabel('Units Sold',fontsize=8,color='#444')
        ax.tick_params(labelsize=7,colors='#444'); ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#ccc'); ax.grid(axis='y',color='#e5e5e5',linewidth=.5,zorder=0)
        ax.legend(fontsize=7,framealpha=0); fig.tight_layout(); return fig
    b1=io.BytesIO(); f1=c_bar(plt); f1.savefig(b1,format='png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close(f1); b1.seek(0)
    story+=[RLImage(b1,width=W,height=W*.35),Spacer(1,.6*cm)]

    sn=metrics['stockout_count']; hn=metrics['total_rows']-sn
    story+=[Paragraph('Chart 2 — Inventory Health Distribution',H(10,INK)),Spacer(1,.25*cm)]
    def c_pie(plt):
        fig,axes=plt.subplots(1,2,figsize=(8,2.6)); fig.patch.set_facecolor('white')
        ax=axes[0]; ax.set_facecolor('white')
        _,t,at=ax.pie([sn,hn],labels=['Stockout','Healthy'],colors=['#d64545','#2e7d52'],
                      autopct='%1.1f%%',startangle=140,textprops={'fontsize':8},
                      wedgeprops={'linewidth':1,'edgecolor':'white'})
        for a in at: a.set_color('white'); a.set_fontweight('bold')
        ax.set_title('Stockout vs Healthy',fontsize=9,color='#111',pad=8)
        ax2=axes[1]; ax2.set_facecolor('#f9f9f9')
        sc=['#d64545' if v>wk['stockouts'].mean() else '#f0a0a0' for v in wk['stockouts']]
        ax2.bar(wk['week'],wk['stockouts'],color=sc,width=.6,zorder=3)
        ax2.axhline(wk['stockouts'].mean(),color='#444',linestyle='--',linewidth=.8,label='Avg')
        ax2.set_xlabel('Week',fontsize=8,color='#444'); ax2.set_ylabel('Stockout Count',fontsize=8,color='#444')
        ax2.set_title('Weekly Stockout Events',fontsize=9,color='#111')
        ax2.tick_params(labelsize=7,colors='#444'); ax2.spines[['top','right']].set_visible(False)
        ax2.spines[['left','bottom']].set_color('#ccc'); ax2.grid(axis='y',color='#e5e5e5',linewidth=.5,zorder=0)
        ax2.legend(fontsize=7,framealpha=0); fig.tight_layout(); return fig
    b2=io.BytesIO(); f2=c_pie(plt); f2.savefig(b2,format='png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close(f2); b2.seek(0)
    story+=[RLImage(b2,width=W,height=W*.33),Spacer(1,.6*cm)]

    story+=[Paragraph('Chart 3 — Cumulative Revenue Loss',H(10,INK)),Spacer(1,.25*cm)]
    df3=df2.copy().sort_values('date')
    ap=df['price_per_unit'].mean() if 'price_per_unit' in df.columns else 1
    df3['is_so']=(df3['inventory']==0)&(df3['units_sold']>0)
    df3['rl']=df3.apply(lambda r:r['units_sold']*ap if r['is_so'] else 0,axis=1)
    df3['cl']=df3['rl'].cumsum()/1e5; df3=df3.reset_index(drop=True)
    def c_line(plt):
        fig,ax=plt.subplots(figsize=(8,2.6)); fig.patch.set_facecolor('white'); ax.set_facecolor('#f9f9f9')
        ax.fill_between(df3.index,df3['cl'],alpha=.12,color='#d64545')
        ax.plot(df3.index,df3['cl'],color='#d64545',linewidth=1.5,zorder=3)
        ax.set_xlabel('Record Index',fontsize=8,color='#444'); ax.set_ylabel('Loss (Rs. Lakhs)',fontsize=8,color='#444')
        ax.tick_params(labelsize=7,colors='#444'); ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#ccc'); ax.grid(color='#e5e5e5',linewidth=.5,zorder=0)
        fl=df3['cl'].iloc[-1]
        ax.annotate(f'Total: Rs.{fl:.1f}L',xy=(len(df3)-1,fl),xytext=(len(df3)*.72,fl*.8),
                    fontsize=7.5,color='#d64545',fontweight='bold',arrowprops=dict(arrowstyle='->',color='#d64545',lw=.8))
        fig.tight_layout(); return fig
    b3=io.BytesIO(); f3=c_line(plt); f3.savefig(b3,format='png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close(f3); b3.seek(0)
    story+=[RLImage(b3,width=W,height=W*.33),Spacer(1,.4*cm),HRFlowable(width=W,thickness=.5,color=RULE),Spacer(1,.3*cm),footer()]

    # PAGE 3 — KPI TABLE
    story.append(PageBreak())
    story.append(sec_head('SECTION 3 — EXECUTIVE SUMMARY & RECOMMENDATIONS'))
    story.append(Spacer(1,.5*cm))
    kd2=[
        [Paragraph('METRIC',H(8,HTXT)),Paragraph('VALUE',H(8,HTXT)),Paragraph('BENCHMARK',H(8,HTXT)),Paragraph('STATUS',H(8,HTXT))],
        [Paragraph('Records Analyzed',S(sz=9)),Paragraph(str(metrics['total_rows']),H(9)),Paragraph('—',S(sz=8,col=INK3)),Paragraph('COMPLETE',S('Helvetica-Bold',8,GRN))],
        [Paragraph('Stockout Events',S(sz=9)),Paragraph(str(metrics['stockout_count']),H(9)),Paragraph('<5% of records',S(sz=8,col=INK3)),Paragraph('CRITICAL',S('Helvetica-Bold',8,RED))],
        [Paragraph('Stockout Rate',S(sz=9)),Paragraph(metrics['stockout_rate'],H(9)),Paragraph('<5%',S(sz=8,col=INK3)),Paragraph('ABOVE THRESHOLD',S('Helvetica-Bold',8,RED))],
        [Paragraph('Revenue Lost',S(sz=9)),Paragraph(rev,H(11)),Paragraph('Rs. 0 (target)',S(sz=8,col=INK3)),Paragraph('RECOVERABLE',S('Helvetica-Bold',8,GRN))],
        [Paragraph('Peak Risk Week',S(sz=9)),Paragraph(f'Week {metrics["peak_week"]}',H(9)),Paragraph('No peak',S(sz=8,col=INK3)),Paragraph('PRIORITY ACTION',S('Helvetica-Bold',8,AMB))],
    ]
    if metrics['worst_product']:
        kd2.append([Paragraph('Worst Product',S(sz=9)),Paragraph(metrics['worst_product'],H(9)),Paragraph('0% target',S(sz=8,col=INK3)),Paragraph('EMERGENCY',S('Helvetica-Bold',8,RED))])
    kt=Table(kd2,colWidths=[W*.3,W*.22,W*.26,W*.22])
    kt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HDR),
                            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,ALT]),
                            ('BOX',(0,0),(-1,-1),.5,RULE),('INNERGRID',(0,0),(-1,-1),.3,RULE),
                            ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                            ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
                            ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story+=[kt,Spacer(1,.8*cm),Paragraph('RECOMMENDATIONS',H(9,INK)),HRFlowable(width=W,thickness=.5,color=RULE),Spacer(1,.3*cm)]
    for num,hor,txt in [('01','IMMEDIATE',f'Increase safety stock for Week {metrics["peak_week"]}. Pre-position 2 weeks prior.'),
                        ('02','30 DAYS','ABC/XYZ segmentation on top stockout SKUs. A-class: daily replenishment review.'),
                        ('03','90 DAYS','Rolling 12-week demand forecast reduces stockout probability 60-70%.')]:
        r=Table([[Paragraph(num,H(13,INK)),Paragraph(hor,M(7,INK3,True)),Paragraph(txt,S(sz=9))]],
                colWidths=[.9*cm,1.6*cm,W-2.5*cm])
        r.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                               ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                               ('LINEABOVE',(0,0),(-1,0),.4,RULE)]))
        story.append(r)
    story+=[Spacer(1,.6*cm),HRFlowable(width=W,thickness=.5,color=RULE),Spacer(1,.3*cm),footer()]

    # PAGE 4 — AI
    story.append(PageBreak())
    story.append(sec_head('SECTION 4 — GROQ AI ROOT CAUSE ANALYSIS'))
    story+=[Spacer(1,.3*cm),Paragraph(f'Model: LLaMA-3.3 70B via Groq Cloud  |  {datetime.now().strftime("%d %b %Y, %H:%M IST")}',M(7,INK3)),Spacer(1,.5*cm)]
    if ai_text and not ai_text.startswith("ERROR"):
        for line in ai_text.split('\n'):
            line=line.strip()
            if not line: story.append(Spacer(1,.2*cm)); continue
            if line.endswith(':') and not line.startswith(('-','•')):
                story+=[Spacer(1,.15*cm),Paragraph(line,H(10,INK)),HRFlowable(width=W,thickness=.4,color=RULE),Spacer(1,.15*cm)]
            elif line.startswith(('-','•')):
                bt=Table([[Paragraph('•',H(10,RED)),Paragraph(line.lstrip('-•').strip(),S(sz=9))]],colWidths=[.5*cm,W-.5*cm])
                bt.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
                story.append(bt)
            else:
                bx=Table([[Paragraph(line,S(sz=9,col=INK))]],colWidths=[W])
                bx.setStyle(TableStyle([('BOX',(0,0),(-1,-1),.5,RULE),('BACKGROUND',(0,0),(-1,-1),ALT),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12)]))
                story+=[Spacer(1,.1*cm),bx,Spacer(1,.1*cm)]
    else:
        story.append(Paragraph('AI diagnosis not generated.',S(sz=9,col=INK2)))
    story+=[Spacer(1,1*cm),HRFlowable(width=W,thickness=.5,color=RULE),Spacer(1,.3*cm),footer()]

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════
# STYLES
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:#080808 !important;color:#ede9e3 !important;font-family:'DM Sans',sans-serif
}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{display:none !important}
::-webkit-scrollbar{width:2px;background:#080808}::-webkit-scrollbar-thumb{background:#2a2a2a}
.block-container{max-width:1340px !important;padding:0 2.5rem 6rem !important}

/* ── HERO ── */
.eyebrow{font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.28em;color:#aaa;text-transform:uppercase;margin-bottom:2rem}
.display-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(5rem,9vw,10.5rem);line-height:.88;letter-spacing:.01em;color:#ede9e3}
.body-text{font-size:.95rem;font-weight:300;color:#aaa;line-height:1.75}
.hero-left{padding:4.5rem 3rem 4rem 0;border-right:1px solid #1c1c1c;min-height:82vh;display:flex;flex-direction:column;justify-content:space-between}
.hero-right{padding:4.5rem 0 4rem 3rem;min-height:82vh;display:flex;flex-direction:column;justify-content:space-between}
.stat-bar{border-top:1px solid #1c1c1c;padding-top:1.2rem;font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.18em;color:#aaa;text-transform:uppercase;line-height:2}
.visual-panel{flex:1;position:relative;background:#0d0d0d;border:1px solid #1a1a1a;margin-bottom:2.5rem;overflow:hidden;min-height:320px}
.visual-panel::before{content:'';position:absolute;inset:0;background-image:radial-gradient(circle,#252525 1px,transparent 1px);background-size:28px 28px}
.visual-panel::after{content:'';position:absolute;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(237,233,227,.07) 0%,transparent 65%);top:50%;left:50%;transform:translate(-50%,-50%);animation:orb 5s ease-in-out infinite}
@keyframes orb{0%,100%{opacity:.35;transform:translate(-50%,-50%) scale(1)}50%{opacity:1;transform:translate(-50%,-50%) scale(1.14)}}
.cross-h{position:absolute;left:0;right:0;height:1px;background:#181818;top:50%;z-index:2}
.cross-v{position:absolute;top:0;bottom:0;width:1px;background:#181818;left:50%;z-index:2}
.vis-label{position:absolute;font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.22em;color:#666;text-transform:uppercase;z-index:4}
.vis-label.tl{top:1.2rem;left:1.4rem}.vis-label.tr{top:1.2rem;right:1.4rem}
.vis-label.bl{bottom:1.2rem;left:1.4rem}.vis-label.br{bottom:1.2rem;right:1.4rem}
.ring-wrap{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:220px;height:220px;z-index:3}
.ring{position:absolute;border-radius:50%;border:1px solid #222}
.ring-1{inset:0;animation:cw 22s linear infinite;border-top-color:#3a3a3a}
.ring-2{inset:24px;animation:ccw 15s linear infinite;border-right-color:#2e2e2e}
.ring-3{inset:48px;animation:cw 10s linear infinite;border-bottom-color:#ede9e3;opacity:.45}
.ring-4{inset:72px;border-color:#1c1c1c}
.ring-dot{position:absolute;inset:90px;background:#ede9e3;border-radius:50%}
@keyframes cw{to{transform:rotate(360deg)}}@keyframes ccw{to{transform:rotate(-360deg)}}

/* ── DIVIDER ── */
.section-divider{border:none;border-top:1px solid #1c1c1c;margin:4rem 0}

/* ── INPUT PANELS ── */
.input-panel{background:#0d0d0d;border:1px solid #1e1e1e;border-right:none;padding:2.2rem 2.2rem 1.8rem}
.input-panel-r{background:#0d0d0d;border:1px solid #1e1e1e;padding:2.2rem 2.2rem 1.8rem}
.input-heading{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;letter-spacing:.06em;color:#ede9e3;margin-bottom:.3rem}
.input-sub{font-size:.82rem;font-weight:300;color:#aaa;margin-bottom:1.2rem;line-height:1.5}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"]>div{background:#111 !important;border:1px dashed #282828 !important;border-radius:0 !important;padding:1rem !important}
[data-testid="stFileUploader"]>div:hover{border-color:#444 !important}
[data-testid="stFileUploader"] label{color:#aaa !important;font-family:'DM Mono',monospace !important;font-size:.68rem !important;letter-spacing:.12em !important}
[data-testid="stFileUploader"] button{background:#1a1a1a !important;border:1px solid #2e2e2e !important;color:#ede9e3 !important;border-radius:0 !important;font-family:'DM Mono',monospace !important;font-size:.66rem !important;letter-spacing:.12em !important}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] input{background:#111 !important;border:1px solid #282828 !important;border-radius:0 !important;color:#ede9e3 !important;font-family:'DM Mono',monospace !important;font-size:.78rem !important;padding:.85rem 1rem !important;height:48px !important}
[data-testid="stTextInput"] input:focus{border-color:#ede9e3 !important;box-shadow:none !important}
[data-testid="stTextInput"] input::placeholder{color:#363636 !important}
[data-testid="stTextInput"] label{display:none !important}

/* ── METRICS GRID ── */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid #1e1e1e;margin-bottom:.5rem}
.kpi-grid-3{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid #1e1e1e;border-top:none;margin-bottom:3rem}
.m-card{padding:1.8rem 1.6rem;border-right:1px solid #1e1e1e;background:#0d0d0d;position:relative}
.m-card:last-child{border-right:none}
.m-card.inv{background:#ede9e3}
.m-card.inv .m-num,.m-card.inv .m-lbl{color:#080808}
.m-card.inv .m-tag,.m-card.inv .m-sub{color:#666}
.m-card.danger{background:#110808}
.m-num{font-family:'Bebas Neue',sans-serif;font-size:3rem;line-height:1;color:#ede9e3;display:block;margin-bottom:.3rem}
.m-num.sm{font-size:2rem}
.m-lbl{font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.22em;color:#bbb;text-transform:uppercase;display:block}
.m-tag{position:absolute;top:.9rem;right:.9rem;font-family:'DM Mono',monospace;font-size:.48rem;color:#555;letter-spacing:.15em;text-transform:uppercase}
.m-sub{font-family:'DM Mono',monospace;font-size:.52rem;color:#444;letter-spacing:.1em;text-transform:uppercase;margin-top:.35rem;display:block}

/* ── ALERT BAR ── */
.alert-bar{border:1px solid #2a2a2a;border-left:3px solid #ede9e3;padding:1.6rem 2rem;background:#0d0d0d;display:flex;align-items:center;gap:2.5rem;margin-bottom:3rem}
.alert-pct{font-family:'Bebas Neue',sans-serif;font-size:2.8rem;color:#ede9e3;white-space:nowrap;line-height:1}
.alert-desc{font-size:.85rem;font-weight:300;color:#bbb;line-height:1.7}

/* ── REPORT GRID ── */
.report-grid{display:grid;grid-template-columns:1fr 1fr;border:1px solid #1e1e1e;margin-bottom:3rem}
.report-l{padding:2.8rem;border-right:1px solid #1e1e1e;background:#0d0d0d}
.report-r{padding:2.8rem;background:#ede9e3}
.report-h{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:.04em;color:#ede9e3;margin-bottom:2rem;line-height:1}
.report-h-dark{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:.04em;color:#0d0d0d;margin-bottom:2rem;line-height:1}
.report-data{font-family:'DM Mono',monospace;font-size:.7rem;line-height:2.1;color:#bbb;white-space:pre-wrap}
.reco{border-top:1px solid #d5d1ca;padding:1.1rem 0;display:flex;gap:1.4rem;align-items:flex-start}
.reco-n{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:#bfbbb4;line-height:1;width:1.8rem;flex-shrink:0}
.reco-t{font-size:.82rem;font-weight:400;color:#2e2e2e;line-height:1.6;padding-top:.15rem}

/* ── PRODUCT SECTION ── */
.prod-head{background:#0d0d0d;border:1px solid #1e1e1e;padding:1.2rem 2rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:0}
.prod-title{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.05em;color:#ede9e3}
.prod-badge{font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.2em;color:#ff6b6b;background:#1a0505;border:1px solid #3a0a0a;padding:.3rem .7rem;text-transform:uppercase}

/* ── CHARTS ── */
.chart-wrap{border:1px solid #1e1e1e;background:#0d0d0d;padding:2rem 2rem 1rem;margin-bottom:3rem}
.chart-head{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.2rem;border-bottom:1px solid #1a1a1a;padding-bottom:1rem}
.chart-title{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:.05em;color:#ede9e3;line-height:1}
.chart-sub{font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.18em;color:#aaa;text-transform:uppercase;margin-top:.3rem}
.chart-badge{font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.2em;color:#ede9e3;background:#1a1a1a;border:1px solid #2a2a2a;padding:.3rem .7rem;text-transform:uppercase;white-space:nowrap}

/* ── STATUS BARS ── */
.success-bar{border:1px solid #1a3d28;border-left:3px solid #5affa0;padding:1.1rem 1.8rem;background:#0a1a12;font-family:'DM Mono',monospace;font-size:.68rem;color:#5affa0;letter-spacing:.1em;margin-bottom:1rem;line-height:1.8}
.error-bar{border:1px solid #3a0a0a;border-left:3px solid #ff6b6b;padding:1.1rem 1.8rem;background:#1a0505;font-family:'DM Mono',monospace;font-size:.68rem;color:#ff6b6b;letter-spacing:.08em;margin-bottom:1rem;line-height:1.8}

/* ── AI PANEL ── */
.ai-panel{border:1px solid #1e1e1e;background:#0a0a0a;margin-bottom:3rem;overflow:hidden}
.ai-panel-hdr{background:#111;border-bottom:1px solid #1e1e1e;padding:1.4rem 2rem;display:flex;align-items:center;gap:1.5rem}
.ai-badge{font-family:'Bebas Neue',sans-serif;font-size:.9rem;letter-spacing:.15em;color:#5affa0;background:#0d1a12;border:1px solid #1a3d28;padding:.3rem .8rem}
.ai-title{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.06em;color:#ede9e3}
.ai-model{font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.18em;color:#aaa;text-transform:uppercase;margin-left:auto}
.ai-body{padding:2rem 2.2rem}
.ai-head{font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.28em;color:#5affa0;text-transform:uppercase;margin-top:1.6rem;margin-bottom:.8rem;border-bottom:1px solid #1a3d28;padding-bottom:.5rem}
.ai-head:first-child{margin-top:0}
.ai-bullet{display:flex;gap:1rem;align-items:flex-start;margin-bottom:.6rem}
.ai-dot{font-family:'DM Mono',monospace;color:#5affa0;font-size:.75rem;flex-shrink:0;margin-top:.05rem}
.ai-bullet span:last-child{font-size:.86rem;font-weight:300;color:#bbb;line-height:1.65}
.ai-para{font-size:.86rem;font-weight:300;color:#bbb;line-height:1.7;margin-bottom:.5rem}

/* ── AWAIT ── */
.await-box{border:1px dashed #1c1c1c;padding:3.5rem 2rem;text-align:center;margin-top:2rem}
.await-lbl{font-family:'DM Mono',monospace;font-size:.62rem;color:#aaa;letter-spacing:.22em;text-transform:uppercase;margin-bottom:.8rem}
.await-big{font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#3a3a3a;letter-spacing:.08em}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{border:1px solid #1a1a1a !important}
.ds-title{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#ede9e3;letter-spacing:.04em;margin-bottom:.3rem}
.ds-sub{font-family:'DM Mono',monospace;font-size:.6rem;color:#aaa;letter-spacing:.2em;text-transform:uppercase;margin-bottom:1rem}

/* ════════════════════════════════════════════════════
   FIX 3 — ACTION BUTTONS: uniform grid, zero gap
   Use a wrapper table so all 4 cells are always equal
   ════════════════════════════════════════════════════ */
.action-grid{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  border:1px solid #1e1e1e;
  margin-bottom:2rem;
  background:#080808
}
.action-cell{
  padding:1.4rem 1.6rem 1.6rem;
  border-right:1px solid #1e1e1e;
  display:flex;flex-direction:column;gap:.7rem
}
.action-cell:last-child{border-right:none}
.action-lbl{
  font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.25em;
  color:#444;text-transform:uppercase;white-space:nowrap
}

/* Streamlit columns inside action-grid — strip all padding/gap */
[data-testid="stHorizontalBlock"]{gap:0 !important}
[data-testid="column"]{padding:0 !important;min-width:0}

/* ALL buttons: full width, zero border-radius, fixed height */
[data-testid="stButton"]>button,
[data-testid="stDownloadButton"]>button{
  width:100% !important;
  height:46px !important;
  border-radius:0 !important;
  font-family:'DM Mono',monospace !important;
  font-size:.63rem !important;
  letter-spacing:.18em !important;
  text-transform:uppercase !important;
  transition:background .2s,opacity .15s !important;
  white-space:nowrap !important;
  overflow:hidden !important;
  text-overflow:ellipsis !important
}

/* Button 1 — AI (green) */
[data-testid="stButton"]:nth-of-type(1)>button{
  background:#0a1a12 !important;color:#5affa0 !important;
  border:1px solid #1a3d28 !important
}
[data-testid="stButton"]:nth-of-type(1)>button:hover{background:#0f2a1e !important}

/* Button 2 — PDF (blue) */
[data-testid="stButton"]:nth-of-type(2)>button{
  background:#0a1220 !important;color:#60aaff !important;
  border:1px solid #1a2a3a !important
}
[data-testid="stButton"]:nth-of-type(2)>button:hover{background:#0f1a2e !important}

/* Button 3 — Send (cream) */
[data-testid="stButton"]:nth-of-type(3)>button{
  background:#ede9e3 !important;color:#080808 !important;border:none !important
}
[data-testid="stButton"]:nth-of-type(3)>button:hover{background:#d5d1ca !important}

/* Button 4 — Download (outline) */
[data-testid="stDownloadButton"]>button{
  background:#0d0d0d !important;color:#ede9e3 !important;
  border:1px solid #ede9e3 !important
}
[data-testid="stDownloadButton"]>button:hover{background:#ede9e3 !important;color:#080808 !important}

/* ── FOOTER ── */

/* sample csv download */
.sample-btn [data-testid="stDownloadButton"]>button{
  background:#0d0d0d !important;color:#aaa !important;
  border:1px dashed #2a2a2a !important;height:36px !important;
  font-size:.55rem !important;letter-spacing:.18em !important
}
.sample-btn [data-testid="stDownloadButton"]>button:hover{
  border-color:#ede9e3 !important;color:#ede9e3 !important
}
.footer{margin-top:6rem;border-top:1px solid #141414;padding-top:2rem;display:flex;justify-content:space-between;align-items:center}
.footer-brand{font-family:'Bebas Neue',sans-serif;font-size:1.1rem;color:#555;letter-spacing:.12em}
.footer-note{font-family:'DM Mono',monospace;font-size:.56rem;color:#555;letter-spacing:.15em}
</style>
""", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════
# SECURITY WARNING BANNER
# ════════════════════════════════════════════════════════
if not GROQ_API_KEY or not SENDER_EMAIL:
    missing = []
    if not GROQ_API_KEY:    missing.append('GROQ_API_KEY')
    if not SENDER_EMAIL:    missing.append('SENDER_EMAIL')
    if not SENDER_PASSWORD: missing.append('SENDER_PASSWORD')
    st.markdown(
        f'''<div class="warn-bar">
⚠ &nbsp; ENV VARS NOT SET: {" · ".join(missing)}<br>
&nbsp;&nbsp;Create a <b>.env</b> file with these keys. AI diagnosis and email will not work without them.<br>
&nbsp;&nbsp;See <b>.env.example</b> in the project folder for reference.
</div>''',
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════
hl, hr = st.columns(2)
with hl:
    st.markdown("""
    <div class="hero-left">
      <div class="eyebrow">QUANTIS &nbsp;—&nbsp; Precision Operations Intelligence Platform</div>
      <div>
        <div class="display-title">QUANTIS</div>
        <div style="font-family:'DM Mono',monospace;font-size:.9rem;letter-spacing:.2em;color:#aaa;text-transform:uppercase;margin-top:.6rem;margin-bottom:2.5rem">
          Precision Operations Intelligence
        </div>
      </div>
      <div class="body-text" style="max-width:360px">
        Upload your sales CSV. Get a complete revenue loss audit — stockouts, demand peaks,
        missed revenue — with Groq AI root-cause analysis, delivered to your inbox in seconds.
      </div>
      <div class="stat-bar">
        Diagnostic Engine v3.0 &nbsp;·&nbsp; Groq LLaMA-3.3 70B · Product-Level Analytics<br>
        Avg. revenue recovered per audit: ₹48L – ₹2.1Cr
      </div>
    </div>""", unsafe_allow_html=True)

with hr:
    st.markdown("""
    <div class="hero-right">
      <div class="visual-panel">
        <div class="cross-h"></div><div class="cross-v"></div>
        <span class="vis-label tl">DEMAND</span><span class="vis-label tr">SUPPLY</span>
        <span class="vis-label bl">FORECAST</span><span class="vis-label br">REVENUE</span>
        <div class="ring-wrap">
          <div class="ring ring-1"></div><div class="ring ring-2"></div>
          <div class="ring ring-3"></div><div class="ring ring-4"></div>
          <div class="ring-dot"></div>
        </div>
      </div>
      <div class="stat-bar">
        Real-time stockout mapping &nbsp;·&nbsp; Product-level attribution &nbsp;·&nbsp; AI root-cause analysis
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# INPUTS
# ════════════════════════════════════════════════════════
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="input-panel">
      <div class="input-heading">01 — Upload Data</div>
      <div class="input-sub">CSV with: date, inventory, units_sold, price_per_unit<br>
        Optional: <strong style="color:#ede9e3">product</strong> column for SKU-level analysis
      </div>
    </div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")

    # ── Sample CSV download ──────────────────────────────────────────────────
    st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
    st.download_button(
        label="↓ Download Sample CSV",
        data=SAMPLE_CSV,
        file_name="quantis_sample.csv",
        mime="text/csv",
        key="sample_dl",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="input-panel-r">
      <div class="input-heading">02 — Delivery Address</div>
      <div class="input-sub">Your Quantis diagnostic report will be dispatched to this inbox</div>
    </div>""", unsafe_allow_html=True)
    email = st.text_input("", placeholder="ops@company.com", label_visibility="collapsed")


# ════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ════════════════════════════════════════════════════════
if uploaded_file is not None and email:
    with open("temp_quantis.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running Quantis diagnostic..."):
        metrics = analyze_operations_metrics("temp_quantis.csv")

    df_temp = metrics['df']
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── KPI CARDS ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="m-card">
        <span class="m-tag">01 / INPUT</span>
        <span class="m-num">{metrics['total_rows']}</span>
        <span class="m-lbl">Records Analyzed</span>
        <span class="m-sub">{metrics['stockout_count']} stockout events</span>
      </div>
      <div class="m-card">
        <span class="m-tag">02 / RATE</span>
        <span class="m-num">{metrics['stockout_rate']}</span>
        <span class="m-lbl">Daily Stockout Rate</span>
        <span class="m-sub">{metrics['stockout_count']} / {metrics['total_rows']} records</span>
      </div>
      <div class="m-card inv">
        <span class="m-tag">03 / IMPACT</span>
        <span class="m-num" style="font-size:2.2rem">{metrics['revenue_loss']}</span>
        <span class="m-lbl">Lost Revenue</span>
        <span class="m-sub">Week {metrics['peak_week']} peak</span>
      </div>
      <div class="m-card">
        <span class="m-tag">04 / PEAK</span>
        <span class="m-num">W{metrics['peak_week']}</span>
        <span class="m-lbl">Highest Risk Week</span>
        <span class="m-sub">Priority action required</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── PRODUCT CARDS (only when product column present) ───────────────────
    if metrics['has_product'] and metrics['worst_product']:
        pt = metrics['product_table']
        bars_html = ""
        for _, row in pt.head(3).iterrows():
            r        = row['Stockout Rate %']
            bw       = min(int(r), 100)
            bc       = '#ff6b6b' if r > 50 else '#ffaa44' if r > 25 else '#ede9e3'
            prod     = row['Product']
            loss_fmt = f"&#8377;{row['Revenue Lost']:,.0f}"
            bars_html += (
                f'<div style="margin-bottom:1rem">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:.35rem">'
                f'<span style="font-family:monospace;font-size:.68rem;color:#ede9e3">{prod}</span>'
                f'<span style="font-family:monospace;font-size:.6rem;color:{bc}">{r:.0f}% &middot; {loss_fmt}</span>'
                f'</div>'
                f'<div style="background:#1a1a1a;height:3px">'
                f'<div style="background:{bc};height:3px;width:{bw}%"></div>'
                f'</div></div>'
            )

        wl = metrics['worst_loss']
        wr = metrics['worst_rate']
        wp = metrics['worst_product']
        st.markdown(f"""
        <div class="kpi-grid-3">
          <div class="m-card danger">
            <span class="m-tag" style="color:#ff4444">05 / WORST SKU</span>
            <span class="m-num sm" style="color:#ff6b6b">{wp}</span>
            <span class="m-lbl" style="color:#ff6b6b">Highest Stockout Product</span>
            <span class="m-sub" style="color:#ff4444">{wr} stockout rate</span>
          </div>
          <div class="m-card danger">
            <span class="m-tag" style="color:#ff4444">06 / SKU LOSS</span>
            <span class="m-num sm" style="color:#ff6b6b;font-size:1.6rem">{wl}</span>
            <span class="m-lbl" style="color:#ff6b6b">Worst Product Revenue Loss</span>
            <span class="m-sub" style="color:#ff4444">Emergency replenishment needed</span>
          </div>
          <div class="m-card" style="padding:1.6rem 1.8rem">
            <span class="m-tag">07 / TOP SKUs</span>
            <div style="font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.18em;color:#555;text-transform:uppercase;margin-bottom:1.1rem">Top Offenders by Rate</div>
            {bars_html}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # No product column — close the grid with bottom margin
        st.markdown('<div style="margin-bottom:3rem"></div>', unsafe_allow_html=True)

    # ── ALERT BAR ──────────────────────────────────────────────────────────
    extra = (f'&nbsp;&nbsp;<span style="font-family:\'DM Mono\',monospace;font-size:.72rem;'
             f'color:#ff6b6b">⚠ {metrics["worst_product"]} needs emergency restock</span>'
             if metrics['worst_product'] else "")
    st.markdown(f"""
    <div class="alert-bar">
      <div class="alert-pct">{metrics['stockout_rate']}</div>
      <div class="alert-desc">
        Stockout rate detected — demand is outpacing inventory across {metrics['stockout_count']} events.<br>
        Immediate action on Week {metrics['peak_week']} advised.{extra}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── DIAGNOSTIC SUMMARY ─────────────────────────────────────────────────
    d0 = df_temp['date'].iloc[0].strftime('%Y-%m-%d')
    d1 = df_temp['date'].iloc[-1].strftime('%Y-%m-%d')
    report_body = (
        f"DATASET       {metrics['total_rows']} transaction rows\n"
        f"PERIOD        {d0} — {d1}\n"
        f"STOCKOUTS     {metrics['stockout_count']} events ({metrics['stockout_rate']})\n"
        f"REVENUE LOSS  {metrics['revenue_loss']}\n"
        f"PEAK WEEK     Week {metrics['peak_week']}\n"
        + (f"WORST SKU     {metrics['worst_product']} ({metrics['worst_rate']})\n"
           if metrics['worst_product'] else "")
        + f"\nGENERATED     {pd.Timestamp.now().strftime('%d %b %Y, %H:%M IST')}\n"
          f"PREPARED FOR  {email}"
    )
    worst_reco = (f'<div class="reco"><div class="reco-n" style="color:#c0392b">04</div>'
                  f'<div class="reco-t"><strong>{metrics["worst_product"]} Emergency:</strong> '
                  f'Contact supplier immediately. Expedite restock — this SKU drives majority of revenue loss.</div></div>'
                  if metrics['worst_product'] else "")
    st.markdown(f"""
    <div class="report-grid">
      <div class="report-l">
        <div class="report-h">DIAGNOSTIC<br>SUMMARY</div>
        <pre class="report-data">{report_body}</pre>
      </div>
      <div class="report-r">
        <div class="report-h-dark">RECOMMENDED<br>ACTIONS</div>
        <div class="reco"><div class="reco-n">01</div><div class="reco-t">Increase safety stock for Week {metrics['peak_week']}. Pre-position inventory 2 weeks prior.</div></div>
        <div class="reco"><div class="reco-n">02</div><div class="reco-t">ABC/XYZ segmentation on top stockout SKUs. A-class: daily replenishment review.</div></div>
        <div class="reco"><div class="reco-n">03</div><div class="reco-t">Rolling 12-week demand forecast reduces stockout probability by 60–70%.</div></div>
        {worst_reco}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── PRODUCT HEATMAP + TABLE ────────────────────────────────────────────
    if metrics['has_product'] and metrics['product_table'] is not None:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        pt = metrics['product_table']
        st.markdown("""
        <div class="prod-head">
          <span class="prod-title">Product Revenue Leak Analysis</span>
          <span class="prod-badge">SKU-Level Attribution</span>
        </div>""", unsafe_allow_html=True)

        top10 = pt.head(10)
        bar_colors = ['#ff6b6b' if r > 50 else '#ffaa44' if r > 25 else '#ede9e3'
                      for r in top10['Stockout Rate %']]
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Bar(
            y=top10['Product'], x=top10['Stockout Rate %'], orientation='h',
            marker_color=bar_colors, marker_line_width=0,
            customdata=top10[['Revenue Lost', 'Stockout Events']].values,
            hovertemplate='<b>%{y}</b><br>Stockout Rate: %{x:.1f}%<br>'
                          'Revenue Lost: ₹%{customdata[0]:,.0f}<br>'
                          'Events: %{customdata[1]}<extra></extra>'
        ))
        fig_heat.update_layout(
            paper_bgcolor='#080808', plot_bgcolor='#0d0d0d',
            font=dict(family='monospace', color='#aaa', size=11),
            xaxis=dict(title='STOCKOUT RATE (%)', gridcolor='#1e1e1e', zeroline=False,
                       title_font=dict(size=10, color='#aaa'), tickfont=dict(color='#aaa')),
            yaxis=dict(gridcolor='#1e1e1e', zeroline=False, tickfont=dict(size=11, color='#ede9e3')),
            margin=dict(l=20, r=30, t=20, b=50), showlegend=False, bargap=0.3,
            height=max(260, len(top10)*38))
        st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})

        disp = pt[['Product','Stockout Events','Stockout Rate %','Revenue Lost']].copy()
        disp['Revenue Lost'] = disp['Revenue Lost'].apply(lambda x: f"₹{x:,.0f}")
        disp['Priority'] = pt['Stockout Rate %'].apply(
            lambda x: '🔴 EMERGENCY' if x > 50 else '🟡 HIGH' if x > 25 else '🟢 MONITOR')
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── CHARTS ────────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:.05em;color:#ede9e3;margin-bottom:.3rem">Data Visualisations</div>
    <div style="font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.22em;color:#aaa;text-transform:uppercase;margin-bottom:2.5rem">3 charts · Demand · Inventory health · Revenue loss</div>
    """, unsafe_allow_html=True)

    df_viz = df_temp.copy()
    df_viz['week'] = df_viz['date'].dt.isocalendar().week.astype(int)
    wkly = df_viz.groupby('week').agg(
        units_sold=('units_sold','sum'),
        stockouts=('inventory', lambda x: (x==0).sum())
    ).reset_index()
    pk_w = int(wkly.loc[wkly['units_sold'].idxmax(),'week'])

    cf, cp = weekly_demand_chart(df_temp)
    st.markdown(f"""<div class="chart-wrap"><div class="chart-head">
      <div><div class="chart-title">Chart 1 — Weekly Demand Map</div>
      <div class="chart-sub">Units sold per week · Highlighted = peak demand</div></div>
      <span class="chart-badge">Peak → Week {cp}</span></div></div>""", unsafe_allow_html=True)
    st.plotly_chart(cf, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""<div class="chart-wrap" style="margin-top:0"><div class="chart-head">
      <div><div class="chart-title">Chart 2 — Inventory Health Distribution</div>
      <div class="chart-sub">Stockout vs healthy records · Weekly event frequency</div></div>
      <span class="chart-badge">{metrics['stockout_rate']} Stockout Rate</span></div></div>""",
                unsafe_allow_html=True)

    pc, bc2 = st.columns(2)
    with pc:
        sn=metrics['stockout_count']; hn=metrics['total_rows']-sn
        fig_pie=go.Figure(data=[go.Pie(
            labels=['Stockout Events','Healthy Records'], values=[sn,hn], hole=.55,
            marker=dict(colors=['#ede9e3','#1e1e1e'],line=dict(color='#080808',width=3)),
            textfont=dict(family='monospace',size=11,color='#080808'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>')])
        fig_pie.add_annotation(text=f"<b>{metrics['stockout_rate']}</b><br>Stockout",
                               x=.5,y=.5,showarrow=False,
                               font=dict(size=13,color='#ede9e3',family='monospace'))
        fig_pie.update_layout(paper_bgcolor='#080808',plot_bgcolor='#080808',showlegend=True,
                              legend=dict(font=dict(color='#aaa',size=10,family='monospace'),
                                          bgcolor='#080808',bordercolor='#1e1e1e',borderwidth=1),
                              margin=dict(l=10,r=10,t=30,b=10),height=300)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar':False})

    with bc2:
        avg_so=wkly['stockouts'].mean()
        sc=['#ede9e3' if v>avg_so else '#2a2a2a' for v in wkly['stockouts']]
        fig_so=go.Figure()
        fig_so.add_trace(go.Bar(x=wkly['week'],y=wkly['stockouts'],marker_color=sc,marker_line_width=0,
                                hovertemplate='<b>Week %{x}</b><br>Stockouts: %{y}<extra></extra>'))
        fig_so.add_hline(y=avg_so,line_dash='dot',line_color='#555',line_width=1,
                         annotation_text=f'Avg {avg_so:.1f}',
                         annotation_font=dict(color='#aaa',size=9,family='monospace'),
                         annotation_position='top right')
        fig_so.update_layout(paper_bgcolor='#080808',plot_bgcolor='#0d0d0d',
                             font=dict(family='monospace',color='#aaa',size=10),
                             xaxis=dict(title='WEEK',gridcolor='#1e1e1e',zeroline=False,
                                         tickprefix='W',tickfont=dict(color='#aaa')),
                             yaxis=dict(title='STOCKOUT EVENTS',gridcolor='#1e1e1e',zeroline=False,
                                         tickfont=dict(color='#aaa')),
                             margin=dict(l=50,r=20,t=30,b=50),showlegend=False,bargap=.4,height=300)
        st.plotly_chart(fig_so, use_container_width=True, config={'displayModeBar':False})

    st.markdown(f"""<div class="chart-wrap" style="margin-top:0"><div class="chart-head">
      <div><div class="chart-title">Chart 3 — Cumulative Revenue Loss</div>
      <div class="chart-sub">Running total · Steep slope = stockout cluster</div></div>
      <span class="chart-badge">Total → {metrics['revenue_loss']}</span></div></div>""",
                unsafe_allow_html=True)
    df_loss=df_viz.copy().sort_values('date').reset_index(drop=True)
    ap=df_loss['price_per_unit'].mean() if 'price_per_unit' in df_loss.columns else 1
    df_loss['is_so']=(df_loss['inventory']==0)&(df_loss['units_sold']>0)
    df_loss['rl']=df_loss.apply(lambda r:r['units_sold']*ap if r['is_so'] else 0,axis=1)
    df_loss['cl']=df_loss['rl'].cumsum()/1e5
    fig_line=go.Figure()
    fig_line.add_trace(go.Scatter(x=df_loss.index,y=df_loss['cl'],mode='lines',
                                  line=dict(color='#ede9e3',width=2),fill='tozeroy',
                                  fillcolor='rgba(237,233,227,.06)',
                                  hovertemplate='Record %{x}<br>Loss: Rs. %{y:.1f}L<extra></extra>'))
    pr=df_loss[df_loss['week']==pk_w]
    if len(pr)>0:
        fig_line.add_vrect(x0=pr.index[0],x1=pr.index[-1],fillcolor='rgba(237,233,227,.08)',
                           line_width=0,annotation_text=f'W{pk_w}',
                           annotation_font=dict(color='#aaa',size=9,family='monospace'),
                           annotation_position='top left')
    fl=df_loss['cl'].iloc[-1]
    fig_line.add_annotation(x=df_loss.index[-1],y=fl,text=f"Rs. {fl:.1f}L",
                            showarrow=True,arrowhead=0,arrowcolor='#ede9e3',arrowwidth=1,
                            font=dict(size=10,color='#ede9e3',family='monospace'),
                            bgcolor='#1a1a1a',borderpad=5,xshift=-60,yshift=10)
    fig_line.update_layout(paper_bgcolor='#080808',plot_bgcolor='#0d0d0d',
                           font=dict(family='monospace',color='#aaa',size=10),
                           xaxis=dict(title='RECORD INDEX',gridcolor='#1e1e1e',zeroline=False,tickfont=dict(color='#aaa')),
                           yaxis=dict(title='CUMULATIVE LOSS (Rs. LAKHS)',gridcolor='#1e1e1e',zeroline=False,tickfont=dict(color='#aaa')),
                           margin=dict(l=70,r=30,t=20,b=50),showlegend=False,height=320)
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar':False})

    # ══════════════════════════════════════════════════════════════════════
    # FIX 3 — ACTION BUTTONS (HTML grid wrapper + st.columns inside)
    # The wrapper div creates the visual grid; st.columns fills each cell.
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if 'ai_result' not in st.session_state: st.session_state.ai_result = ""
    if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
    if 'pdf_fname' not in st.session_state: st.session_state.pdf_fname = ""

    # Label row
    st.markdown("""
    <div class="action-grid">
      <div class="action-cell"><span class="action-lbl">AI Diagnosis</span></div>
      <div class="action-cell"><span class="action-lbl">Build PDF</span></div>
      <div class="action-cell"><span class="action-lbl">Send to Inbox</span></div>
      <div class="action-cell"><span class="action-lbl">Download PDF</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Button row (separate columns, no wrapper div so gap:0 applies cleanly)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        run_ai = st.button("⬡ RUN GROQ AI", use_container_width=True)
    with b2:
        gen_pdf = st.button("⬡ GENERATE PDF", use_container_width=True)
    with b3:
        send_email = st.button("📤 SEND QUANTIS REPORT", use_container_width=True)
    with b4:
        if st.session_state.pdf_bytes:
            st.download_button(
                label="↓ DOWNLOAD PDF",
                data=st.session_state.pdf_bytes,
                file_name=st.session_state.pdf_fname,
                mime="application/pdf",
                key="dl_pdf",
                use_container_width=True)
        else:
            st.markdown(
                '<div style="height:46px;display:flex;align-items:center;'
                'font-family:\'DM Mono\',monospace;font-size:.55rem;color:#333;'
                'letter-spacing:.12em;border:1px dashed #1a1a1a;padding:0 1rem">'
                'Generate PDF first</div>',
                unsafe_allow_html=True)

    # ── GROQ AI ──────────────────────────────────────────────────────────
    if run_ai:
        with st.spinner("Groq LLaMA-3.3 analyzing your operations data..."):
            ai_raw = groq_ai_diagnosis(metrics, df_temp)
        st.session_state.ai_result = ai_raw
        st.session_state.pdf_bytes = None

    # ── GENERATE PDF ──────────────────────────────────────────────────────
    if gen_pdf:
        with st.spinner("Building 4-page Quantis PDF..."):
            st.session_state.pdf_bytes = generate_pdf_report(
                metrics, df_temp, email, st.session_state.ai_result)
            st.session_state.pdf_fname = f"Quantis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        has_ai = bool(st.session_state.ai_result and not st.session_state.ai_result.startswith("ERROR"))
        note = "AI insights baked in ✓" if has_ai else "No AI insights — run Groq first, then regenerate"
        st.markdown(f'<div class="success-bar">✓ &nbsp; PDF READY — {note}<br>'
                    f'&nbsp; &nbsp; Click ↓ DOWNLOAD PDF to save</div>', unsafe_allow_html=True)

    # ── SEND EMAIL ────────────────────────────────────────────────────────
    if send_email:
        if not st.session_state.pdf_bytes:
            with st.spinner("Auto-generating PDF..."):
                st.session_state.pdf_bytes = generate_pdf_report(
                    metrics, df_temp, email, st.session_state.ai_result)
                st.session_state.pdf_fname = f"Quantis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        with st.spinner(f"Sending Quantis report to {email}..."):
            ok, msg = send_real_email(email, metrics, st.session_state.pdf_bytes,
                                      st.session_state.ai_result, st.session_state.pdf_fname)
        if ok:
            st.markdown(f"""<div class="success-bar">
              ✓ &nbsp; QUANTIS REPORT DELIVERED — {email} — {pd.Timestamp.now().strftime('%H:%M:%S IST')}<br>
              &nbsp; &nbsp; PDF attached · HTML report with product metrics + AI insights
            </div>""", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'<div class="error-bar">✗ &nbsp; EMAIL FAILED<br>&nbsp; &nbsp; {msg}</div>',
                        unsafe_allow_html=True)

    # ── AI PANEL ──────────────────────────────────────────────────────────
    if st.session_state.ai_result:
        ai_raw = st.session_state.ai_result
        st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
        if ai_raw.startswith("ERROR"):
            st.markdown(f'<div class="error-bar">✗ &nbsp; {ai_raw}</div>', unsafe_allow_html=True)
        else:
            lbl = "LLaMA-3.3 70B · Fresh Analysis" if run_ai else "LLaMA-3.3 70B · Cached"
            st.markdown(f"""
            <div class="ai-panel">
              <div class="ai-panel-hdr">
                <span class="ai-badge">GROQ AI</span>
                <span class="ai-title">Root Cause Analysis</span>
                <span class="ai-model">{lbl}</span>
              </div>
              <div class="ai-body">{fmt_ai(ai_raw)}</div>
            </div>""", unsafe_allow_html=True)

    # ── DATA PREVIEW ──────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="ds-title">Raw Data Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-sub">First 10 rows · Uploaded dataset</div>', unsafe_allow_html=True)
    st.dataframe(df_temp.head(10), use_container_width=True, hide_index=False)

    if os.path.exists("temp_quantis.csv"):
        os.remove("temp_quantis.csv")

elif uploaded_file and not email:
    st.markdown("""<div class="await-box">
      <div class="await-lbl">Enter delivery email to generate report</div>
      <div class="await-big">ADD YOUR EMAIL ADDRESS ABOVE</div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="await-box">
      <div class="await-lbl">Awaiting input — Upload CSV + Enter email to begin audit</div>
      <div class="await-big">₹1CR+ REVENUE INSIGHTS + AI DIAGNOSIS IN &lt;30 SECONDS</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <div class="footer-brand">QUANTIS</div>
  <div class="footer-note">Precision Operations Intelligence — v3.0 — Powered by Groq LLaMA-3.3 70B</div>
</div>""", unsafe_allow_html=True)