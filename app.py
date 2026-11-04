"""
Bike Demand Intelligence – Operations Assistant (Streamlit prototype)

Run from the project folder:
    pip install -r requirements.txt
    streamlit run app.py

Ask questions in plain English. Answers come from the trained gradient-boosting model
(models/best_model.joblib) and the 2011-2012 history - nothing is hard-coded.
"""
import io
import os
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import query_engine as qe  # noqa: E402

st.set_page_config(page_title="Bike Demand Intelligence", page_icon="🚲", layout="wide")

# Expander and sidebar arrows use plain-text glyphs, so they still render if the icon font fails to load
st.markdown("""<style>
[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {display: none;}
[data-testid="stExpander"] summary::before {content: "▸"; margin-right: .5rem;}
[data-testid="stExpander"] details[open] summary::before {content: "▾";}
[data-testid="stSidebarCollapseButton"] button::before {content: "«"; font-size: 1.3rem;}
[data-testid="stExpandSidebarButton"]::before {content: "»"; font-size: 1.3rem;}
</style>""", unsafe_allow_html=True)

EXAMPLES = [
    "How many bikes are expected to be rented tomorrow at 8 AM?",
    "What are the peak-demand hours on working days?",
    "How does demand change between weekdays and weekends?",
    "What happens to demand during rainy weather?",
    "Which hours have the highest predicted demand?",
    "What is the expected demand during the evening rush hour?",
    "How does predicted demand compare with historical demand?",
    "How many bikes are needed at 5 pm on Friday if it rains?",
    "Which months are busiest?",
]


@st.cache_resource(show_spinner="Loading model (retrains automatically if needed, ~1 min)...")
def load():
    return qe.load_or_train(ROOT, verbose=False)


bundle, hour = load()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    with st.form("date_form", border=False):
        entered = st.text_input("Treat 'today' as (YYYY-MM-DD)",
                                value=st.session_state.get("today_text", date.today().isoformat()),
                                help="Used for 'today', 'tomorrow', 'Friday'... Enter a 2012 date to compare with recorded actuals.")
        if st.form_submit_button("Enter", type="primary", use_container_width=True):
            st.session_state["today_text"] = entered.strip()
    today_text = st.session_state.get("today_text", date.today().isoformat())
    try:
        today = pd.Timestamp(today_text).date()
        st.caption(f"Using 'today' = **{today:%a %d %b %Y}**")
    except (ValueError, TypeError):
        today = date.today()
        st.warning("Date not recognised, using today's date.")
    trip = st.slider("Average trip time (minutes) - ASSUMPTION", 10, 45, 20, 5,
                     help="Not in the dataset. Used only to convert rentals/hour into bikes on the road (Little's law).")
    st.divider()
    st.subheader("Try a question")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["pending"] = ex
    st.divider()
    st.subheader("About the model")
    m = bundle.get("test_metrics", {})
    st.markdown(
        f"**{bundle.get('model_name', 'Gradient boosting')}** · trained on {bundle.get('trained_on', '2011-2012')}  \n"
        + (f"Test (Jul–Dec 2012, unseen): MAE **{m['MAE']:.0f}**/hr · R² **{m['R2']:.2f}** · WAPE **{m['WAPE_%']:.0f}%**  \n"
           if m else "")
        + f"P90 plan multipliers: rush ×{bundle['planning_multipliers']['rush']:.2f}, other ×{bundle['planning_multipliers']['other']:.2f}"
    )
    st.caption("Demand level = 2012. No station, fleet or revenue data - the assistant says so instead of guessing.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["history"] = []

engine = qe.DemandAssistant(bundle, hour, today=pd.Timestamp(today), trip_minutes=trip)

# ------------------------------------------------------------------ main
st.title("🚲 Bike Demand Intelligence – Operations Assistant")
st.caption("Ask about expected rentals, peak hours, weekdays vs weekends, weather effects or how many bikes are on the road. "
           "Type **help** for the full list.")

if "history" not in st.session_state:
    st.session_state["history"] = []


def render(ans: qe.Answer):
    st.markdown(f"#### {ans.title}")
    if not ans.supported:
        st.warning("Outside what this data can answer - no numbers invented.", icon="⚠️")
    st.markdown(ans.text)
    if ans.metrics:
        cols = st.columns(len(ans.metrics))
        for c, (label, value, helptext) in zip(cols, ans.metrics):
            c.metric(label, value, help=helptext or None)
    if ans.chart:
        png = getattr(ans, "_png", None)
        if png is None:
            fig = qe.make_figure(ans.chart)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            png = ans._png = buf.getvalue()
        st.image(png, use_container_width=True)
    if ans.table is not None:
        with st.expander("Numbers behind the answer"):
            st.table(ans.table)
    with st.expander("How I understood your question"):
        st.table(pd.DataFrame({"value": {k: str(v) for k, v in ans.interpretation.items()}}))
    if ans.assumptions:
        with st.expander("Assumptions"):
            st.markdown("\n".join(f"- {a}" for a in ans.assumptions))


for q, ans in st.session_state["history"][-6:]:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        render(ans)

question = st.chat_input("e.g. How many bikes are expected to be rented tomorrow at 8 AM?")
if question is None and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    ans = engine.ask(question)
    st.session_state["history"].append((question, ans))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render(ans)
