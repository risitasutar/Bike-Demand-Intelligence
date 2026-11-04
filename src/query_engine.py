"""
Natural-language query engine for the Operations Assistant.

Transparent, rule-based (no paid API):
  1. parse()  – extract intent + slots (date, hours, weather, temperature, day type) with regular expressions.
  2. answer() – compute the answer from the trained model and the historical data.
  3. Every answer reports how the question was interpreted and which assumptions were used.
  4. Questions outside the data (stations, revenue, fleet owned...) get an explicit "can't answer" with the reason.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import feature_engineering as fe
import model_training as mt
import preprocessing as pp

WEATHER_NAMES = {1: "clear / partly cloudy", 2: "mist / cloudy", 3: "light rain / snow"}
WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

WEATHER_WORDS = [
    (3, r"\b(rain\w*|storm\w*|snow\w*|wet|showers?|thunder\w*|drizzl\w*|bad weather)\b"),
    (2, r"\b(mist\w*|cloud(y|s)?|overcast|fog\w*|grey|gray)\b"),
    (1, r"\b(clear|sunny|sun|dry|nice weather|good weather|fine weather)\b"),
]
UNSUPPORTED = [
    (r"\b(stations?|docks?|where|location|neighbou?rhoods?|streets?|area|zone|map)\b",
     "station / location",
     "The dataset is a single city-wide count per hour. It has no station or location information, "
     "so I can't say *where* demand or shortages occur."),
    (r"\b(revenue|profit|price|pricing|cost|money|income|\$)\b",
     "money",
     "The dataset contains rental counts only - no prices, revenue or costs."),
    (r"\b(fleet|own|owned|inventory|stock(out)?s?|available bikes|availability|empty|unavailable)\b",
     "fleet / availability",
     "The dataset records completed rentals only. It does not contain the fleet size, bikes available at stations, "
     "or failed rental attempts, so I can't measure shortages directly. I *can* estimate bikes on the road "
     "(ask e.g. 'how many bikes are needed at 5 pm tomorrow?')."),
    (r"\b(duration|how long|trip length|distance|minutes per trip)\b",
     "trip duration",
     "Trip durations are not in the dataset. The bikes-on-the-road estimate uses the trip time set in the sidebar (an assumption)."),
]


@dataclass
class Parsed:
    intent: str
    question: str
    dates: list = field(default_factory=list)
    hours: list | None = None
    hour_label: str | None = None
    weather: int | None = None
    temp_c: float | None = None
    day_type: str | None = None          # "Working day" / "Non-working day"
    holiday: int | None = None
    notes: list = field(default_factory=list)
    unsupported_reason: tuple | None = None


@dataclass
class Answer:
    supported: bool
    title: str
    text: str
    metrics: list = field(default_factory=list)       # [(label, value, help)]
    table: pd.DataFrame | None = None
    chart: dict | None = None
    interpretation: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)


# ====================================================================== loading
def load_or_train(project_dir: str = ".", verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Load the saved bundle; if it is missing or was saved with another scikit-learn version, retrain (~30-60 s)."""
    import sklearn
    hour = fe.add_features(pp.clean_hour(pp.load_hour(os.path.join(project_dir, "data", "hour.csv"))))
    path = os.path.join(project_dir, "models", "best_model.joblib")
    if os.path.exists(path):
        try:
            b = mt.load_bundle(path)
            if b.get("sklearn_version") == sklearn.__version__:
                return b, hour
            if verbose:
                print(f"Saved model built with scikit-learn {b.get('sklearn_version')}, you have {sklearn.__version__} -> retraining.")
        except Exception as e:                       # corrupted / incompatible pickle
            if verbose:
                print("Could not load saved model:", e, "-> retraining.")
    return train_bundle(hour, path, verbose), hour


def train_bundle(hour: pd.DataFrame, path: str, verbose: bool = True) -> dict:
    """Same recipe as notebook 03: GB depth 6, multipliers from time-ordered CV folds, refit on all data."""
    from sklearn.base import clone
    F = fe.MODEL_FEATURES
    est = mt.candidate_models()["Gradient boosting (depth 6)"]
    train, _ = mt.chrono_split(hour)
    if verbose:
        print("Calibrating planning multipliers on validation folds...")
    oof = mt.out_of_fold_predictions(est, train, F)
    mult = mt.planning_multipliers(train.loc[oof.index, "cnt"], oof, train.loc[oof.index, "rush_hour"])
    if verbose:
        print("Fitting final model on 2011-2012...")
    model = clone(est).fit(hour[F], hour.cnt)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    test_metrics = {}
    res = os.path.join(os.path.dirname(os.path.dirname(path)), "outputs", "model_results", "test_metrics.csv")
    if os.path.exists(res):                      # accuracy figures from notebook 03 (same model recipe)
        tm = pd.read_csv(res, index_col=0)
        if "Gradient boosting (depth 6)" in tm.index:
            test_metrics = tm.loc["Gradient boosting (depth 6)"].to_dict()
    mt.save_bundle(path, model=model, features=F, model_name="Gradient boosting (depth 6)",
                   planning_multipliers=mult, climate=mt.climate_table(hour), test_metrics=test_metrics,
                   trained_on="2011-01-01..2012-12-31", notes="retrained by query_engine.load_or_train")
    return mt.load_bundle(path)


# ====================================================================== engine
class DemandAssistant:
    def __init__(self, bundle: dict, hour: pd.DataFrame, today=None, trip_minutes: float = 20):
        self.model = bundle["model"]
        self.F = bundle["features"]
        self.mult = bundle["planning_multipliers"]
        self.climate = bundle["climate"]
        self.metrics = bundle.get("test_metrics", {})
        self.hour = hour
        self.today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
        self.trip_minutes = trip_minutes
        self.holidays = set(hour.loc[hour.holiday == 1, "dteday"].dt.normalize())
        self.data_start, self.data_end = hour.dteday.min(), hour.dteday.max()
        # how often each weather type occurs, by month and hour (for "weather not specified")
        self.weather_freq = (hour.groupby(["mnth", "hr"]).weather.value_counts(normalize=True)
                             .unstack(fill_value=0).reindex(columns=[1, 2, 3], fill_value=0))
        # typical humidity for each weather type in each month: rainy hours are much more humid (84% vs 57% clear),
        # so a rain scenario with 'normal' humidity would understate the rain effect
        self.hum_by_weather = hour.groupby(["mnth", "weather"]).hum_pct.median()

    # ------------------------------------------------------------------ parsing
    def parse(self, question: str) -> Parsed:
        q = " " + question.lower().replace("’", "'").strip() + " "
        p = Parsed(intent="", question=question)

        p.weather = self._parse_weather(q)
        p.temp_c = self._parse_temp(q, p)
        p.hours, p.hour_label = self._parse_hours(q, p)
        p.dates = self._parse_dates(q, p)
        if re.search(r"\b(holiday|bank holiday|public holiday)\b", q):
            p.holiday = 1
        if re.search(r"\b(working ?days?|work ?days?|weekdays?|business days?)\b", q):
            p.day_type = "Working day"
        if re.search(r"\b(weekends?|non[- ]?working|saturdays?|sundays?|holidays?)\b", q):
            p.day_type = "Non-working day" if p.day_type is None else "both"

        # 1) out-of-scope topics
        asks_bikes_needed = re.search(r"\bbikes? (are |be |will be )?(needed|required)\b|\bneed\b", q)
        for pattern, topic, reason in UNSUPPORTED:
            if re.search(pattern, q):
                if topic == "fleet / availability" and asks_bikes_needed:
                    continue                                  # answered as bikes-on-the-road estimate
                p.intent, p.unsupported_reason = "unsupported", (topic, reason)
                return p
        # 2) intents, most specific first
        if re.search(r"\b(help|what can you|examples?|what (questions|can i ask))\b", q):
            p.intent = "help"
        elif (re.search(r"\b(weekdays?|working ?days?)\b", q) and re.search(r"\b(weekends?|non[- ]?working)\b", q)) \
                or re.search(r"\bweekdays? (vs|versus|and|or) weekends?\b", q):
            p.intent = "compare_daytypes"
        elif re.search(r"\b(histor\w*|past|last year|previous year|actual|observed|compare[ds]? with)\b", q):
            p.intent = "historical_compare"
        elif p.weather and not p.hours and re.search(r"\b(happen\w*|effect|impact|affect\w*|change\w*|during|when it|influence|differ\w*)\b", q):
            p.intent = "weather_effect"
        elif re.search(r"\b(month|months|season\w*|winter|summer|autumn|fall|spring|year)\b", q) \
                and re.search(r"\b(busiest|highest|lowest|quietest|peak|compare|demand|change|vary|varies)\b", q) and not p.dates:
            p.intent = "seasonal"
        elif re.search(r"\brush\b|\bcommut\w*", q):
            p.intent = "rush_hour"
        elif re.search(r"\b(peak|busiest|highest|most|top)\b", q) and re.search(r"\bhours?\b", q) and (not p.hours or len(p.hours) > 1):
            p.intent = "peak_hours"
        elif re.search(r"\b(bikes? (are |be )?(needed|required)|need|should we have|how many bikes do we)\b", q) and p.hours:
            p.intent = "point_forecast"
        elif p.hours and (p.dates or p.day_type or re.search(r"\b(expect\w*|predict\w*|forecast\w*|how many|demand|rent\w*)\b", q)):
            p.intent = "point_forecast"
        elif p.dates or re.search(r"\b(forecast|predict\w*|expect\w*|hourly|profile|whole day|all day)\b", q):
            p.intent = "day_profile"
        else:
            p.intent = "unknown"
        return p

    def _parse_weather(self, q):
        for code, pattern in WEATHER_WORDS:
            if re.search(pattern, q):
                return code
        return None

    def _parse_temp(self, q, p):
        m = re.search(r"(-?\d{1,2}(?:\.\d)?)\s*(?:°\s*c?|degrees?(?: c(?:elsius)?)?|deg\b|celsius|c\b)", q)
        if m:
            return float(m.group(1))
        for word, val in [("freezing", 2.0), ("very cold", 4.0), ("cold", 6.0), ("mild", 18.0),
                          ("warm", 25.0), ("very hot", 35.0), ("hot", 32.0)]:
            if re.search(rf"\b{word}\b", q):
                p.notes.append(f"'{word}' interpreted as {val:.0f} °C")
                return val
        return None

    def _parse_hours(self, q, p):
        if re.search(r"\bmorning (rush|peak|commute)\b|\brush hour in the morning\b", q):
            return [7, 8, 9], "morning rush (07:00-09:59)"
        if re.search(r"\b(evening|afternoon) (rush|peak|commute)\b|\brush hour in the evening\b", q):
            return [16, 17, 18, 19], "evening rush (16:00-19:59)"
        if re.search(r"\brush ?hours?\b|\bcommut\w*", q):
            return [7, 8, 9, 16, 17, 18, 19], "both rush windows (07-09 and 16-19)"
        hours = []
        for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", q):
            h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
            hours.append(h)
        for m in re.finditer(r"(?<![\d-])([01]?\d|2[0-3]):([0-5]\d)(?!\s*(a\.?m|p\.?m))", q):
            hours.append(int(m.group(1)))
        if re.search(r"\b(noon|midday|lunch ?time)\b", q):
            hours.append(12)
        if re.search(r"\bmidnight\b", q):
            hours.append(0)
        if not hours:
            m = re.search(r"\bat (\d{1,2})\b(?!\s*(?:°|degrees?|deg|c\b|celsius|%|th|st|nd|rd))", q)
            if m and int(m.group(1)) <= 23:
                h = int(m.group(1))
                hours.append(h)
                p.notes.append(f"'at {h}' read as {h:02d}:00 (24-hour clock)")
        if not hours:
            return None, None
        hours = sorted(set(hours))
        return hours, ", ".join(f"{h:02d}:00" for h in hours)

    def _parse_dates(self, q, p):
        t = self.today
        if re.search(r"\bday after tomorrow\b", q):
            return [t + pd.Timedelta(days=2)]
        if re.search(r"\btomorrow\b", q):
            return [t + pd.Timedelta(days=1)]
        if re.search(r"\b(today|tonight|this evening|this morning)\b", q):
            return [t]
        m = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", q)
        if m:
            return [pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))]
        mon = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
        m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{mon}(?:,?\s+(20\d\d))?\b", q) or \
            re.search(rf"\b{mon}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", q)
        if m:
            g = m.groups()
            if g[0].isdigit():
                day, month, year = int(g[0]), MONTHS[g[1][:3]], g[2]
            else:
                month, day, year = MONTHS[g[0][:3]], int(g[1]), g[2]
            year = int(year) if year else t.year
            try:
                return [pd.Timestamp(year, month, day)]
            except ValueError:
                p.notes.append("could not read that date")
        m = re.search(r"\b(?:in|during|for|of)\s+(january|february|march|april|may|june|july|august|september|october|"
                      r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", q)
        if m:                                              # month only -> a representative day in that month
            month = MONTHS[m.group(1)[:3]]
            year = t.year if month >= t.month else t.year + 1
            d = pd.Timestamp(year, month, 15)
            for i, name in enumerate(WEEKDAY_NAMES):          # "a saturday in june" -> first Saturday from the 15th
                if re.search(rf"\b{name}\b", q):
                    d += pd.Timedelta(days=(i - d.dayofweek) % 7)
            p.notes.append(f"month only -> used a representative day: {d:%a %d %b %Y}")
            return [d]
        for i, name in enumerate(WEEKDAY_NAMES):
            if re.search(rf"\b{name}s?\b", q) and not re.search(rf"\b{name}s\b", q):
                delta = (i - t.dayofweek) % 7                 # 0 = today
                if delta == 0 and re.search(rf"\bnext {name}\b", q):
                    delta = 7
                return [t + pd.Timedelta(days=delta)]
        if re.search(r"\b(this|next|the|coming) weekend\b", q):
            delta = (5 - t.dayofweek) % 7
            return [t + pd.Timedelta(days=delta)]
        return []

    # ------------------------------------------------------------------ model helpers
    def _is_holiday(self, date):
        return int(pd.Timestamp(date).normalize() in self.holidays)

    def predict_day(self, date, hours=range(24), weather=None, temp_c=None, holiday=None) -> pd.DataFrame:
        """Point and P90 forecast per hour. If weather is None: average over weather types,
        weighted by how often each occurred in that month and hour (2011-2012)."""
        date = pd.Timestamp(date)
        hol = self._is_holiday(date) if holiday is None else holiday
        hours = list(hours)
        hum = lambda w: self.hum_by_weather.get((date.month, w), self.hum_by_weather.xs(w, level="weather").median())
        if weather is not None:
            fr = mt.scenario_frame(date, hours, weather=weather, temp_c=temp_c, hum_pct=hum(weather), holiday=hol,
                                   climate=self.climate)
            point = np.clip(self.model.predict(fr[self.F]), 0, None)
        else:
            point = np.zeros(len(hours))
            for w in (1, 2, 3):
                fr_w = mt.scenario_frame(date, hours, weather=w, temp_c=temp_c, hum_pct=hum(w), holiday=hol,
                                         climate=self.climate)
                wts = self.weather_freq.loc[[(date.month, h) for h in hours], w].to_numpy()
                point += wts * np.clip(self.model.predict(fr_w[self.F]), 0, None)
            fr = fr_w
        out = fr[["dteday", "hr", "workingday", "holiday", "rush_hour", "temp", "hum"]].copy()
        out["temp_c"] = out.temp * 41
        out["point"] = point
        out["plan90"] = mt.planning_forecast(point, fr.rush_hour, self.mult)
        out["bikes_on_road"] = mt.bikes_in_use(out.point, self.trip_minutes)
        out["bikes_on_road_p90"] = mt.bikes_in_use(out.plan90, self.trip_minutes)
        return out.reset_index(drop=True)

    def history(self, month=None, day_type=None, year=1) -> pd.Series:
        h = self.hour
        if year is not None:
            h = h[h.yr == year]
        if month is not None:
            h = h[h.mnth == month]
        if day_type in ("Working day", "Non-working day"):
            h = h[h.day_type == day_type]
        return h.groupby("hr").cnt.mean()

    def actual_for(self, date) -> pd.Series | None:
        d = self.hour[self.hour.dteday == pd.Timestamp(date).normalize()]
        return d.set_index("hr").cnt if len(d) else None

    def _default_date(self, p: Parsed, want="Working day"):
        if p.dates:
            return p.dates[0], None
        d = self.today + pd.Timedelta(days=1)
        for _ in range(8):
            wd = d.dayofweek < 5 and not self._is_holiday(d)
            if (want == "Working day" and wd) or (want == "Non-working day" and not wd):
                return d, f"no date given -> used the next {want.lower()} ({d:%a %d %b %Y})"
            d += pd.Timedelta(days=1)
        return d, None

    def _weather_note(self, p, date):
        if p.weather is not None:
            return f"weather: {WEATHER_NAMES[p.weather]} (from your question)"
        f = self.weather_freq.loc[date.month].mean()
        return (f"weather not specified -> averaged over weather types, weighted by how often each occurred in "
                f"{date:%B} ({f[1]:.0%} clear, {f[2]:.0%} mist, {f[3]:.0%} rain/snow)")

    def _common_assumptions(self, date=None):
        a = ["Demand level of 2012 (the latest year in the data); the model has not seen later growth.",
             f"Point forecast = expected rentals. P90 plan = forecast x calibrated multiplier "
             f"({self.mult['rush']:.2f} at rush hours, {self.mult['other']:.2f} otherwise) - covered ~90% of test hours."]
        if date is not None:
            date = pd.Timestamp(date)
            if date > self.data_end or date < self.data_start:
                a.append("Public holidays are only known for 2011-2012; add 'holiday' to the question to treat a date as one.")
            a.append("Temperature and wind default to the historical median for that month and hour (unless you give a temperature); "
                     "humidity = typical humidity for that weather type in that month.")
        return a

    def _interp(self, p: Parsed, **extra):
        d = {"intent": p.intent,
             "date(s)": ", ".join(f"{x:%a %d %b %Y}" for x in p.dates) or "-",
             "hour(s)": p.hour_label or "-",
             "weather": WEATHER_NAMES.get(p.weather, "not specified"),
             "temperature": f"{p.temp_c:.0f} °C" if p.temp_c is not None else "not specified",
             "day type": p.day_type or "-"}
        d.update(extra)
        if p.notes:
            d["notes"] = "; ".join(p.notes)
        return d

    # ------------------------------------------------------------------ answering
    def ask(self, question: str) -> Answer:
        p = self.parse(question)
        handler = getattr(self, f"_a_{p.intent}", self._a_unknown)
        return handler(p)

    def _a_unsupported(self, p):
        topic, reason = p.unsupported_reason
        return Answer(False, f"Can't answer that from this data ({topic})", reason +
                      "\n\nQuestions I can answer are about **how many rentals to expect, when, and under which weather** "
                      "- type `help` for examples.", interpretation=self._interp(p))

    def _a_unknown(self, p):
        return Answer(False, "I didn't understand that question",
                      "I only answer the question types listed under `help`, so I won't guess. Try for example:\n"
                      "- *How many bikes are expected to be rented tomorrow at 8 AM?*\n"
                      "- *What are the peak-demand hours on working days?*\n"
                      "- *What happens to demand during rainy weather?*", interpretation=self._interp(p))

    def _a_help(self, p):
        txt = """I answer questions about **hourly bike rental demand** using the trained model and 2011-2012 history:

| Question type | Example |
|---|---|
| Forecast for a date and hour | *How many bikes are expected to be rented tomorrow at 8 AM?* · *Friday 6 pm if it rains?* · *14 Sep 2012 at 17:00* |
| Whole-day forecast | *Forecast for next Saturday* |
| Rush hours | *What is the expected demand during the evening rush hour?* |
| Peak hours | *What are the peak-demand hours on working days?* · *Which hours have the highest predicted demand?* |
| Weekday vs weekend | *How does demand change between weekdays and weekends?* |
| Weather | *What happens to demand during rainy weather?* |
| Season | *Which months are busiest?* |
| Model vs history | *How does predicted demand compare with historical demand?* |
| Bikes needed | *How many bikes are needed at 5 pm tomorrow?* (bikes on the road, with the trip-time assumption) |

I **cannot** answer questions about stations/locations, revenue, fleet size or stock-outs - that data isn't in the dataset."""
        return Answer(True, "What I can answer", txt, interpretation=self._interp(p))

    def _a_point_forecast(self, p):
        date, note = self._default_date(p, "Non-working day" if p.day_type == "Non-working day" else "Working day")
        if note:
            p.notes.append(note)
        df = self.predict_day(date, p.hours, p.weather, p.temp_c, p.holiday)
        full = self.predict_day(date, range(24), p.weather, p.temp_c, p.holiday)
        dt = "Working day" if full.workingday.iloc[0] else "Non-working day"
        hist = self.history(date.month, dt)
        act = self.actual_for(date)
        lines, metrics = [], []
        for r in df.itertuples():
            hh = hist.get(r.hr, np.nan)
            lines.append(f"- **{r.hr:02d}:00** - expected **{r.point:.0f} rentals** (P90 plan {r.plan90:.0f}); "
                         f"≈ **{r.bikes_on_road:.0f} bikes on the road** ({r.bikes_on_road_p90:.0f} at P90). "
                         f"Historical average for {date:%B} {dt.lower()}s in 2012: {hh:.0f}."
                         + (f" *Actual recorded that hour: {act.get(r.hr):.0f}.*" if act is not None and r.hr in act.index else ""))
        tot_point, tot_plan = df.point.sum(), df.plan90.sum()
        metrics.append(("Expected rentals" + (" (total)" if len(df) > 1 else ""), f"{tot_point:,.0f}", "point forecast"))
        metrics.append(("P90 planning level", f"{tot_plan:,.0f}", "covers ~90% of hours"))
        metrics.append((f"Bikes on the road (peak hour)", f"{df.bikes_on_road_p90.max():.0f}",
                        f"P90 rentals x {self.trip_minutes:.0f}-min trip / 60"))
        txt = (f"**{date:%A %d %B %Y}** ({dt.lower()}{', holiday' if full.holiday.iloc[0] else ''}):\n\n" + "\n".join(lines) +
               f"\n\n*Bikes on the road* = rentals x assumed {self.trip_minutes:.0f}-minute trip ÷ 60 (Little's law). "
               "It is **not** the fleet size: the fleet also needs bikes parked at stations and bikes in repair.")
        chart = {"kind": "line", "title": f"Hourly forecast - {date:%a %d %b %Y}", "x": list(range(24)),
                 "series": {"Forecast": full.point.round(1).tolist(), "P90 plan": full.plan90.round(1).tolist(),
                            f"History ({date:%b} 2012, {dt.lower()}s)": hist.reindex(range(24)).round(1).tolist()},
                 "highlight": p.hours, "xlabel": "hour of day", "ylabel": "rentals per hour"}
        if act is not None:
            chart["series"]["Actual (recorded)"] = act.reindex(range(24)).round(1).tolist()
        tbl = df[["hr", "point", "plan90", "bikes_on_road", "bikes_on_road_p90", "temp_c"]].round(1)
        return Answer(True, f"Forecast for {date:%a %d %b %Y}, {p.hour_label}", txt, metrics, tbl, chart,
                      self._interp(p, **{"used date": f"{date:%a %d %b %Y}", "used weather": self._weather_note(p, date)}),
                      self._common_assumptions(date))

    def _a_day_profile(self, p):
        date, note = self._default_date(p, "Non-working day" if p.day_type == "Non-working day" else "Working day")
        if note:
            p.notes.append(note)
        df = self.predict_day(date, range(24), p.weather, p.temp_c, p.holiday)
        dt = "Working day" if df.workingday.iloc[0] else "Non-working day"
        top = df.nlargest(3, "point")
        act = self.actual_for(date)
        txt = (f"**{date:%A %d %B %Y}** ({dt.lower()}): expected **{df.point.sum():,.0f} rentals** over the day "
               f"(P90 plan {df.plan90.sum():,.0f}).\n\n"
               f"Busiest hours: " + ", ".join(f"**{r.hr:02d}:00** ({r.point:.0f})" for r in top.itertuples()) +
               f".\n\nQuietest stretch: 01:00-05:00 (≈ {df[df.hr.between(1, 5)].point.mean():.0f}/hr) - the cheapest time to move bikes.")
        hist = self.history(date.month, dt)
        chart = {"kind": "line", "title": f"Hourly forecast - {date:%a %d %b %Y}", "x": list(range(24)),
                 "series": {"Forecast": df.point.round(1).tolist(), "P90 plan": df.plan90.round(1).tolist(),
                            f"History ({date:%b} 2012, {dt.lower()}s)": hist.reindex(range(24)).round(1).tolist()},
                 "xlabel": "hour of day", "ylabel": "rentals per hour"}
        if act is not None:
            chart["series"]["Actual (recorded)"] = act.reindex(range(24)).round(1).tolist()
            txt += (f"\n\n*This date is in the historical data: **{act.sum():,.0f}** rentals were actually recorded "
                    f"({len(act)} hours on file). Large gaps usually mean an event the model can't know about (e.g. a storm closure).*")
        return Answer(True, f"Day forecast - {date:%a %d %b %Y}", txt,
                      [("Expected rentals (day)", f"{df.point.sum():,.0f}", ""), ("Busiest hour", f"{top.hr.iloc[0]:02d}:00", ""),
                       ("Peak bikes on the road (P90)", f"{df.bikes_on_road_p90.max():.0f}", f"{self.trip_minutes:.0f}-min trips")],
                      df[["hr", "point", "plan90", "bikes_on_road_p90"]].round(1), chart,
                      self._interp(p, **{"used date": f"{date:%a %d %b %Y}", "used weather": self._weather_note(p, date)}),
                      self._common_assumptions(date))

    def _a_rush_hour(self, p):
        date, note = self._default_date(p)
        if note:
            p.notes.append(note)
        hours = p.hours or [7, 8, 9, 16, 17, 18, 19]
        df = self.predict_day(date, range(24), p.weather, p.temp_c, p.holiday)
        win = df[df.hr.isin(hours)]
        dt = "Working day" if df.workingday.iloc[0] else "Non-working day"
        txt = (f"**{p.hour_label or 'Rush hours'}** on **{date:%A %d %B %Y}** ({dt.lower()}):\n\n"
               f"- Expected **{win.point.sum():,.0f} rentals** in the window (avg {win.point.mean():.0f}/hr, peak "
               f"**{win.point.max():.0f}** at {int(win.loc[win.point.idxmax(), 'hr']):02d}:00).\n"
               f"- P90 planning level: {win.plan90.sum():,.0f} (peak hour {win.plan90.max():.0f}).\n"
               f"- ≈ **{win.bikes_on_road_p90.max():.0f} bikes on the road** at the busiest hour (P90, {self.trip_minutes:.0f}-min trips).\n"
               f"- The window holds **{win.point.sum() / df.point.sum():.0%}** of the day's expected rentals in {len(hours)} of 24 hours.")
        if dt == "Non-working day":
            txt += "\n\n*Note: this is a non-working day, so there is no commute peak; demand peaks around midday instead.*"
        chart = {"kind": "line", "title": f"Forecast with rush window - {date:%a %d %b %Y}", "x": list(range(24)),
                 "series": {"Forecast": df.point.round(1).tolist(), "P90 plan": df.plan90.round(1).tolist()},
                 "highlight": hours, "xlabel": "hour of day", "ylabel": "rentals per hour"}
        return Answer(True, f"Rush-hour demand - {date:%a %d %b}", txt,
                      [("Rentals in window", f"{win.point.sum():,.0f}", ""), ("Peak hour (P90)", f"{win.plan90.max():.0f}", ""),
                       ("Bikes on the road (P90)", f"{win.bikes_on_road_p90.max():.0f}", "")],
                      win[["hr", "point", "plan90", "bikes_on_road_p90"]].round(1), chart,
                      self._interp(p, **{"used date": f"{date:%a %d %b %Y}", "used weather": self._weather_note(p, date)}),
                      self._common_assumptions(date))

    def _a_peak_hours(self, p):
        types = ["Working day", "Non-working day"] if p.day_type in (None, "both") else [p.day_type]
        series, lines, tables = {}, [], []
        for dt in types:
            h = self.history(None, dt, year=None)
            date = p.dates[0] if p.dates else self._default_date(Parsed("", ""), dt)[0]
            pred = self.predict_day(date, range(24), p.weather, p.temp_c).set_index("hr").point
            top_h, top_p = h.nlargest(3), pred.nlargest(3)
            lines.append(f"**{dt}s** - historically busiest: " + ", ".join(f"{i:02d}:00 ({v:.0f})" for i, v in top_h.items()) +
                         f"; model forecast for {date:%a %d %b %Y}: " + ", ".join(f"{i:02d}:00 ({v:.0f})" for i, v in top_p.items()) + ".")
            series[f"History 2011-12 ({dt.lower()}s)"] = h.reindex(range(24)).round(1).tolist()
            series[f"Forecast {date:%a %d %b}"] = pred.reindex(range(24)).round(1).tolist()
            tables.append(pd.DataFrame({"day type": dt, "hour": top_h.index, "historical avg": top_h.values.round(0)}))
        txt = "\n\n".join(lines) + ("\n\nOn working days demand has **two commute peaks (08:00 and 17:00-18:00)**; "
                                    "on non-working days a **single midday plateau (12:00-15:00)**. "
                                    "Peak hours are where availability problems are most likely.")
        chart = {"kind": "line", "title": "Peak hours: history vs forecast", "x": list(range(24)), "series": series,
                 "xlabel": "hour of day", "ylabel": "rentals per hour"}
        return Answer(True, "Peak-demand hours", txt, [], pd.concat(tables, ignore_index=True), chart,
                      self._interp(p), self._common_assumptions(p.dates[0] if p.dates else None))

    def _a_compare_daytypes(self, p):
        h = self.hour[self.hour.yr == 1]
        prof = h.groupby(["hr", "day_type"]).cnt.mean().unstack()
        daily = h.groupby(["dteday", "day_type"]).cnt.sum().reset_index().groupby("day_type").cnt.mean()
        peak = h[h.hr.isin(fe.PEAK_HOURS)].groupby("day_type").cnt.sum() / h.groupby("day_type").cnt.sum()
        dw, dn = self._default_date(Parsed("", ""), "Working day")[0], self._default_date(Parsed("", ""), "Non-working day")[0]
        pw = self.predict_day(dw, weather=p.weather).point
        pn = self.predict_day(dn, weather=p.weather).point
        txt = (f"**2012 history:** working days average **{daily['Working day']:,.0f}** rentals/day vs **{daily['Non-working day']:,.0f}** "
               f"on non-working days - similar volume (≈ {daily['Working day'] / daily['Non-working day'] - 1:+.0%}).\n\n"
               f"The big difference is **timing**: working days put **{peak['Working day']:.0%}** of rentals into the 7 commute hours "
               f"(07-09, 16-19) vs **{peak['Non-working day']:.0%}** on non-working days. Working days peak at "
               f"{prof['Working day'].idxmax():02d}:00 ({prof['Working day'].max():.0f}/hr); non-working days at "
               f"{prof['Non-working day'].idxmax():02d}:00 ({prof['Non-working day'].max():.0f}/hr).\n\n"
               f"**Model forecast:** {dw:%a %d %b} (working) {pw.sum():,.0f} rentals, peak {pw.max():.0f}/hr; "
               f"{dn:%a %d %b} (non-working) {pn.sum():,.0f}, peak {pn.max():.0f}/hr.\n\n"
               "*Operational meaning:* weekdays need bikes pre-positioned for two short commute peaks; weekends need a broad midday supply.")
        chart = {"kind": "line", "title": "Average rentals by hour, 2012", "x": list(range(24)),
                 "series": {"Working day": prof["Working day"].round(1).tolist(), "Non-working day": prof["Non-working day"].round(1).tolist()},
                 "xlabel": "hour of day", "ylabel": "rentals per hour"}
        tbl = pd.DataFrame({"avg rentals/day (2012)": daily.round(0), "share in commute hours": (100 * peak).round(1)})
        return Answer(True, "Weekdays vs weekends", txt,
                      [("Working-day share in commute hours", f"{peak['Working day']:.0%}", ""),
                       ("Non-working-day share", f"{peak['Non-working day']:.0%}", "")], tbl, chart, self._interp(p),
                      ["History = 2012 averages (latest level).", "Forecast dates: next working and next non-working day."])

    def _a_weather_effect(self, p):
        date, note = self._default_date(p)
        if note:
            p.notes.append(note)
        sc = {w: self.predict_day(date, weather=w, temp_c=p.temp_c, holiday=p.holiday) for w in (1, 2, 3)}
        h = self.hour[self.hour.rush_hour == 1].groupby("weather").cnt.mean()
        hall = self.hour.groupby("weather").cnt.mean()
        w = p.weather
        txt = (f"**Model scenario for {date:%A %d %B %Y}:** clear {sc[1].point.sum():,.0f} rentals, "
               f"mist/cloudy {sc[2].point.sum():,.0f} ({sc[2].point.sum() / sc[1].point.sum() - 1:+.0%}), "
               f"light rain/snow **{sc[3].point.sum():,.0f} ({sc[3].point.sum() / sc[1].point.sum() - 1:+.0%})**. "
               f"Peak hour: {sc[1].point.max():.0f} → {sc[3].point.max():.0f} in rain.\n\n"
               f"**History (2011-12):** at working-day rush hours, average {h[1]:.0f} rentals/hr in clear weather, "
               f"{h[2]:.0f} in mist ({h[2] / h[1] - 1:+.0%}), **{h[3]:.0f} in rain/snow ({h[3] / h[1] - 1:+.0%})**. "
               f"Across all hours: {hall[1]:.0f} / {hall[2]:.0f} / {hall[3]:.0f}.\n\n"
               "Rain lowers demand at every hour but **doesn't move the peaks** - on rainy working days plan for roughly half the "
               "usual peak stock and use spare truck/staff time for maintenance.")
        chart = {"kind": "line", "title": f"Weather scenarios - {date:%a %d %b %Y}", "x": list(range(24)),
                 "series": {f"Clear": sc[1].point.round(1).tolist(), "Mist / cloudy": sc[2].point.round(1).tolist(),
                            "Light rain / snow": sc[3].point.round(1).tolist()},
                 "xlabel": "hour of day", "ylabel": "rentals per hour"}
        tbl = pd.DataFrame({"weather": [WEATHER_NAMES[k] for k in (1, 2, 3)],
                            "forecast rentals (day)": [round(sc[k].point.sum()) for k in (1, 2, 3)],
                            "forecast peak hour": [round(sc[k].point.max()) for k in (1, 2, 3)],
                            "history: rush-hour avg": [round(h[k]) for k in (1, 2, 3)]})
        return Answer(True, "Effect of weather on demand", txt,
                      [("Rain vs clear (forecast, day)", f"{sc[3].point.sum() / sc[1].point.sum() - 1:+.0%}", ""),
                       ("Rain vs clear (history, rush hours)", f"{h[3] / h[1] - 1:+.0%}", "")], tbl, chart,
                      self._interp(p, **{"used date": f"{date:%a %d %b %Y}"}), self._common_assumptions(date) +
                      ["Same temperature in all three scenarios; humidity follows each weather type's typical level for the month."])

    def _a_seasonal(self, p):
        d = self.hour.groupby(["yr", "mnth", "dteday"]).cnt.sum().groupby(["yr", "mnth"]).mean().unstack(0)
        m12 = d[1]
        txt = (f"In 2012 the busiest month was **{pd.Timestamp(2012, m12.idxmax(), 1):%B}** ({m12.max():,.0f} rentals/day) "
               f"and the quietest **{pd.Timestamp(2012, m12.idxmin(), 1):%B}** ({m12.min():,.0f}/day) - a "
               f"**{m12.max() / m12.min():.1f}×** swing. 2011 swung {d[0].max() / d[0].min():.1f}× (Jan → Jun).\n\n"
               "Operational meaning: the bikes needed in winter are a fraction of autumn needs - use winter for maintenance, "
               "and cover the summer/autumn peak with rebalancing or seasonal leasing rather than permanent purchases.")
        chart = {"kind": "bar", "title": "Average daily rentals by month", "x": [pd.Timestamp(2012, m, 1).strftime("%b") for m in range(1, 13)],
                 "series": {"2011": d[0].round(0).tolist(), "2012": d[1].round(0).tolist()}, "xlabel": "month", "ylabel": "rentals per day"}
        return Answer(True, "Seasonal demand", txt, [("Busiest month 2012", f"{pd.Timestamp(2012, m12.idxmax(), 1):%b}", ""),
                                                      ("Peak / low ratio", f"{m12.max() / m12.min():.1f}×", "")],
                      d.rename(columns={0: "2011", 1: "2012"}).round(0), chart, self._interp(p), ["Historical averages (not forecasts)."])

    def _a_historical_compare(self, p):
        if p.dates and p.dates[0] <= self.data_end and p.dates[0] >= self.data_start:
            date = p.dates[0]
            pred = self.predict_day(date, weather=p.weather)
            act = self.actual_for(date).reindex(range(24))
            err = (pred.point.values - act.values)
            txt = (f"**{date:%A %d %B %Y}** is in the historical data, so the forecast can be checked against what happened: "
                   f"forecast **{pred.point.sum():,.0f}** vs actual **{np.nansum(act):,.0f}** rentals "
                   f"({pred.point.sum() / np.nansum(act) - 1:+.0%}); average hourly miss {np.nanmean(np.abs(err)):.0f}.\n\n"
                   "*Note: the deployed model was trained on 2011-2012, so this date was seen in training - for honest "
                   "accuracy see the test metrics below.*")
            series = {"Forecast": pred.point.round(1).tolist(), "Actual": act.round(1).tolist()}
            title = f"Forecast vs actual - {date:%a %d %b %Y}"
        else:
            date, note = self._default_date(p)
            if note:
                p.notes.append(note)
            pred = self.predict_day(date, weather=p.weather)
            dt = "Working day" if pred.workingday.iloc[0] else "Non-working day"
            h12, h11 = self.history(date.month, dt, 1), self.history(date.month, dt, 0)
            txt = (f"Forecast for **{date:%a %d %b %Y}** ({dt.lower()}): **{pred.point.sum():,.0f}** rentals. "
                   f"Historical average for {date:%B} {dt.lower()}s: **{h12.sum():,.0f}** in 2012 and {h11.sum():,.0f} in 2011.\n\n"
                   f"Forecast vs 2012 history: {pred.point.sum() / h12.sum() - 1:+.0%} (difference = weather/temperature assumptions "
                   "and day-specific effects; the level is 2012's).")
            series = {"Forecast": pred.point.round(1).tolist(), f"History 2012 ({date:%b}, {dt.lower()}s)": h12.reindex(range(24)).round(1).tolist(),
                      f"History 2011 ({date:%b}, {dt.lower()}s)": h11.reindex(range(24)).round(1).tolist()}
            title = f"Forecast vs history - {date:%a %d %b %Y}"
        m = self.metrics
        if m:
            txt += (f"\n\n**Model accuracy on unseen months (Jul-Dec 2012):** MAE {m.get('MAE', np.nan):.0f} rentals/hr, "
                    f"R² {m.get('R2', np.nan):.2f}, WAPE {m.get('WAPE_%', np.nan):.0f}%; under-predicts busy peaks "
                    f"(rush-hour bias {m.get('Peak bias', np.nan):.0f}) - hence the P90 plan.")
        chart = {"kind": "line", "title": title, "x": list(range(24)), "series": series, "xlabel": "hour of day", "ylabel": "rentals per hour"}
        return Answer(True, "Predicted vs historical demand", txt, [], None, chart, self._interp(p), self._common_assumptions(date))


# ====================================================================== chart rendering (shared by app + notebook)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]


def make_figure(chart: dict):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    x = chart["x"]
    if chart.get("highlight"):
        for h in chart["highlight"]:
            ax.axvspan(h - 0.5, h + 0.5, color="#e6e5e0", alpha=.7, zorder=0)
    if chart["kind"] == "bar":
        n = len(chart["series"]); w = 0.8 / n
        for i, (name, vals) in enumerate(chart["series"].items()):
            ax.bar(np.arange(len(x)) + (i - (n - 1) / 2) * w, vals, width=w, color=PALETTE[i], label=name, edgecolor="#fcfcfb")
        ax.set_xticks(range(len(x))); ax.set_xticklabels(x)
    else:
        for i, (name, vals) in enumerate(chart["series"].items()):
            style = "--" if "P90" in name else (":" if name.startswith("History") and len(chart["series"]) > 2 else "-")
            color = "#52514e" if "Actual" in name else PALETTE[i % len(PALETTE)]
            ax.plot(x, vals, style, color=color, lw=2 if style == "-" else 1.6, marker="o" if style == "-" else None,
                    markersize=3, label=name)
        ax.set_xticks(x if len(x) <= 24 else x[::2])
    ax.set_title(chart.get("title", ""), loc="left", fontweight="bold")
    ax.set_xlabel(chart.get("xlabel", "")); ax.set_ylabel(chart.get("ylabel", ""))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(chart["series"])))
    fig.tight_layout()
    return fig
