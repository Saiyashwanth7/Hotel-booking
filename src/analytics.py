# src/analytics.py
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # no GUI needed on a server
import seaborn as sns
from matplotlib.figure import Figure


def _to_b64(fig: Figure) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class BookingAnalytics:
    def __init__(self, data):
        self.data = data

    def revenue_trends(self):
        """Revenue per month"""
        monthly = (
            self.data.groupby(self.data["arrival_date"].dt.to_period("M"))["revenue"]
            .sum()
            .reset_index()
        )
        monthly["month_year"] = monthly["arrival_date"].dt.strftime("%b %Y")

        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()
        ax.plot(monthly["arrival_date"].dt.to_timestamp(), monthly["revenue"], marker="o")
        ax.set_title("Revenue Trends Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        return {
            "chart": _to_b64(fig),
            "data": [
                {"month_year": r.month_year, "revenue": float(r.revenue)}
                for r in monthly.itertuples()
            ],
        }

    def cancellation_rate(self):
        """Cancellation rate"""
        total = len(self.data)
        canceled = int(self.data["is_canceled"].sum())
        rate = canceled / total * 100 if total else 0.0

        fig = Figure(figsize=(8, 8))
        ax = fig.subplots()
        ax.pie(
            [total - canceled, canceled],
            labels=["Confirmed", "Canceled"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
        )
        ax.set_title("Booking Cancellation Rate")

        return {
            "chart": _to_b64(fig),
            "rate": float(rate),
            "total_bookings": int(total),
            "canceled_bookings": canceled,
        }

    def geographical_distribution(self):
        """Bookings per country"""
        counts = self.data["country"].value_counts().reset_index()
        counts.columns = ["country", "count"]

        fig = Figure(figsize=(12, 8))
        ax = fig.subplots()
        sns.barplot(x="count", y="country", data=counts.head(15), ax=ax)
        ax.set_title("Top 15 Countries by Number of Bookings")
        ax.set_xlabel("Number of Bookings")
        ax.set_ylabel("Country")
        fig.tight_layout()

        return {"chart": _to_b64(fig), "data": counts.to_dict("records")}

    def lead_time_distribution(self):
        """Lead time histogram and stats"""
        lead = self.data["lead_time_days"]

        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()
        sns.histplot(lead, bins=30, kde=len(lead) > 1, ax=ax)
        ax.set_title("Distribution of Booking Lead Time")
        ax.set_xlabel("Lead Time (Days)")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        return {
            "chart": _to_b64(fig),
            "stats": {
                "mean": float(lead.mean()),
                "median": float(lead.median()),
                "min": int(lead.min()),
                "max": int(lead.max()),
            },
        }

    def additional_analytics(self):
        """Guest composition (adults, children, babies)"""
        comp = self.data.groupby(["adults", "children", "babies"]).size().reset_index()
        comp.columns = ["adults", "children", "babies", "count"]
        comp = comp.sort_values("count", ascending=False).head(10)
        comp["label"] = comp.apply(
            lambda r: f"A:{int(r['adults'])}, C:{int(r['children'])}, B:{int(r['babies'])}", axis=1
        )

        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()
        sns.barplot(x="count", y="label", data=comp, ax=ax)
        ax.set_title("Top 10 Guest Compositions")
        ax.set_xlabel("Number of Bookings")
        ax.set_ylabel("Guest Composition (Adults, Children, Babies)")
        fig.tight_layout()

        return {
            "chart": _to_b64(fig),
            "data": [
                {
                    "adults": int(r.adults),
                    "children": int(r.children),
                    "babies": int(r.babies),
                    "count": int(r.count),
                }
                for r in comp.itertuples()
            ],
        }

    def generate_all_analytics(self):
        """Generate all analytics"""
        return {
            "revenue_trends": self.revenue_trends(),
            "cancellation_rate": self.cancellation_rate(),
            "geographical_distribution": self.geographical_distribution(),
            "lead_time_distribution": self.lead_time_distribution(),
            "additional_analytics": self.additional_analytics(),
        }
