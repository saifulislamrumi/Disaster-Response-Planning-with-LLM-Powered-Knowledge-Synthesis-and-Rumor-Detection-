from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import datetime as dt


@dataclass
class Tweet:
    """Structured representation of a tweet from the dataset."""

    user_id: str
    tweet: str
    location: Optional[str]
    timestamp: Optional[dt.datetime]
    followers: Optional[int]

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation of the tweet."""
        return {
            "user_id": self.user_id,
            "tweet": self.tweet,
            "location": self.location,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "followers": self.followers,
        }


def parse_timestamp(value: str) -> Optional[dt.datetime]:
    """Attempt to parse a timestamp string into a ``datetime`` object.

    The dataset includes timestamp strings in a variety of formats (e.g.
    ``"Aug 22, 2024"``). This helper uses ``pd.to_datetime`` with
    ``errors='coerce'`` to gracefully handle parsing failures.

    Parameters
    ----------
    value: str
        The raw timestamp string from the dataset.

    Returns
    -------
    Optional[datetime.datetime]
        A ``datetime`` object if parsing succeeds, otherwise ``None``.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        # pandas parses many human-friendly date formats; coerce on error
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        # convert to naive datetime (drop timezone)
        return ts.to_pydatetime()
    except Exception:
        return None


def parse_followers(value: str | int | float | None) -> Optional[int]:
    """Convert a followers field to an integer if possible.

    Parameters
    ----------
    value: str | int | float | None
        The raw value from the dataset. Followers counts may be strings,
        numbers, or missing.

    Returns
    -------
    Optional[int]
        An integer follower count, or ``None`` if not parseable.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def load_dataset(file_path: str) -> List[Tweet]:
    """Load the input dataset of tweets from a TSV file.

    This function uses ``pandas`` to read a tab-separated file containing
    columns ``user_id``, ``tweet``, ``location``, ``timestamp`` and
    ``followers``. It applies simple cleaning such as stripping whitespace
    and converting timestamps and follower counts to appropriate types.

    Parameters
    ----------
    file_path: str
        Path to the TSV file.

    Returns
    -------
    List[Tweet]
        A list of :class:`Tweet` objects.
    """
    try:
        # Try reading the file with proper handling for CSV/TSV formats
        df = pd.read_csv(file_path, sep=None, engine="python")  # Let pandas detect the delimiter
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset '{file_path}': {exc}")

    # Standardize column names
    expected_cols = {"user_id", "tweet", "location", "timestamp", "followers"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # Clean data and create Tweet objects
    tweets: List[Tweet] = []
    for _, row in df.iterrows():
        user_id = str(row.get("user_id")) if pd.notna(row.get("user_id")) else ""
        tweet_text = str(row.get("tweet")) if pd.notna(row.get("tweet")) else ""
        location = str(row.get("location")) if pd.notna(row.get("location")) else None
        ts = parse_timestamp(row.get("timestamp"))
        followers = parse_followers(row.get("followers"))

        # Create Tweet object
        tweet = Tweet(user_id, tweet_text, location, ts, followers)
        tweets.append(tweet)

    return tweets
