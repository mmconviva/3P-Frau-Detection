"""
generate_synthetic_transactions_small.py

Creates a synthetic transactions dataset for a small fraud detection task.

Output:
    data/transactions_small.csv
"""

import os
import numpy as np
import pandas as pd


def main(
    n_samples: int = 5000,
    random_seed: int = 7,
    output_path: str = "data/transactions_small.csv",
):
    rng = np.random.default_rng(random_seed)

    # Basic ID pools
    n_users = 500
    n_devices = 300
    n_merchants = 200

    user_ids = np.array([f"U{i:04d}" for i in range(n_users)])
    device_ids = np.array([f"D{i:04d}" for i in range(n_devices)])
    merchant_ids = np.array([f"M{i:04d}" for i in range(n_merchants)])

    merchant_categories = [
        "grocery",
        "electronics",
        "fashion",
        "travel",
        "restaurants",
        "gambling",
        "digital_services",
        "fuel",
    ]

    countries = [
        "US",
        "GB",
        "DE",
        "IN",
        "BR",
        "CA",
        "AU",
        "NG",
    ]

    user_sample = rng.choice(user_ids, size=n_samples)
    device_sample = rng.choice(device_ids, size=n_samples)
    merchant_sample = rng.choice(merchant_ids, size=n_samples)
    merchant_cat_sample = rng.choice(merchant_categories, size=n_samples)
    country_sample = rng.choice(countries, size=n_samples)

    # Numeric features
    amount = np.clip(rng.lognormal(mean=3.0, sigma=0.9, size=n_samples), 1, 4000)
    device_risk_score = np.clip(rng.beta(a=2, b=6, size=n_samples), 0, 1)
    user_credit_score = np.clip(rng.normal(loc=660, scale=70, size=n_samples), 300, 850)
    hour_of_day = rng.integers(0, 24, size=n_samples)

    past_7d_txn_count = np.clip(rng.poisson(lam=2.5, size=n_samples), 0, 30)
    past_7d_avg_amount = np.clip(
        amount * rng.uniform(0.3, 1.3, size=n_samples) + rng.normal(0, 15, size=n_samples),
        1,
        3500,
    )

    # Simple fraud logic to generate labels
    logit = -4.2  # base log-odds

    night = (hour_of_day >= 0) & (hour_of_day <= 4)
    high_amount = amount > 1000
    risky_country = np.isin(country_sample, ["BR", "NG"])
    risky_cat = np.isin(merchant_cat_sample, ["gambling", "digital_services"])
    high_risk_device = device_risk_score > 0.75
    very_good_credit = user_credit_score > 760

    logit = (
        logit
        + night * 0.9
        + high_amount * 0.7
        + risky_country * 0.7
        + risky_cat * 0.8
        + high_risk_device * 1.0
        - very_good_credit * 0.8
    )

    fraud_proba = 1 / (1 + np.exp(-logit))
    is_fraud = rng.binomial(n=1, p=fraud_proba)

    description_templates = [
        "Payment at {merchant_cat} store {merchant} in {country}",
        "POS purchase {merchant_cat} {merchant}",
        "Online transaction with {merchant} ({merchant_cat})",
        "Subscription renewal at {merchant}",
        "Chip card transaction at {merchant} in {country}",
        "Mobile payment to {merchant} ({merchant_cat})",
        "International transaction at {merchant} in {country}",
    ]

    desc_idx = rng.integers(0, len(description_templates), size=n_samples)
    descriptions = []
    for i in range(n_samples):
        tmpl = description_templates[desc_idx[i]]
        desc = tmpl.format(
            merchant_cat=merchant_cat_sample[i],
            merchant=merchant_sample[i],
            country=country_sample[i],
        )
        descriptions.append(desc)

    df = pd.DataFrame(
        {
            "amount": amount,
            "device_risk_score": device_risk_score,
            "user_credit_score": user_credit_score,
            "hour_of_day": hour_of_day,
            "past_7d_txn_count": past_7d_txn_count,
            "past_7d_avg_amount": past_7d_avg_amount,
            "merchant_category": merchant_cat_sample,
            "country": country_sample,
            "description": descriptions,
            "user_id": user_sample,
            "device_id": device_sample,
            "merchant_id": merchant_sample,
            "is_fraud": is_fraud.astype(int),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print("Fraud rate:", df["is_fraud"].mean())


if __name__ == "__main__":
    main()
