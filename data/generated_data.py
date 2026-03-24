import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "lead_id": range(n),
    "source": np.random.choice(["Google", "Facebook", "LinkedIn"], n),
    "clicks": np.random.randint(1, 20, n),
    "time_spent": np.random.randint(1, 10, n),
    "email_opened": np.random.choice([0, 1], n),
})

# simple logic for target
data["converted"] = (
    (data["clicks"] > 10) &
    (data["time_spent"] > 5) &
    (data["email_opened"] == 1)
).astype(int)

data.to_csv("crm_data.csv", index=False)
print("Dataset created!")