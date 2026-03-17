# ==============================
# Required libraries
# ==============================

import torch                      # Main PyTorch library
import torch.nn as nn             # Neural network layers
import torch.nn.functional as F   # Activation + loss functions
import pandas as pd               # Data handling
from sklearn.feature_extraction.text import TfidfVectorizer  # Text -> numeric
import random                     # Random sampling


# ==============================
# 1️⃣ Load Yelp-style dataset
# ==============================

# Assumes CSV has columns:
# review_text, stars, city
df = pd.read_csv("yelp_reviews.csv")

# Convert text to TF-IDF vectors (numerical features)
vectorizer = TfidfVectorizer(max_features=5000)

# Fit TF-IDF on all text and convert to dense numpy array
X_all = vectorizer.fit_transform(df["review_text"]).toarray()

# Extract star ratings
y_all = df["stars"].values

# Extract city names (used as tasks)
cities = df["city"].values

# Convert features to PyTorch tensor
X_all = torch.tensor(X_all, dtype=torch.float32)

# Convert ratings to float tensor and make shape (N,1)
y_all = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)


# ==============================
# 2️⃣ Organize tasks by city
# ==============================

tasks = {}   # dictionary: city -> its data

# Loop over each row in dataset
for i, city in enumerate(cities):

    # If city not already in dictionary, create entry
    if city not in tasks:
        tasks[city] = {"X": [], "y": []}

    # Append feature vector and label to that city
    tasks[city]["X"].append(X_all[i])
    tasks[city]["y"].append(y_all[i])


# Convert lists to stacked tensors
for city in tasks:
    tasks[city]["X"] = torch.stack(tasks[city]["X"])
    tasks[city]["y"] = torch.stack(tasks[city]["y"])

# List of all task names (cities)
task_names = list(tasks.keys())


# ==============================
# 3️⃣ Define Regression Model
# ==============================

class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, 128)

        # Output layer (single regression value)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, params=None):

        # If no adapted parameters given, use model’s own weights
        if params is None:
            x = F.relu(self.fc1(x))
            return self.fc2(x)

        # If adapted parameters provided (inner loop case),
        # manually apply linear layers with those parameters
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
        return x


# Helper function to extract model parameters into dictionary
def get_params(model):
    return {name: param for name, param in model.named_parameters()}


# ==============================
# 4️⃣ Inner Update (Support Set)
# ==============================

def inner_update(model, params, x_s, y_s, inner_lr):

    # Forward pass on support set using current parameters
    preds = model(x_s, params=params)

    # Compute regression loss (Mean Squared Error)
    loss = F.mse_loss(preds, y_s)

    # Compute gradients w.r.t. parameters
    grads = torch.autograd.grad(
        loss,
        params.values(),
        create_graph=False  # FO-MAML: do NOT compute second-order gradients
    )

    # Perform gradient descent update: theta' = theta - lr * grad
    new_params = {}
    for (name, p), g in zip(params.items(), grads):
        new_params[name] = p - inner_lr * g

    return new_params


# ==============================
# 5️⃣ Meta Training Loop
# ==============================

def meta_train(meta_iters=200,
               tasks_per_batch=4,
               k_support=16,
               k_query=32,
               inner_lr=0.01,
               meta_lr=1e-3):

    # Initialize model
    model = Regressor(input_dim=X_all.shape[1])

    # Optimizer for meta-parameters
    meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)

    for it in range(meta_iters):

        meta_opt.zero_grad()  # reset gradients
        meta_loss = 0         # accumulate loss over tasks

        # Randomly sample tasks (cities) for this batch
        sampled_tasks = random.sample(task_names, tasks_per_batch)

        for task in sampled_tasks:

            X = tasks[task]["X"]
            y = tasks[task]["y"]

            # Skip if not enough samples
            if len(X) < (k_support + k_query):
                continue

            # Shuffle indices
            idx = torch.randperm(len(X))

            # Split support set (few-shot)
            x_s = X[idx[:k_support]]
            y_s = y[idx[:k_support]]

            # Split query set (evaluation set)
            x_q = X[idx[k_support:k_support+k_query]]
            y_q = y[idx[k_support:k_support+k_query]]

            # Get current meta-parameters
            params = get_params(model)

            # ---- INNER LOOP (adapt to task) ----
            params = inner_update(model, params, x_s, y_s, inner_lr)

            # FO-MAML trick:
            # Detach adapted params to avoid second-order gradients
            params = {
                k: v.detach().requires_grad_(True)
                for k, v in params.items()
            }

            # ---- OUTER LOOP (meta objective) ----
            q_preds = model(x_q, params=params)
            q_loss = F.mse_loss(q_preds, y_q)

            meta_loss += q_loss

        # Average loss across tasks
        meta_loss /= tasks_per_batch

        # Backpropagate meta-loss
        meta_loss.backward()

        # Update meta-parameters
        meta_opt.step()

        if it % 50 == 0:
            print(f"Iter {it} | Meta Loss {meta_loss.item():.4f}")

    return model


# ==============================
# 6️⃣ Evaluate on New Task
# ==============================

def evaluate(model, city, inner_steps=3):

    X = tasks[city]["X"]
    y = tasks[city]["y"]

    idx = torch.randperm(len(X))

    # Few-shot support set
    x_s = X[idx[:16]]
    y_s = y[idx[:16]]

    # Query set
    x_q = X[idx[16:64]]
    y_q = y[idx[16:64]]

    # ---- Before adaptation ----
    with torch.no_grad():
        base_loss = F.mse_loss(model(x_q), y_q).item()

    # ---- Adaptation ----
    params = get_params(model)

    for _ in range(inner_steps):
        preds = model(x_s, params=params)
        loss = F.mse_loss(preds, y_s)
        grads = torch.autograd.grad(loss, params.values())
        params = {
            k: v - 0.01 * g
            for (k, v), g in zip(params.items(), grads)
        }

    # ---- After adaptation ----
    with torch.no_grad():
        adapted_loss = F.mse_loss(
            model(x_q, params=params),
            y_q
        ).item()

    print(f"{city} | Before: {base_loss:.4f} | After: {adapted_loss:.4f}")


# ==============================
# Run Training + Evaluation
# ==============================

model = meta_train()

# Example evaluation
evaluate(model, task_names[0])