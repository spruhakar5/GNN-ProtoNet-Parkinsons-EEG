"""
Episodic Few-Shot Training for GNN-ProtoNet.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    N_QUERY, N_EPISODES_TRAIN, LEARNING_RATE, N_TRAIN_EPOCHS,
    DEVICE, SCHEDULER_STEP, SCHEDULER_GAMMA, ENCODER_TYPE,
)
from models.proto_net import GNNProtoNet


def create_episode(subjects, k_shot, n_query=N_QUERY):
    """
    Create a single 2-way K-shot episode from subject graphs.

    Parameters
    ----------
    subjects : list of Subject (with .graphs populated)
    k_shot : int
    n_query : int

    Returns
    -------
    support_graphs, support_labels, query_graphs, query_labels
    """
    pd_subjects = [s for s in subjects if s.label == 1]
    hc_subjects = [s for s in subjects if s.label == 0]

    total_per_class = k_shot + n_query

    def sample_graphs(subj_list, n_total):
        all_graphs = []
        for s in subj_list:
            all_graphs.extend(s.graphs)
        if len(all_graphs) < n_total:
            # Sample with replacement if not enough graphs
            indices = np.random.choice(len(all_graphs), n_total, replace=True)
        else:
            indices = np.random.choice(len(all_graphs), n_total, replace=False)
        return [all_graphs[i] for i in indices]

    pd_graphs = sample_graphs(pd_subjects, total_per_class)
    hc_graphs = sample_graphs(hc_subjects, total_per_class)

    support_graphs = hc_graphs[:k_shot] + pd_graphs[:k_shot]
    query_graphs = hc_graphs[k_shot:k_shot+n_query] + pd_graphs[k_shot:k_shot+n_query]

    support_labels = torch.tensor([0]*k_shot + [1]*k_shot, dtype=torch.long)
    query_labels = torch.tensor([0]*n_query + [1]*n_query, dtype=torch.long)

    return support_graphs, support_labels, query_graphs, query_labels


def move_graphs_to_device(graphs, device):
    """Move list of PyG Data objects to device."""
    return [g.to(device) for g in graphs]


def train_one_fold(model, train_subjects, k_shot=5, n_query=N_QUERY,
                   n_episodes=N_EPISODES_TRAIN, n_epochs=N_TRAIN_EPOCHS,
                   lr=LEARNING_RATE):
    """
    Train model for one LOSO fold using episodic training.

    Parameters
    ----------
    model : GNNProtoNet
    train_subjects : list of Subject
    k_shot : int
    n_query : int
    n_episodes : int — episodes per epoch
    n_epochs : int — training epochs
    lr : float

    Returns
    -------
    model, losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA,
    )

    model.train()
    epoch_losses = []

    for epoch in range(n_epochs):
        running_loss = 0.0
        n_valid = 0

        for ep in range(n_episodes):
            try:
                s_graphs, s_labels, q_graphs, q_labels = create_episode(
                    train_subjects, k_shot, n_query
                )
            except (ValueError, IndexError):
                continue

            # Move to device
            s_graphs = move_graphs_to_device(s_graphs, DEVICE)
            q_graphs = move_graphs_to_device(q_graphs, DEVICE)
            s_labels = s_labels.to(DEVICE)
            q_labels = q_labels.to(DEVICE)

            # Forward
            log_probs, _ = model(s_graphs, s_labels, q_graphs)
            loss = F.nll_loss(log_probs, q_labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            n_valid += 1

        scheduler.step()
        avg_loss = running_loss / max(n_valid, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} — Loss: {avg_loss:.4f}")

    return model, epoch_losses


def create_fresh_model(encoder_type=ENCODER_TYPE):
    """Create a new GNNProtoNet model on DEVICE."""
    model = GNNProtoNet(encoder_type=encoder_type).to(DEVICE)
    return model
