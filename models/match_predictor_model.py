import torch
import torch.nn as nn

class MatchPredictor(nn.Module):
    def __init__(self, embedding_dim, num_numerical_features, output_dim):
        super(MatchPredictor, self).__init__()
        # Assuming 'embedding_dim' is unused in this modified version, consider reviewing its use if necessary.
        total_input_size = num_numerical_features

        # Additional layer for prioritized features
        self.fc_priority = nn.Linear(2, 300)  # Process the 1:3 range features specifically
        self.fc = nn.Linear(int(total_input_size-4) + 300, 2)  # Adjust input size to account for the transformation

    def forward(self, features):
        # Separate numerical features
        numerical_features = features[:, :6].float()

        # Process priority features (1:3 range) separately
        priority_features = numerical_features[:, 0:2]
        priority_features_transformed = self.fc_priority(priority_features)

        # Optionally, process other features as well
        other_features = torch.cat([numerical_features[:, 2:4]], dim=1)  # Excluding 1:3

        # Combine processed priority features with other features
        combined_features = torch.cat([
            priority_features_transformed,
            other_features
        ], dim=1)

        # Pass combined features through the main FC layer
        output = self.fc(combined_features)
        return output
