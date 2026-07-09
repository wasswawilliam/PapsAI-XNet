"""
=============================================================
PapsAI XNet
=============================================================

Official implementation of:

PapsAI XNet:
An Explainable Hybrid Deep Learning Framework for
Automated Cervical Cytology Classification

Architecture

Input Image
      │
      ▼
Attention-enhanced CNN
      │
      ▼
32-dimensional Feature Embedding
      │
      ▼
Feature Normalization
      │
      ▼
ANFIS Classifier
      │
      ▼
Seven Cytology Classes

The CNN performs deep feature extraction while ANFIS
performs interpretable neuro-fuzzy reasoning.

=============================================================
"""

import torch
import torch.nn as nn

from .cnn_backbone import PapsAICNN
from .anfis import SugenoANFIS


class PapsAIXNet(nn.Module):
    """
    Complete PapsAI XNet Architecture.

    Parameters
    ----------
    input_channels : int
        Number of image channels.

    feature_dimension : int
        CNN embedding dimension.

    num_rules : int
        Number of optimized fuzzy rules.

    num_classes : int
        Number of cervical cytology classes.
    """

    def __init__(self,
                 input_channels=3,
                 feature_dimension=32,
                 num_rules=16,
                 num_classes=7):

        super().__init__()

        ####################################################
        # CNN Backbone
        ####################################################

        self.cnn = PapsAICNN()

        ####################################################
        # ANFIS Classifier
        ####################################################

        self.anfis = SugenoANFIS(
            input_dim=feature_dimension,
            num_rules=num_rules,
            num_classes=num_classes
        )

    ##########################################################

    def extract_features(self, x):
        """
        Extract normalized CNN embedding.

        Returns
        -------
        Tensor
            Shape:
            (batch_size, 32)
        """

        return self.cnn(x)

    ##########################################################

    def classify(self, embedding):

        """
        ANFIS reasoning.

        Input
        -----

        32-dimensional feature vector

        Output

        logits
        """

        return self.anfis(embedding)

    ##########################################################

    def forward(self,
                x,
                explain=False):

        """
        End-to-end forward propagation.

        Parameters
        ----------

        x

            Input image

        explain

            If True, returns intermediate
            explainability information.
        """

        ##################################################
        # CNN
        ##################################################

        embedding = self.extract_features(x)

        ##################################################
        # ANFIS
        ##################################################

        if explain:

            outputs = self.anfis(
                embedding,
                return_interpretability=True
            )

            outputs["embedding"] = embedding

            return outputs

        logits = self.classify(embedding)

        return logits

    ##########################################################

    @torch.no_grad()

    def predict(self,
                x):

        """
        Predict cervical cytology class.
        """

        logits = self.forward(x)

        prediction = torch.argmax(
            logits,
            dim=1
        )

        return prediction

    ##########################################################

    @torch.no_grad()

    def predict_proba(self,
                      x):

        """
        Return class probabilities.
        """

        logits = self.forward(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        return probabilities

    ##########################################################

    @torch.no_grad()

    def explain(self,
                x):

        """
        Return interpretable reasoning.

        Outputs include

        • CNN embedding

        • Membership functions

        • Rule strengths

        • Rule outputs

        • Final prediction
        """

        outputs = self.forward(
            x,
            explain=True
        )

        outputs["prediction"] = torch.argmax(
            outputs["logits"],
            dim=1
        )

        outputs["probabilities"] = torch.softmax(
            outputs["logits"],
            dim=1
        )

        return outputs

    ##########################################################

    def freeze_cnn(self):

        """
        Freeze CNN parameters.

        Used during Stage 2 ANFIS training.
        """

        for parameter in self.cnn.parameters():

            parameter.requires_grad = False

    ##########################################################

    def unfreeze_cnn(self):

        """
        Unfreeze CNN parameters.
        """

        for parameter in self.cnn.parameters():

            parameter.requires_grad = True

    ##########################################################

    def cnn_parameters(self):

        """
        Return CNN parameters.
        """

        return self.cnn.parameters()

    ##########################################################

    def anfis_parameters(self):

        """
        Return ANFIS parameters.
        """

        return self.anfis.parameters()

    ##########################################################

    def number_of_parameters(self):

        """
        Count trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    ##########################################################

    def summary(self):

        """
        Model summary.
        """

        print("=" * 60)
        print("               PapsAI XNet")
        print("=" * 60)

        print("CNN Backbone")
        print("   • 4 Convolution Blocks")
        print("   • SE Channel Attention")
        print("   • Global Average Pooling")
        print("   • 32-dimensional Embedding")

        print()

        print("ANFIS")
        print("   • Gaussian Membership Functions")
        print("   • 16 Fuzzy Rules")
        print("   • First-order Sugeno")
        print("   • 7 Output Classes")

        print()

        print(
            "Trainable Parameters:",
            self.number_of_parameters()
        )

        print("=" * 60)
