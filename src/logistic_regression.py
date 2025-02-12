import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ):
        """
        Train the logistic regression model using pre-processed features and labels.

        Args:
            features (torch.Tensor): The bag of words representations of the training examples.
            labels (torch.Tensor): The target labels.
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of iterations over the training dataset.

        Returns:
            None: The function updates the model weights in place.
        """
        # TODO: Implement gradient-descent algorithm to optimize logistic regression weights
        num_samples, num_features = features.shape
        features = torch.cat([features, torch.ones((num_samples, 1))], dim=1)  # Añadir sesgo

        if self._weights is None:
            self._weights = self.initialize_parameters(num_features + 1, self.random_state)

        for _ in range(epochs):
            predictions = self.predict_proba(features)
            errors = predictions.view(-1, 1) - labels.view(-1, 1)  # Column vectors
            gradient = torch.matmul(features.T, errors) / num_samples
            self._weights -= learning_rate * gradient            
    
    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        """
        Predict class labels for given examples based on a cutoff threshold.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.
            cutoff (float): The threshold for classifying a sample as positive. Defaults to 0.5.

        Returns:
            torch.Tensor: Predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(features)
        return (probabilities >= cutoff).float()

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probability of each sample belonging to the positive class using pre-processed features.
        
        Args:
            features (torch.Tensor): The bag of words representations of the input examples.
            
        Returns:
            torch.Tensor: A tensor of probabilities for each input sample being in the positive class.
            
        Raises:
            ValueError: If the model weights are not initialized (model not trained).
        """
        if self._weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")
        
        num_samples = features.shape[0]
        features = torch.cat([features, torch.ones((num_samples, 1))], dim=1)  # Agregar el sesgo

        logits = torch.matmul(features, self.weights.view(-1, 1))  # Matmul con pesos
        return self.sigmoid(logits).squeeze()



    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probability of each sample belonging to the positive class using pre-processed features.
        """
        if self._weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")

        # Verificar si features ya tiene el sesgo (misma cantidad de columnas que los pesos)
        if features.shape[1] == self.weights.shape[0] - 1:
            num_samples = features.shape[0]
            features = torch.cat([features, torch.ones((num_samples, 1))], dim=1)  # Agregar el sesgo solo si es necesario

        logits = torch.matmul(features, self.weights.view(-1, 1))  # Matmul con pesos
        return self.sigmoid(logits).squeeze()


    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        """
        Initialize the weights for logistic regression using a normal distribution.

        This function initializes the weights (and bias as the last element) with values drawn from a normal distribution.
        The use of random weights can help in breaking the symmetry and improve the convergence during training.

        Args:
            dim (int): The number of features (dimension) in the input data.
            random_state (int): A seed value for reproducibility of results.

        Returns:
            torch.Tensor: Initialized weights as a tensor with size (dim + 1,).
        """
        torch.manual_seed(random_state)
        return torch.randn((dim, 1), dtype=torch.float32) #Column vector


    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigmoid of z.

        This function applies the sigmoid function, which is defined as 1 / (1 + exp(-z)).
        It is used to map predictions to probabilities in logistic regression.

        Args:
            z (torch.Tensor): A tensor containing the linear combination of weights and features.

        Returns:
            torch.Tensor: The sigmoid of z.
        """
        return 1 / (1 + torch.exp(-z))

    @staticmethod
    def binary_cross_entropy_loss(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss.

        The binary cross-entropy loss is a common loss function for binary classification. It calculates the difference
        between the predicted probabilities and the actual labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities from the logistic regression model.
            targets (torch.Tensor): Actual labels (0 or 1).

        Returns:
            torch.Tensor: The computed binary cross-entropy loss.
        """
        epsilon = 1e-9  
        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
        return -torch.mean(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))

    @property
    def weights(self):
        """Get the weights of the logistic regression model."""
        return self._weights.view(-1)

    @weights.setter
    def weights(self, value):
        """Set the weights of the logistic regression model."""
        self._weights: torch.Tensor = value

